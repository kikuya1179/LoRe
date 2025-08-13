import argparse
import os
import time
from typing import Optional, Dict, Any

import numpy as np
import torch
import torch.nn.functional as F
import gymnasium as gym
from minigrid.wrappers import RGBImgPartialObsWrapper, ImgObsWrapper

from .conf import load_config, Config
from .agents.dreamer_v3 import DreamerV3Agent, DreamerV3ActionSpec
from .llm.controller import LLMController
from .utils.health_monitor import HealthMonitor, create_health_dashboard
from .utils.metrics_aggregator import MetricsAggregator
from .utils.replay_buffer import ReplayBuffer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="DreamerV3 + MiniGrid")
    p.add_argument("--env_id", type=str, default=None)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--total_steps", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--ckpt", type=str, default=None)
    return p.parse_args()


def make_minigrid_env(env_id: str, image_size: int = 64, grayscale: bool = True):
    env = gym.make(env_id, render_mode="rgb_array")
    env = RGBImgPartialObsWrapper(env)
    env = ImgObsWrapper(env)
    return env


def preprocess_obs(obs: np.ndarray, image_size: int = 64, grayscale: bool = True) -> torch.Tensor:
    # obs: HxWx3 uint8 -> [1,C,H,W] float32 in [0,1]
    ten = torch.from_numpy(obs).float() / 255.0  # [H,W,3]
    ten = ten.permute(2, 0, 1).unsqueeze(0)  # [1,3,H,W]
    if grayscale:
        r, g, b = ten[:, 0:1], ten[:, 1:2], ten[:, 2:3]
        ten = 0.2989 * r + 0.5870 * g + 0.1140 * b  # [1,1,H,W]
    ten = F.interpolate(ten, size=(image_size, image_size), mode="area")
    return ten


def main() -> Optional[int]:
    args = parse_args()
    cfg = load_config()
    if args.env_id:
        cfg.env.id = args.env_id
    if args.device:
        cfg.train.device = args.device
    if args.total_steps is not None:
        cfg.train.total_steps = args.total_steps
    if args.seed is not None:
        cfg.train.seed = args.seed

    # Enhanced reproducibility setup (Phase 0 requirement)
    device = torch.device("cuda" if (cfg.train.device == "cuda" and torch.cuda.is_available()) else "cpu")
    
    # Set all random seeds for reproducibility
    torch.manual_seed(cfg.train.seed)
    np.random.seed(cfg.train.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed(cfg.train.seed)
        torch.cuda.manual_seed_all(cfg.train.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    print(f"[Reproducibility] Seed: {cfg.train.seed}, Device: {device}, Deterministic: True")

    env = make_minigrid_env(cfg.env.id, cfg.env.image_size, cfg.env.grayscale)
    action_spec = DreamerV3ActionSpec(n=env.action_space.n)
    n_actions = env.action_space.n
    action_counts = np.zeros(n_actions, dtype=int)
    recent_prob_max: list[float] = []
    recent_entropy: list[float] = []
    # Minigrid-specific skill stats
    skill_stats = {
        'steps': 0,
        'pickups': 0,
        'toggles': 0,
        'unlock_opens': 0,
        'has_key_steps': 0,
        'invalid': 0,
        'dist_ag_key': [],
        'dist_key_door': [],
        'dist_door_goal': [],
    }

    cfg.model.obs_channels = 1 if cfg.env.grayscale else 3
    agent = DreamerV3Agent(cfg.model, action_spec, device=device, lr=cfg.train.learning_rate,
                           gamma=cfg.train.gamma, entropy_coef=cfg.train.entropy_coef,
                           epsilon_greedy=getattr(cfg.train, 'epsilon_start', 0.3))
    
    # Set LoRe configuration on agent
    agent.lore_cfg = cfg.lore

    # Initialize LLM controller if enabled
    llm_controller = None
    if cfg.llm.enabled:
        llm_controller = LLMController(cfg.llm, env.action_space.n, cfg.lore)
        llm_controller.to(device)
        print(f"[LLM] Controller initialized with budget {cfg.llm.budget_total}")
        
        if cfg.lore.use_priornet:
            print("[LLM] PriorNet distillation enabled")
        
        # Temperature estimation with sample observations
        if hasattr(agent.ac, 'policy_logits'):
            print("[LLM] Estimating temperature for logits alignment...")
            sample_obs, sample_contexts, sample_logits = collect_temperature_samples(env, agent, cfg, device)
            if sample_obs:
                llm_controller.llm_adapter.estimate_temperature(sample_obs, sample_contexts, sample_logits)
                print(f"[LLM] Temperature estimated: {llm_controller.llm_adapter.temperature:.3f}")

    if args.ckpt and os.path.exists(args.ckpt):
        agent.load(args.ckpt, strict=False)
        
        # Load PriorNet if available
        if llm_controller and llm_controller.priornet_distiller:
            priornet_path = args.ckpt.replace('.pt', '_priornet.pt')
            if os.path.exists(priornet_path):
                llm_controller.load_priornet(priornet_path)
                print(f"[PriorNet] Model loaded from {priornet_path}")

    # Connect PriorNet to agent for imagination integration
    if llm_controller and llm_controller.priornet_distiller:
        agent.set_priornet_distiller(llm_controller.priornet_distiller)

    obs, info = env.reset(seed=cfg.train.seed)
    prev_struct = None
    total_steps = int(cfg.train.total_steps)

    # Training loop state
    episode_return = 0.0
    episode_count = 0
    episode_success = 0
    last_log_time = time.time()
    metrics_buffer = []
    
    # Health monitoring
    # Health monitor can be disabled entirely by config
    health_monitor = HealthMonitor() if getattr(cfg.log, 'enable_health_monitor', True) else None
    all_metrics_history = []
    aggregator = MetricsAggregator(env_id=cfg.env.id, seed=cfg.train.seed, delta_target=getattr(cfg.lore, 'delta_target', 0.1), save_dir=cfg.log_dir)
    # snapshot training config (for .txt)
    aggregator.set_training_config({
        'lr': cfg.train.learning_rate,
        'entropy_coef': cfg.train.entropy_coef,
        'batch_size': cfg.train.batch_size,
        'seq_len': cfg.train.seq_len,
        'updates_per_step': cfg.train.updates_per_step,
        'epsilon_start': cfg.train.epsilon_start,
        'epsilon_end': cfg.train.epsilon_end,
        'tau_start': cfg.train.tau_start,
        'tau_end': cfg.train.tau_end,
    })
    
    # Initialize replay buffer
    obs_shape = (1 if cfg.env.grayscale else 3, cfg.env.image_size, cfg.env.image_size)
    replay = ReplayBuffer(capacity=cfg.train.replay_capacity, obs_shape=obs_shape)

    for step in range(1, total_steps + 1):
        obs_t = preprocess_obs(obs, cfg.env.image_size, cfg.env.grayscale).to(device)
        
        # LLM integration for real environment interaction only
        llm_response = None
        if llm_controller and not cfg.lore.mix_in_imagination:
            # Step LLM controller (updates cooldown)
            llm_controller.step()
            
            # Get context for LLM
            context = extract_minigrid_context(obs, info)
            context_structured = extract_structured_state(env)
            if isinstance(context_structured, dict) and context_structured:
                context['structured'] = context_structured
            
            # Request LLM guidance based on current state and history
            with torch.no_grad():
                # Get current value and TD error estimates for triggering
                obs_encoded = agent.world.encoder(obs_t)
                h_seq, _ = agent.world.rssm(obs_encoded.unsqueeze(1), 
                                          torch.zeros(1, obs_t.size(0), obs_encoded.size(-1), device=device))
                h = h_seq.squeeze(1)
                current_value = agent.ac.value(h)
                
                # Simple TD error estimate (using reward prediction)
                _, _, reward_pred = agent.world(obs_t.unsqueeze(1))
                td_error = abs(float(reward_pred.squeeze()) - episode_return / max(step, 1))
            
            llm_response = llm_controller.request(
                obs_np=obs,
                context=context,
                reward=episode_return,
                done=False,  # Updated after step
                td_error=td_error,
                step=step
            )
        
        # Get action from agent with real-environment LLM prior mixing
        with torch.no_grad():
            # Base logits from world model
            z = agent.world.encoder(obs_t)
            h_seq, _ = agent.world.rssm(z.unsqueeze(1), torch.zeros(1, obs_t.size(0), z.size(-1), device=device))
            h = h_seq.squeeze(1)
            logits_wm = agent.ac.policy_logits(h)

            logits_mix = logits_wm
            if llm_controller and llm_response is not None and hasattr(llm_response, 'policy') and llm_response.policy:
                llm_logits = torch.tensor(llm_response.policy.get('logits', []), dtype=torch.float32, device=device).unsqueeze(0)
                if llm_logits.shape == logits_wm.shape:
                    # Compute β from uncertainty
                    values = agent.ac.value(h).unsqueeze(-1)
                    uncertainty = agent.uncertainty_gate.compute_uncertainty(logits_wm, values)
                    beta = agent.uncertainty_gate.compute_beta(uncertainty).unsqueeze(-1)
                    # stopgrad on LLM logits and mix
                    logits_mix = logits_wm + beta * llm_logits.detach()

            # Sample action to ensure exploration (softmax with epsilon)
            # Temperature annealing for sampling
            t = min(1.0, step / max(1, cfg.train.tau_anneal_steps))
            tau = (1 - t) * cfg.train.tau_start + t * cfg.train.tau_end
            probs = torch.softmax(logits_mix / max(tau, 1e-6), dim=-1)
            # diagnostics
            prob_max_t = float(probs.max(dim=-1).values.item())
            entropy_t = float(-(probs * torch.log(probs.clamp_min(1e-8))).sum(dim=-1).item())
            argmax_wm = int(torch.argmax(logits_wm, dim=-1).item())
            argmax_mix = int(torch.argmax(logits_mix, dim=-1).item())
            # Epsilon annealing
            t_eps = min(1.0, step / max(1, cfg.train.epsilon_anneal_steps))
            eps = (1 - t_eps) * cfg.train.epsilon_start + t_eps * cfg.train.epsilon_end
            if eps > 0:
                uniform = torch.ones_like(probs) / probs.size(-1)
                probs = (1 - eps) * probs + eps * uniform
            action = torch.multinomial(probs, num_samples=1)
            sampled = int(action.item())
            # expose instantaneous policy stats for aggregator
            metrics_instant = {
                'policy/entropy_inst': float(entropy_t),
                'policy/prob_max_inst': float(prob_max_t),
            }
        action_np = int(action.item())
        # update diagnostics
        action_counts[action_np] += 1
        recent_prob_max.append(prob_max_t)
        recent_entropy.append(entropy_t)
        if len(recent_prob_max) > 200:
            recent_prob_max.pop(0)
        if len(recent_entropy) > 200:
            recent_entropy.pop(0)
        
        # Environment step
        next_obs, reward, terminated, truncated, info = env.step(action_np)
        done = terminated or truncated
        
        # Update success tracking for LLM controller
        if llm_controller and done and reward > 0:
            llm_controller.last_success_step = step

        # Prepare training batch with LLM data
        observation = obs_t  # [1,C,H,W]
        next_obs_t = preprocess_obs(next_obs, cfg.env.image_size, cfg.env.grayscale).to(device)
        
        batch = _make_enhanced_batch(
            observation=observation, 
            action=action_np, 
            reward=float(reward), 
            done=done, 
            next_observation=next_obs_t,
            llm_response=llm_response,
            env_context=extract_minigrid_context(next_obs, info),
            device=device
        )

        # Write to replay with success tag
        replay.add(
            observation.squeeze(0).cpu().numpy(), action_np, float(reward), bool(done),
            next_obs_t.squeeze(0).cpu().numpy(), success=bool(done and reward > 0)
        )

        # Minigrid skill metrics update
        curr_struct = extract_structured_state(env)
        try:
            # steps
            skill_stats['steps'] += 1
            # has key ratio
            if curr_struct.get('has_key', False):
                skill_stats['has_key_steps'] += 1
            # pickup detection
            prev_has_key = prev_struct.get('has_key', False) if isinstance(prev_struct, dict) else False
            if action_np == 3 and (not prev_has_key) and curr_struct.get('has_key', False):
                skill_stats['pickups'] += 1
            # door toggle/unlock-open detection (any door state change)
            def _door_state(s):
                d = s.get('door') if isinstance(s, dict) else None
                return d.get('state') if isinstance(d, dict) else None
            prev_state = _door_state(prev_struct)
            curr_state = _door_state(curr_struct)
            if action_np == 5 and prev_state is not None and curr_state is not None:
                if prev_state != curr_state:
                    skill_stats['toggles'] += 1
                if prev_state in ('locked', 'closed') and curr_state == 'open':
                    skill_stats['unlock_opens'] += 1
            # invalid action heuristic
            def _agent_pos(s):
                a = s.get('agent') if isinstance(s, dict) else None
                return tuple(a.get('pos')) if (isinstance(a, dict) and a.get('pos') is not None) else None
            prev_pos = _agent_pos(prev_struct)
            curr_pos = _agent_pos(curr_struct)
            invalid = False
            if action_np == 2 and prev_pos is not None and curr_pos is not None and prev_pos == curr_pos:
                invalid = True  # forward into wall
            if action_np == 3 and not (action_np == 3 and (not prev_has_key) and curr_struct.get('has_key', False)):
                # pickup did not result in having key
                invalid = invalid or True
            if action_np == 5 and not (prev_state is not None and curr_state is not None and prev_state != curr_state):
                invalid = invalid or True
            if action_np == 4 and not prev_has_key:
                invalid = invalid or True
            if invalid:
                skill_stats['invalid'] += 1
            # distances (BFS over grid)
            def _bfs_distance(env_unwrapped, start, goal):
                if start is None or goal is None:
                    return None
                base = env_unwrapped
                grid = getattr(base, 'grid', None)
                if grid is None:
                    return None
                W = getattr(grid, 'width', 0)
                H = getattr(grid, 'height', 0)
                from collections import deque
                q = deque()
                seen = set()
                q.append((start[0], start[1], 0))
                seen.add((start[0], start[1]))
                def passable(x, y):
                    obj = grid.get(x, y)
                    if obj is None:
                        return True
                    t = getattr(obj, 'type', None)
                    if t == 'wall':
                        return False
                    if t == 'door':
                        return bool(getattr(obj, 'is_open', False))
                    return True
                while q:
                    x, y, d = q.popleft()
                    if (x, y) == (goal[0], goal[1]):
                        return d
                    for dx, dy in ((1,0),(-1,0),(0,1),(0,-1)):
                        nx, ny = x+dx, y+dy
                        if 0 <= nx < W and 0 <= ny < H and (nx, ny) not in seen and passable(nx, ny):
                            seen.add((nx, ny))
                            q.append((nx, ny, d+1))
                return None
            base = env.unwrapped
            # positions
            ag = _agent_pos(curr_struct)
            key = curr_struct.get('key', {}).get('pos') if isinstance(curr_struct.get('key', {}), dict) else None
            door = curr_struct.get('door', {}).get('pos') if isinstance(curr_struct.get('door', {}), dict) else None
            goal = curr_struct.get('goal', {}).get('pos') if isinstance(curr_struct.get('goal', {}), dict) else None
            d1 = _bfs_distance(base, ag, key)
            d2 = _bfs_distance(base, key, door)
            d3 = _bfs_distance(base, door, goal)
            if d1 is not None:
                skill_stats['dist_ag_key'].append(d1)
            if d2 is not None:
                skill_stats['dist_key_door'].append(d2)
            if d3 is not None:
                skill_stats['dist_door_goal'].append(d3)
        except Exception:
            pass
        prev_struct = curr_struct

        # Default metrics from online single-step (fallback)
        metrics = {"loss/policy": 0.0, "value/mean": 0.0, "policy/entropy": 0.0}

        # Warmup then train from replay with sequences
        if step >= cfg.train.warmup_steps:
            for _ in range(getattr(cfg.train, 'updates_per_step', 1)):
                batch_seq = replay.sample_sequences(cfg.train.batch_size, cfg.train.seq_len, device)
                metrics = agent.update_sequence(batch_seq)
        
        # Add episode/success metrics for health monitoring
        success_rate = episode_success / max(episode_count, 1) if episode_count > 0 else 0.0
        metrics['success_rate'] = success_rate
        metrics['episode_count'] = episode_count
        
        # Health monitoring
        if health_monitor is not None:
            alerts = health_monitor.update(
                step,
                metrics,
                warmup_steps=getattr(cfg.log, 'health_warmup_steps', 0),
                verbose=getattr(cfg.log, 'health_verbose', False),
            )
            if getattr(cfg.log, 'health_verbose', False):
                for alert in alerts:
                    print(f"[HEALTH] {alert}")
        
        # attach skill metrics to metrics dict for aggregator snapshot
        metrics.update({
            'skill/pickup_key_rate': float(skill_stats['pickups'] / max(1, skill_stats['steps'])),
            'skill/door_toggle_rate': float(skill_stats['toggles'] / max(1, skill_stats['steps'])),
            'skill/door_unlock_open_rate': float(skill_stats['unlock_opens'] / max(1, skill_stats['steps'])),
            'skill/has_key_ratio': float(skill_stats['has_key_steps'] / max(1, skill_stats['steps'])),
            'skill/invalid_action_ratio': float(skill_stats['invalid'] / max(1, skill_stats['steps'])),
            'skill/dist_med_ag_key': float(np.median(skill_stats['dist_ag_key'])) if skill_stats['dist_ag_key'] else -1.0,
            'skill/dist_med_key_door': float(np.median(skill_stats['dist_key_door'])) if skill_stats['dist_key_door'] else -1.0,
            'skill/dist_med_door_goal': float(np.median(skill_stats['dist_door_goal'])) if skill_stats['dist_door_goal'] else -1.0,
        })
        metrics.update(metrics_instant)
        metrics_buffer.append(metrics)
        all_metrics_history.append(metrics.copy())

        # Update aggregator step metrics (rolling SR is success_rate here)
        aggregator.record_step_metrics(step, metrics, rolling_success_rate=success_rate)

        episode_return += float(reward)
        obs = next_obs
        
        # Episode end handling
        if done:
            episode_count += 1
            if reward > 0:
                episode_success += 1
            
            success_rate = episode_success / episode_count if episode_count > 0 else 0.0
            print(f"step={step} episode={episode_count} return={episode_return:.2f} success_rate={success_rate:.3f}")
            aggregator.record_episode(step, episode_return, info.get('episode_length', 0) if isinstance(info, dict) else 0, success=(reward > 0))
            
            obs, info = env.reset()
            episode_return = 0.0
        
        # Periodic logging and health dashboard
        if step % cfg.log.metrics_interval == 0 or step == total_steps:
            log_enhanced_metrics(step, metrics_buffer, llm_controller, episode_count, episode_success,
                                 extra_diag={
                                     'prob_max_mean': float(np.mean(recent_prob_max)) if recent_prob_max else 0.0,
                                     'prob_max_std': float(np.std(recent_prob_max)) if recent_prob_max else 0.0,
                                     'entropy_mean': float(np.mean(recent_entropy)) if recent_entropy else 0.0,
                                     'entropy_std': float(np.std(recent_entropy)) if recent_entropy else 0.0,
                                     'action_hist': action_counts.tolist(),
                                     # skill metrics (rates over all steps so far)
                                     'pickup_key_rate': (skill_stats['pickups'] / max(1, skill_stats['steps'])),
                                     'door_toggle_rate': (skill_stats['toggles'] / max(1, skill_stats['steps'])),
                                     'door_unlock_open_rate': (skill_stats['unlock_opens'] / max(1, skill_stats['steps'])),
                                     'has_key_ratio': (skill_stats['has_key_steps'] / max(1, skill_stats['steps'])),
                                     'invalid_action_ratio': (skill_stats['invalid'] / max(1, skill_stats['steps'])),
                                     'dist_med_ag_key': (float(np.median(skill_stats['dist_ag_key'])) if skill_stats['dist_ag_key'] else -1.0),
                                     'dist_med_key_door': (float(np.median(skill_stats['dist_key_door'])) if skill_stats['dist_key_door'] else -1.0),
                                     'dist_med_door_goal': (float(np.median(skill_stats['dist_door_goal'])) if skill_stats['dist_door_goal'] else -1.0),
                                 })
            metrics_buffer = []
            last_log_time = time.time()
            
            # Show health dashboard every 500 steps
            if health_monitor is not None and getattr(cfg.log, 'health_verbose', False):
                if step % cfg.log.health_check_interval == 0:
                    dashboard = create_health_dashboard(health_monitor, all_metrics_history[-100:])
                    print("\n" + dashboard)

    # Save aggregated summaries
    summary = aggregator.compute_summary(total_steps)
    json_path = aggregator.save_json(summary, filename=f"summary_seed{cfg.train.seed}.json")
    txt_path = aggregator.save_txt(summary, filename=f"summary_seed{cfg.train.seed}.txt")
    print(f"[SUMMARY] Saved to {json_path} and {txt_path}")

    if args.ckpt:
        os.makedirs(os.path.dirname(args.ckpt), exist_ok=True)
        agent.save(args.ckpt)
        
        # Save PriorNet if available
        if llm_controller and llm_controller.priornet_distiller:
            priornet_path = args.ckpt.replace('.pt', '_priornet.pt')
            llm_controller.save_priornet(priornet_path)
            print(f"[PriorNet] Model saved to {priornet_path}")

    return 0


def _make_enhanced_batch(observation: torch.Tensor, action: int, reward: float, done: bool, 
                        next_observation: torch.Tensor, llm_response=None, env_context=None, device=None):
    """Create enhanced batch with LLM integration data."""
    class EnhancedBatch:
        def __init__(self):
            self.d = {
                "observation": observation,
                "action": torch.tensor([action], dtype=torch.long, device=observation.device),
                "reward": torch.tensor([reward], dtype=torch.float32, device=observation.device),
                "done": torch.tensor([1.0 if done else 0.0], dtype=torch.float32, device=observation.device),
                ("next", "observation"): next_observation,
            }
            
            # Add LLM integration data if available
            if llm_response is not None and hasattr(llm_response, 'policy') and llm_response.policy:
                logits = llm_response.policy.get('logits', [])
                if len(logits) > 0:
                    self.d["llm_prior_logits"] = torch.tensor(logits, dtype=torch.float32, device=observation.device).unsqueeze(0)
                
                mask = llm_response.mask if hasattr(llm_response, 'mask') else None
                if mask and len(mask) > 0:
                    self.d["llm_mask"] = torch.tensor(mask, dtype=torch.float32, device=observation.device).unsqueeze(0)
                
                features = llm_response.features if hasattr(llm_response, 'features') else None
                if features and len(features) > 0:
                    self.d["llm_features"] = torch.tensor(features, dtype=torch.float32, device=observation.device).unsqueeze(0)
            
        def get(self, k):
            return self.d.get(k, torch.empty(0))
        
        def keys(self, *args, **kwargs):
            return self.d.keys()
    
    return EnhancedBatch()


def extract_minigrid_context(obs: np.ndarray, info: Dict[str, Any]) -> Dict[str, Any]:
    """Extract MiniGrid-specific context for LLM."""
    return {
        'mission': info.get('mission', ''),
        'remaining_steps': info.get('remaining_steps', 0),
        'obs_shape': obs.shape if obs is not None else (0, 0, 0)
    }


def extract_structured_state(env) -> Dict[str, Any]:
    """Build a lightweight symbolic summary from MiniGrid env (for LLM).

    Uses env.unwrapped to access grid and agent metadata even under wrappers.
    """
    try:
        base = env.unwrapped
        grid = getattr(base, 'grid', None)
        agent_pos = tuple(getattr(base, 'agent_pos', (None, None)))
        agent_dir_idx = int(getattr(base, 'agent_dir', -1))
        dir_map = {0: 'E', 1: 'S', 2: 'W', 3: 'N'}
        agent_dir = dir_map.get(agent_dir_idx, '?')
        has_key = bool(getattr(base, 'carrying', None) is not None and getattr(getattr(base, 'carrying', None), 'type', None) == 'key')

        grid_w = getattr(grid, 'width', 0) if grid is not None else 0
        grid_h = getattr(grid, 'height', 0) if grid is not None else 0
        walls = []
        key_pos = None
        door_pos = None
        door_state = None
        goal_pos = None

        if grid is not None:
            for y in range(grid_h):
                for x in range(grid_w):
                    obj = grid.get(x, y)
                    if obj is None:
                        continue
                    t = getattr(obj, 'type', None)
                    if t == 'wall':
                        walls.append([x, y])
                    elif t == 'key' and key_pos is None:
                        key_pos = [x, y]
                    elif t == 'door' and door_pos is None:
                        door_pos = [x, y]
                        is_open = bool(getattr(obj, 'is_open', False))
                        is_locked = bool(getattr(obj, 'is_locked', False))
                        door_state = 'open' if is_open else ('locked' if is_locked else 'closed')
                    elif t == 'goal' and goal_pos is None:
                        goal_pos = [x, y]

        legal_actions = [
            'turn_left', 'turn_right', 'forward', 'pickup', 'drop', 'toggle', 'done'
        ]

        return {
            'grid_size': [grid_w, grid_h],
            'agent': {'pos': list(agent_pos) if None not in agent_pos else None, 'dir': agent_dir},
            'key': {'pos': key_pos, 'available': not has_key},
            'door': {'pos': door_pos, 'state': door_state},
            'goal': {'pos': goal_pos},
            'walls': walls,
            'has_key': has_key,
            'legal_actions': legal_actions,
        }
    except Exception:
        return {}


def collect_temperature_samples(env, agent, cfg, device, num_samples: int = 20):
    """Collect sample observations and corresponding agent logits for temperature estimation."""
    sample_obs = []
    sample_contexts = []
    sample_logits = []
    
    # Reset environment
    obs, info = env.reset()
    
    for _ in range(num_samples):
        obs_t = preprocess_obs(obs, cfg.env.image_size, cfg.env.grayscale).to(device)
        
        with torch.no_grad():
            # Get agent's logits
            z = agent.world.encoder(obs_t)
            h, _ = agent.world.rssm(z.unsqueeze(1), torch.zeros(1, obs_t.size(0), z.size(-1), device=device))
            h = h.squeeze(1)
            logits = agent.ac.policy_logits(h)
            
            sample_obs.append(obs.copy())
            sample_contexts.append(extract_minigrid_context(obs, info))
            sample_logits.append(logits.cpu().numpy().squeeze())
        
        # Take random action to get variety
        action = env.action_space.sample()
        obs, _, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            obs, info = env.reset()
    
    return sample_obs, sample_contexts, sample_logits


def log_enhanced_metrics(step: int, metrics_buffer: list, llm_controller, episode_count: int, episode_success: int, extra_diag: dict | None = None):
    """Log enhanced metrics including LLM controller statistics."""
    if not metrics_buffer:
        return
    
    # Aggregate metrics
    avg_metrics = {}
    for key in metrics_buffer[0].keys():
        values = [m[key] for m in metrics_buffer if key in m]
        if values:
            avg_metrics[key] = sum(values) / len(values)
    
    # Basic training metrics + world model diagnostics
    line = (f"[{step:6d}] Loss: {avg_metrics.get('loss/policy', 0):.4f} "
            f"Value: {avg_metrics.get('value/mean', 0):.3f} "
            f"Entropy: {avg_metrics.get('policy/entropy', 0):.3f} "
            f"| world_psnr={avg_metrics.get('world/psnr_db', 0):.2f}dB "
            f"recon_mse={avg_metrics.get('world/recon_mse', 0):.4f} "
            f"reward_mae={avg_metrics.get('reward/mae', 0):.4f} "
            f"val_ev={avg_metrics.get('value/explained_variance', 0):.3f} "
            f"rew_ev={avg_metrics.get('reward/explained_variance', 0):.3f}")
    if extra_diag:
        line += (f" | prob_max(mean±std)={extra_diag.get('prob_max_mean',0):.3f}±{extra_diag.get('prob_max_std',0):.3f}"
                 f" entropy(mean±std)={extra_diag.get('entropy_mean',0):.3f}±{extra_diag.get('entropy_std',0):.3f}"
                 f" | pickup={extra_diag.get('pickup_key_rate',0):.3f} toggle={extra_diag.get('door_toggle_rate',0):.3f}"
                 f" unlockOpen={extra_diag.get('door_unlock_open_rate',0):.3f} hasKey={extra_diag.get('has_key_ratio',0):.3f} invalid={extra_diag.get('invalid_action_ratio',0):.3f}"
                 f" | d_ag->key={extra_diag.get('dist_med_ag_key',-1):.1f} d_key->door={extra_diag.get('dist_med_key_door',-1):.1f} d_door->goal={extra_diag.get('dist_med_door_goal',-1):.1f}")
    print(line)
    
    # LLM controller metrics
    if llm_controller:
        llm_stats = llm_controller.get_statistics()
        priornet_info = ""
        if 'priornet_api_calls_saved' in llm_stats:
            priornet_info = f" priornet_saved={llm_stats['priornet_api_calls_saved']}"
        
        print(f"[{step:6d}] LLM: calls={llm_stats['llm_calls_used']}/{llm_stats['llm_calls_used'] + llm_stats['llm_budget_remaining']} "
              f"cache_hit={llm_stats['llm_cache_hit_rate']:.3f} "
              f"cooldown={llm_stats['llm_cooldown_remaining']}{priornet_info}")
    
    # Success metrics
    success_rate = episode_success / max(episode_count, 1)
    if extra_diag and 'action_hist' in extra_diag:
        hist = extra_diag['action_hist']
        hist_str = ", ".join(str(v) for v in hist)
        print(f"[{step:6d}] Episodes: {episode_count} Success rate: {success_rate:.3f} | action_hist: [{hist_str}]")
    else:
        print(f"[{step:6d}] Episodes: {episode_count} Success rate: {success_rate:.3f}")


def _make_batch(observation: torch.Tensor, action: int, reward: float, done: bool, next_observation: torch.Tensor):
    """Legacy batch creation for backward compatibility."""
    return _make_enhanced_batch(observation, action, reward, done, next_observation)


if __name__ == "__main__":
    raise SystemExit(main())


