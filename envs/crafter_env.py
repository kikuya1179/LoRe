from dataclasses import asdict
from typing import Optional
from types import SimpleNamespace


def _patch_numpy_for_gym() -> None:
    """Gym 旧版が NumPy 2.x で削除された np.bool8 を参照して落ちる問題の暫定回避。

    - np.bool8 が無ければ np.bool_ にエイリアスする。
    """
    try:
        import numpy as _np

        if not hasattr(_np, "bool8"):
            _np.bool8 = _np.bool_  # type: ignore[attr-defined]
    except Exception:
        pass


class _GymToGymnasiumV26:
    """Minimal wrapper to adapt Gym (v0) API to Gymnasium v0.26+ API.

    - reset() -> (obs, info)
    - step(a) -> (obs, reward, terminated, truncated, info)
    """

    def __init__(self, env):
        self._env = env

    @property
    def action_space(self):
        return self._env.action_space

    @property
    def observation_space(self):
        return self._env.observation_space

    @property
    def unwrapped(self):
        return getattr(self._env, "unwrapped", self._env)

    def reset(self, *, seed=None, options=None, **kwargs):
        if hasattr(self._env, "reset"):
            if seed is not None and hasattr(self._env, "seed"):
                try:
                    self._env.seed(seed)
                except Exception:
                    pass
            # Ignore extra kwargs such as env_ids passed by vector wrappers
            try:
                obs = self._env.reset()
            except TypeError:
                # Some gym envs may support (seed=, options=)
                try:
                    obs = self._env.reset(seed=seed)
                except Exception:
                    obs = self._env.reset()
            return obs, {}
        raise RuntimeError("Underlying env has no reset()")

    def step(self, action, **kwargs):
        obs, reward, done, info = self._env.step(action)
        terminated = bool(done)
        truncated = bool(info.get("TimeLimit.truncated", False))
        return obs, reward, terminated, truncated, info

    def render(self, *args, **kwargs):
        return self._env.render(*args, **kwargs)

    def close(self):
        return self._env.close()


class _ActionRepeatGymnasium:
    """Simple action-repeat wrapper for Gymnasium-like API envs.

    step(a) を n 回繰り返し、報酬を合算・早期終端に対応する。
    """

    def __init__(self, env, repeats: int = 1):
        self._env = env
        self._repeats = max(1, int(repeats))

    @property
    def action_space(self):
        return getattr(self._env, "action_space", None)

    @property
    def observation_space(self):
        return getattr(self._env, "observation_space", None)

    @property
    def unwrapped(self):
        return getattr(self._env, "unwrapped", self._env)

    def reset(self, *args, **kwargs):
        return self._env.reset(*args, **kwargs)

    def step(self, action, **kwargs):
        total_reward = 0.0
        info_agg = {}
        obs = None
        terminated = False
        truncated = False
        for _ in range(self._repeats):
            obs, reward, term, trunc, info = self._env.step(action, **kwargs)
            total_reward += float(reward)
            terminated = bool(term)
            truncated = bool(trunc)
            # merge minimal info (last wins)
            info_agg = info
            if terminated or truncated:
                break
        return obs, total_reward, terminated, truncated, info_agg

    def render(self, *args, **kwargs):
        return self._env.render(*args, **kwargs)

    def close(self):
        return self._env.close()


def make_crafter_env(env_cfg) -> "torchrl.envs.EnvBase":  # type: ignore[name-defined]
    """Crafter 環境を TorchRL の `TransformedEnv` として構築する。

    設定 `env_cfg` に応じて以下を適用:
    - ToTensorImage (常時)
    - GrayScale（任意）
    - Resize（任意）
    - CatFrames（フレームスタック任意）
    """
    # 依存
    try:
        _patch_numpy_for_gym()
        import gym  # Crafter は Gym 登録を提供
    except Exception as e:
        raise RuntimeError("Gym is required. `pip install gym==0.26.2`") from e

    try:
        import crafter  # noqa: F401 登録を確実に起こす
    except Exception as e:
        raise RuntimeError("Crafter is required. `pip install crafter`") from e

    try:
        from torchrl.envs import GymWrapper
        from torchrl.envs.transforms import (
            TransformedEnv,
            Compose,
            ToTensorImage,
            GrayScale,
            Resize,
            CatFrames,
            RenameTransform,
        )
    except Exception as e:
        raise RuntimeError("TorchRL is required. `pip install torchrl`") from e

    # allow dict config
    if isinstance(env_cfg, dict):
        env_cfg = SimpleNamespace(**env_cfg)

    # Gym 環境生成
    env_id = getattr(env_cfg, "name", "CrafterReward-v1")
    try:
        base_gym = gym.make(env_id)
    except Exception:
        alt = "CrafterNoReward-v1" if env_id == "CrafterReward-v1" else "CrafterReward-v1"
        base_gym = gym.make(alt)

    # Gym の自動ラッパ（TimeLimit 等）を外すことで旧API(4タプル)に揃える
    try:
        base_gym = base_gym.unwrapped
    except Exception:
        pass

    # Gym -> Gymnasium API 変換（簡易アダプタ）
    base_gym = _GymToGymnasiumV26(base_gym)

    # Action repeat (CPU側の env.step 回数を削減)
    frame_skip = int(getattr(env_cfg, "frame_skip", 1))
    action_repeat = int(getattr(env_cfg, "action_repeat", frame_skip))
    if action_repeat and action_repeat > 1:
        base_gym = _ActionRepeatGymnasium(base_gym, repeats=action_repeat)

    # TorchRL の GymWrapper
    base = GymWrapper(base_gym)

    # 観測キーを特定（'pixels' 優先、無ければ 'observation'）
    try:
        keys = set(base.observation_spec.keys(True, True))
    except Exception:
        keys = {"observation"}
    obs_key = "pixels" if "pixels" in keys else "observation"

    # 前処理パイプライン（設定駆動）
    tfs = [ToTensorImage(in_keys=[obs_key])]

    if getattr(env_cfg, "grayscale", True):
        tfs.append(GrayScale(in_keys=[obs_key]))

    img_size = int(getattr(env_cfg, "image_size", 64))
    if img_size and img_size > 0:
        tfs.append(Resize(img_size, img_size, in_keys=[obs_key]))

    frame_stack = int(getattr(env_cfg, "frame_stack", 1))
    if frame_stack and frame_stack > 1:
        # channels-first を前提としてフレーム連結（C,H,W -> C*N,H,W なので dim=-3）
        tfs.append(CatFrames(N=frame_stack, dim=-3, in_keys=[obs_key]))

    # 最後にキーを 'observation' に正規化
    if obs_key != "observation":
        tfs.append(RenameTransform(in_keys=[obs_key], out_keys=["observation"]))

    env_single = TransformedEnv(base, Compose(*tfs))

    # Vectorize if requested
    try:
        n_envs = int(getattr(env_cfg, "num_envs", 1))
    except Exception:
        n_envs = 1

    if n_envs <= 1:
        return env_single

    try:
        from torchrl.envs import ParallelEnv
    except Exception as e:  # pragma: no cover
        # Fallback to single env if ParallelEnv is not available
        return env_single

    def _make_one():
        # Recreate a fresh single env for each worker (share same cfg)
        # Inline build to avoid recursion and attribute issues
        try:
            _patch_numpy_for_gym()
            import gym
            import crafter  # noqa: F401
            from torchrl.envs import GymWrapper
            from torchrl.envs.transforms import (
                TransformedEnv, Compose, ToTensorImage, GrayScale, Resize, CatFrames, RenameTransform
            )
        except Exception as e:
            raise e
        base_gym = gym.make(getattr(env_cfg, 'name', 'CrafterReward-v1'))
        try:
            base_gym = base_gym.unwrapped
        except Exception:
            pass
        base_gym = _GymToGymnasiumV26(base_gym)
        # action repeat
        frame_skip = int(getattr(env_cfg, 'frame_skip', 1))
        action_repeat = int(getattr(env_cfg, 'action_repeat', frame_skip))
        if action_repeat and action_repeat > 1:
            base_gym = _ActionRepeatGymnasium(base_gym, repeats=action_repeat)
        base = GymWrapper(base_gym)
        try:
            keys = set(base.observation_spec.keys(True, True))
        except Exception:
            keys = {"observation"}
        obs_key = "pixels" if "pixels" in keys else "observation"
        tfs = [ToTensorImage(in_keys=[obs_key])]
        if getattr(env_cfg, 'grayscale', True):
            tfs.append(GrayScale(in_keys=[obs_key]))
        img_size = int(getattr(env_cfg, 'image_size', 64))
        if img_size and img_size > 0:
            tfs.append(Resize(img_size, img_size, in_keys=[obs_key]))
        frame_stack = int(getattr(env_cfg, 'frame_stack', 1))
        if frame_stack and frame_stack > 1:
            tfs.append(CatFrames(N=frame_stack, dim=-3, in_keys=[obs_key]))
        if obs_key != "observation":
            tfs.append(RenameTransform(in_keys=[obs_key], out_keys=["observation"]))
        return TransformedEnv(base, Compose(*tfs))

    # Build n_envs workers
    env = ParallelEnv(n_envs, _make_one)
    # TorchRL may pass env_ids to reset(); wrap reset to swallow extra kwargs
    try:
        _orig_reset = env.reset
        def _safe_reset(*args, **kwargs):  # type: ignore
            kwargs.pop('env_ids', None)
            kwargs.pop('reset_envs', None)
            kwargs.pop('mask', None)
            return _orig_reset(*args, **kwargs)
        env.reset = _safe_reset  # type: ignore
    except Exception:
        pass
    return env

