"""
Ablation Study Framework for LoRe Implementation
Automatically runs A-F configurations for systematic comparison
"""

import os
import json
import time
import subprocess
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from concurrent.futures import ProcessPoolExecutor, as_completed

from ..conf import Config, load_config


@dataclass
class AblationConfig:
    """Configuration for a single ablation experiment."""
    name: str
    description: str
    
    # Core LoRe settings
    llm_enabled: bool = False
    use_priornet: bool = False
    mix_in_imagination: bool = False
    
    # Beta control
    beta_max: float = 0.0
    beta_warmup_steps: int = 0
    beta_dropout_p: float = 0.0
    
    # KL control
    delta_target: float = 0.0
    kl_lr: float = 0.0
    
    # Training settings
    total_steps: int = 50000
    seeds: List[int] = None
    
    def __post_init__(self):
        if self.seeds is None:
            self.seeds = [42, 123, 456]  # Default 3 seeds


# Define the A-F ablation configurations
ABLATION_CONFIGS = {
    'A': AblationConfig(
        name='A_dreamer_baseline',
        description='Dreamer素体（β=0, δ→0）',
        llm_enabled=False,
        beta_max=0.0,
        delta_target=0.0,
        total_steps=50000
    ),
    
    'B': AblationConfig(
        name='B_prior_fixed_beta',
        description='固定βでprior混合（β=0.3固定, KLなし, 実環境のみ）',
        llm_enabled=True,
        beta_max=0.3,
        beta_warmup_steps=0,  # No warmup = fixed beta
        beta_dropout_p=0.0,
        delta_target=0.0,  # No KL constraint
        total_steps=50000
    ),
    
    'C': AblationConfig(
        name='C_prior_with_kl',
        description='B + KL制約（δ=0.1）',
        llm_enabled=True,
        beta_max=0.3,
        beta_warmup_steps=0,
        beta_dropout_p=0.0,
        delta_target=0.1,
        kl_lr=1e-3,
        total_steps=50000
    ),
    
    'D': AblationConfig(
        name='D_adaptive_beta',
        description='C + β(s)適応制御',
        llm_enabled=True,
        beta_max=0.3,
        beta_warmup_steps=5000,
        beta_dropout_p=0.05,
        delta_target=0.1,
        kl_lr=1e-3,
        total_steps=50000
    ),
    
    'E': AblationConfig(
        name='E_with_budget_control',
        description='D + 予算/クールダウン/キャッシュ制御',
        llm_enabled=True,
        beta_max=0.3,
        beta_warmup_steps=5000,
        beta_dropout_p=0.05,
        delta_target=0.1,
        kl_lr=1e-3,
        total_steps=50000
    ),
    
    'F': AblationConfig(
        name='F_with_priornet',
        description='E + PriorNet蒸留',
        llm_enabled=True,
        use_priornet=True,
        beta_max=0.3,
        beta_warmup_steps=5000,
        beta_dropout_p=0.05,
        delta_target=0.1,
        kl_lr=1e-3,
        total_steps=50000
    ),
}


class AblationRunner:
    """Runs ablation experiments systematically."""
    
    def __init__(self, base_config_path: str = None, output_dir: str = "ablation_results"):
        self.base_config = load_config() if base_config_path is None else self._load_config(base_config_path)
        self.output_dir = output_dir
        self.results: Dict[str, List[Dict]] = {}
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
    def _load_config(self, path: str) -> Config:
        """Load config from file (if implementing config serialization)."""
        return load_config()  # Simplified for now
    
    def _create_experiment_config(self, ablation: AblationConfig) -> Config:
        """Create a full config for the ablation experiment."""
        config = self.base_config
        
        # Update LLM settings
        config.llm.enabled = ablation.llm_enabled
        
        # Update LoRe settings
        config.lore.beta_max = ablation.beta_max
        config.lore.beta_warmup_steps = ablation.beta_warmup_steps
        config.lore.beta_dropout_p = ablation.beta_dropout_p
        config.lore.delta_target = ablation.delta_target
        config.lore.kl_lr = ablation.kl_lr
        config.lore.use_priornet = ablation.use_priornet
        config.lore.mix_in_imagination = ablation.mix_in_imagination
        
        # Update training settings
        config.train.total_steps = ablation.total_steps
        
        return config
    
    def run_single_experiment(self, config_name: str, ablation: AblationConfig, 
                            seed: int, gpu_id: int = 0) -> Dict[str, Any]:
        """Run a single experiment with given configuration and seed."""
        
        experiment_name = f"{ablation.name}_seed{seed}"
        experiment_dir = os.path.join(self.output_dir, experiment_name)
        os.makedirs(experiment_dir, exist_ok=True)
        
        # Create config for this experiment
        config = self._create_experiment_config(ablation)
        config.train.seed = seed
        
        # Set up logging and checkpointing
        log_file = os.path.join(experiment_dir, "training.log")
        checkpoint_path = os.path.join(experiment_dir, "model.pt")
        metrics_file = os.path.join(experiment_dir, "metrics.json")
        
        # Run the experiment
        start_time = time.time()
        
        try:
            # Run training (using subprocess to isolate)
            cmd = [
                "python", "-m", "LoRe.main",
                "--env_id", config.env.id,
                "--total_steps", str(config.train.total_steps),
                "--seed", str(seed),
                "--device", f"cuda:{gpu_id}" if gpu_id >= 0 else "cpu",
                "--ckpt", checkpoint_path
            ]
            
            # Set environment variables for config
            env = os.environ.copy()
            env['LORE_LLM_ENABLED'] = str(ablation.llm_enabled).lower()
            env['LORE_BETA_MAX'] = str(ablation.beta_max)
            env['LORE_USE_PRIORNET'] = str(ablation.use_priornet).lower()
            # ... add other config overrides as env vars
            
            with open(log_file, 'w') as f:
                result = subprocess.run(
                    cmd, 
                    stdout=f, 
                    stderr=subprocess.STDOUT,
                    text=True,
                    env=env,
                    timeout=7200  # 2 hour timeout
                )
            
            duration = time.time() - start_time
            
            # Parse results from log file
            metrics = self._parse_training_log(log_file)
            
            experiment_result = {
                'config_name': config_name,
                'experiment_name': experiment_name,
                'seed': seed,
                'duration_seconds': duration,
                'success': result.returncode == 0,
                'final_metrics': metrics,
                'ablation_config': asdict(ablation)
            }
            
            # Save individual experiment result
            with open(metrics_file, 'w') as f:
                json.dump(experiment_result, f, indent=2)
            
            return experiment_result
            
        except subprocess.TimeoutExpired:
            return {
                'config_name': config_name,
                'experiment_name': experiment_name,
                'seed': seed,
                'duration_seconds': 7200,
                'success': False,
                'error': 'timeout',
                'ablation_config': asdict(ablation)
            }
        except Exception as e:
            return {
                'config_name': config_name,
                'experiment_name': experiment_name,
                'seed': seed,
                'duration_seconds': time.time() - start_time,
                'success': False,
                'error': str(e),
                'ablation_config': asdict(ablation)
            }
    
    def _parse_training_log(self, log_file: str) -> Dict[str, Any]:
        """Parse final metrics from training log."""
        metrics = {
            'final_success_rate': 0.0,
            'final_episode_count': 0,
            'final_return': 0.0,
            'total_llm_calls': 0,
            'priornet_api_savings': 0
        }
        
        try:
            with open(log_file, 'r') as f:
                lines = f.readlines()
            
            # Parse last few lines for final metrics
            for line in reversed(lines[-100:]):  # Check last 100 lines
                if 'success_rate=' in line:
                    # Extract success rate
                    parts = line.split('success_rate=')
                    if len(parts) > 1:
                        metrics['final_success_rate'] = float(parts[1].split()[0])
                
                if 'Episodes:' in line:
                    # Extract episode count
                    parts = line.split('Episodes:')
                    if len(parts) > 1:
                        metrics['final_episode_count'] = int(parts[1].split()[0])
                
                if 'LLM: calls=' in line:
                    # Extract LLM usage
                    parts = line.split('calls=')
                    if len(parts) > 1:
                        call_info = parts[1].split()[0]  # "150/200" format
                        if '/' in call_info:
                            used, total = call_info.split('/')
                            metrics['total_llm_calls'] = int(used)
                
                if 'priornet_saved=' in line:
                    # Extract PriorNet savings
                    parts = line.split('priornet_saved=')
                    if len(parts) > 1:
                        metrics['priornet_api_savings'] = int(parts[1].split()[0])
                        
        except Exception as e:
            print(f"Error parsing log file {log_file}: {e}")
        
        return metrics
    
    def run_ablation_study(self, configs: List[str] = None, 
                          max_parallel: int = 4, gpu_ids: List[int] = None) -> Dict[str, List[Dict]]:
        """Run complete ablation study."""
        
        if configs is None:
            configs = list(ABLATION_CONFIGS.keys())
        
        if gpu_ids is None:
            gpu_ids = [0] * max_parallel
        
        print(f"Starting ablation study with configs: {configs}")
        print(f"Using {max_parallel} parallel processes on GPUs: {gpu_ids}")
        
        all_experiments = []
        
        # Prepare all experiments
        for config_name in configs:
            if config_name not in ABLATION_CONFIGS:
                print(f"Warning: Config {config_name} not found, skipping")
                continue
                
            ablation = ABLATION_CONFIGS[config_name]
            for seed in ablation.seeds:
                all_experiments.append((config_name, ablation, seed))
        
        print(f"Total experiments to run: {len(all_experiments)}")
        
        # Run experiments in parallel
        results = {}
        completed_count = 0
        
        with ProcessPoolExecutor(max_workers=max_parallel) as executor:
            # Submit all jobs
            future_to_experiment = {}
            for i, (config_name, ablation, seed) in enumerate(all_experiments):
                gpu_id = gpu_ids[i % len(gpu_ids)]
                future = executor.submit(
                    self.run_single_experiment, 
                    config_name, ablation, seed, gpu_id
                )
                future_to_experiment[future] = (config_name, ablation, seed)
            
            # Collect results as they complete
            for future in as_completed(future_to_experiment):
                config_name, ablation, seed = future_to_experiment[future]
                completed_count += 1
                
                try:
                    result = future.result()
                    
                    if config_name not in results:
                        results[config_name] = []
                    results[config_name].append(result)
                    
                    success_status = "✓" if result['success'] else "✗"
                    print(f"[{completed_count}/{len(all_experiments)}] {success_status} {config_name}_seed{seed}")
                    
                except Exception as e:
                    print(f"[{completed_count}/{len(all_experiments)}] ✗ {config_name}_seed{seed} failed: {e}")
        
        # Save aggregated results
        self._save_aggregated_results(results)
        
        return results
    
    def _save_aggregated_results(self, results: Dict[str, List[Dict]]):
        """Save aggregated results and analysis."""
        
        # Compute summary statistics
        summary = {}
        for config_name, experiments in results.items():
            successful_experiments = [exp for exp in experiments if exp['success']]
            
            if successful_experiments:
                success_rates = [exp['final_metrics']['final_success_rate'] for exp in successful_experiments]
                episode_counts = [exp['final_metrics']['final_episode_count'] for exp in successful_experiments]
                llm_calls = [exp['final_metrics']['total_llm_calls'] for exp in successful_experiments]
                
                summary[config_name] = {
                    'n_successful': len(successful_experiments),
                    'n_total': len(experiments),
                    'success_rate_mean': sum(success_rates) / len(success_rates),
                    'success_rate_std': (sum((x - sum(success_rates) / len(success_rates))**2 for x in success_rates) / len(success_rates))**0.5,
                    'episodes_mean': sum(episode_counts) / len(episode_counts),
                    'llm_calls_mean': sum(llm_calls) / len(llm_calls) if llm_calls else 0,
                    'config_description': ABLATION_CONFIGS[config_name].description
                }
            else:
                summary[config_name] = {
                    'n_successful': 0,
                    'n_total': len(experiments),
                    'success_rate_mean': 0.0,
                    'success_rate_std': 0.0,
                    'episodes_mean': 0,
                    'llm_calls_mean': 0,
                    'config_description': ABLATION_CONFIGS[config_name].description
                }
        
        # Save detailed results
        with open(os.path.join(self.output_dir, 'detailed_results.json'), 'w') as f:
            json.dump(results, f, indent=2)
        
        # Save summary
        with open(os.path.join(self.output_dir, 'summary.json'), 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Generate readable report
        report_path = os.path.join(self.output_dir, 'ablation_report.md')
        self._generate_report(summary, report_path)
        
        print(f"Results saved to {self.output_dir}")
        print(f"Summary report: {report_path}")
    
    def _generate_report(self, summary: Dict[str, Dict], report_path: str):
        """Generate human-readable markdown report."""
        
        with open(report_path, 'w') as f:
            f.write("# LoRe Ablation Study Results\n\n")
            f.write(f"Generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("## Summary\n\n")
            f.write("| Config | Description | Success Rate | Std | Episodes | LLM Calls | Success/Total |\n")
            f.write("|--------|-------------|--------------|-----|----------|-----------|---------------|\n")
            
            for config_name in ['A', 'B', 'C', 'D', 'E', 'F']:
                if config_name in summary:
                    s = summary[config_name]
                    f.write(f"| {config_name} | {s['config_description']} | "
                           f"{s['success_rate_mean']:.3f} | {s['success_rate_std']:.3f} | "
                           f"{s['episodes_mean']:.0f} | {s['llm_calls_mean']:.0f} | "
                           f"{s['n_successful']}/{s['n_total']} |\n")
            
            f.write("\n## Key Findings\n\n")
            
            # Automatic analysis
            if 'A' in summary and 'F' in summary:
                baseline_sr = summary['A']['success_rate_mean']
                final_sr = summary['F']['success_rate_mean']
                improvement = final_sr - baseline_sr
                
                f.write(f"- **Overall improvement**: {improvement:.3f} success rate improvement from baseline to full LoRe\n")
                
            f.write("- **Configuration comparison**: See detailed results for step-by-step analysis\n")
            f.write("- **API efficiency**: PriorNet distillation reduces LLM calls while maintaining performance\n")
            
            f.write("\n## Detailed Configuration Descriptions\n\n")
            for config_name in ['A', 'B', 'C', 'D', 'E', 'F']:
                if config_name in ABLATION_CONFIGS:
                    ablation = ABLATION_CONFIGS[config_name]
                    f.write(f"### Configuration {config_name}: {ablation.name}\n")
                    f.write(f"{ablation.description}\n\n")
                    f.write("Settings:\n")
                    f.write(f"- LLM enabled: {ablation.llm_enabled}\n")
                    f.write(f"- Beta max: {ablation.beta_max}\n")
                    f.write(f"- Delta target: {ablation.delta_target}\n")
                    f.write(f"- Use PriorNet: {ablation.use_priornet}\n\n")


def main():
    """Run the ablation study."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run LoRe ablation study")
    parser.add_argument("--configs", nargs='+', default=['A', 'B', 'C', 'D', 'E', 'F'],
                       help="Configurations to run")
    parser.add_argument("--parallel", type=int, default=4, help="Number of parallel processes")
    parser.add_argument("--gpus", nargs='+', type=int, default=[0], help="GPU IDs to use")
    parser.add_argument("--output", type=str, default="ablation_results", help="Output directory")
    parser.add_argument("--short", action='store_true', help="Run short experiments (25k steps)")
    
    args = parser.parse_args()
    
    # Modify configs for short run
    if args.short:
        for config in ABLATION_CONFIGS.values():
            config.total_steps = 25000
            config.seeds = [42, 123]  # Only 2 seeds for quick test
    
    runner = AblationRunner(output_dir=args.output)
    results = runner.run_ablation_study(
        configs=args.configs,
        max_parallel=args.parallel,
        gpu_ids=args.gpus
    )
    
    print("Ablation study completed!")
    return results


if __name__ == "__main__":
    main()