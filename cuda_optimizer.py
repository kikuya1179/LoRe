"""CUDA optimization utilities for maximum GPU performance."""

import torch
import os
from typing import Dict, Any


def setup_cuda_optimization() -> Dict[str, Any]:
    """
    Configure CUDA for maximum performance.
    
    Returns:
        Dict with optimization status and settings
    """
    if not torch.cuda.is_available():
        return {"status": "cuda_not_available", "optimizations": []}
    
    optimizations = []
    
    # 1. Enable TF32 for faster training on Ampere GPUs (RTX 30xx, A100, etc.)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    optimizations.append("TF32 enabled")
    
    # 2. Enable cuDNN benchmark for fixed input sizes
    torch.backends.cudnn.benchmark = True
    optimizations.append("cuDNN benchmark enabled")
    
    # 3. Set deterministic operations if needed (disable for max speed)
    torch.backends.cudnn.deterministic = False
    optimizations.append("Deterministic disabled (max speed)")
    
    # 4. Enable memory pool for faster allocation (check platform support)
    try:
        # Test if expandable_segments is supported
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            test_tensor = torch.randn(10, 10, device="cuda")
            del test_tensor
        
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128,expandable_segments:True"
        optimizations.append("Memory pool optimization")
    except Exception:
        # Fallback without expandable_segments
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
        optimizations.append("Memory pool optimization (basic)")
    
    # 5. Set CUDA cache for faster compilation
    os.environ["CUDA_MODULE_LOADING"] = "LAZY"
    optimizations.append("Lazy CUDA module loading")
    
    # 6. GPU memory management
    torch.cuda.empty_cache()
    if hasattr(torch.cuda, 'set_per_process_memory_fraction'):
        torch.cuda.set_per_process_memory_fraction(0.9)
        optimizations.append("GPU memory fraction set to 90%")
    
    # 7. Enable faster GPU communications if multi-GPU
    if torch.cuda.device_count() > 1:
        os.environ["NCCL_BLOCKING_WAIT"] = "1"
        optimizations.append("NCCL optimization for multi-GPU")
    
    # 8. JIT compilation optimizations
    torch.jit.set_fusion_strategy([("STATIC", 3), ("DYNAMIC", 3)])
    optimizations.append("JIT fusion optimization")
    
    return {
        "status": "optimized",
        "device_count": torch.cuda.device_count(),
        "device_name": torch.cuda.get_device_name(0),
        "memory_gb": torch.cuda.get_device_properties(0).total_memory / 1024**3,
        "compute_capability": f"{torch.cuda.get_device_properties(0).major}.{torch.cuda.get_device_properties(0).minor}",
        "optimizations": optimizations
    }




def optimize_batch_size(base_batch_size: int, device: torch.device) -> int:
    """
    Automatically optimize batch size based on GPU memory.
    
    Args:
        base_batch_size: Starting batch size
        device: CUDA device
        
    Returns:
        Optimized batch size
    """
    if device.type != "cuda":
        return base_batch_size
    
    # Get GPU memory in GB
    memory_gb = torch.cuda.get_device_properties(device).total_memory / 1024**3
    
    # Scale batch size based on memory
    if memory_gb >= 24:  # RTX 4090, A100
        multiplier = 2.0
    elif memory_gb >= 16:  # RTX 4080, RTX 3090
        multiplier = 1.5
    elif memory_gb >= 12:  # RTX 4070 Ti, RTX 3080
        multiplier = 1.2
    elif memory_gb >= 8:   # RTX 4060, RTX 3070
        multiplier = 1.0
    else:  # Lower memory GPUs
        multiplier = 0.8
    
    optimized_size = int(base_batch_size * multiplier)
    print(f"[GPU] Batch size optimized: {base_batch_size} -> {optimized_size} (GPU: {memory_gb:.1f}GB)")
    
    return optimized_size


def warmup_cuda(device: torch.device) -> None:
    """
    Warm up CUDA for faster first iteration.
    
    Args:
        device: CUDA device
    """
    if device.type != "cuda":
        return
    
    print("[GPU] Warming up CUDA...")
    
    # Create dummy tensors and perform operations (suppress warnings)
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with torch.no_grad():
            x = torch.randn(256, 256, device=device)
            y = torch.randn(256, 256, device=device)
            
            # Matrix operations
            _ = torch.mm(x, y)
            _ = torch.nn.functional.relu(x)
            _ = torch.nn.functional.conv2d(x.unsqueeze(0).unsqueeze(0), 
                                           y[:3, :3].unsqueeze(0).unsqueeze(0))
    
    torch.cuda.synchronize()
    print("[GPU] CUDA warmup completed")


def print_gpu_status() -> None:
    """Print detailed GPU status information."""
    if not torch.cuda.is_available():
        print("[GPU] CUDA is not available")
        return
    
    print("\n" + "="*50)
    print("GPU STATUS")
    print("="*50)
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA version: {torch.version.cuda}")
    print(f"cuDNN version: {torch.backends.cudnn.version()}")
    print(f"Device count: {torch.cuda.device_count()}")
    
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        memory_gb = props.total_memory / 1024**3
        print(f"GPU {i}: {props.name}")
        print(f"  Memory: {memory_gb:.1f} GB")
        print(f"  Compute: {props.major}.{props.minor}")
        print(f"  Multiprocessors: {props.multi_processor_count}")
    
    print(f"\nCUDA Optimizations:")
    print(f"  TF32 MatMul: {torch.backends.cuda.matmul.allow_tf32}")
    print(f"  TF32 cuDNN: {torch.backends.cudnn.allow_tf32}")
    print(f"  cuDNN Benchmark: {torch.backends.cudnn.benchmark}")
    print(f"  Deterministic: {torch.backends.cudnn.deterministic}")
    print("="*50 + "\n")