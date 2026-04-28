"""
Common ML utility functions.
"""

import os
import logging

# Fallback logger if main logger isn't available
logger = logging.getLogger("ml_utils")

def is_running_in_container() -> bool:
    """Detect if code is running inside a container."""
    if os.path.exists('/.dockerenv'):
        return True
    if os.path.exists('/run/.containerenv'):
        return True
    try:
        with open('/proc/1/cgroup', 'r') as f:
            content = f.read()
            if 'docker' in content or 'kubepods' in content or 'containerd' in content:
                return True
    except (FileNotFoundError, PermissionError):
        pass
    return False

def get_device() -> str:
    """
    Return 'cuda' if NVIDIA GPU is available and working, 'mps' if Apple Silicon is available, else 'cpu'.
    Can be overridden by setting the FORCE_DEVICE environment variable.
    """
    # Allow manual override via environment variable
    forced_device = os.getenv("FORCE_DEVICE")
    if forced_device:
        print(f"ML_UTILS: Forced device to {forced_device}")
        return forced_device

    try:
        import torch
        # Defensively check if we can even talk to CUDA
        print("ML_UTILS: Detecting device...")
        
        has_cuda = False
        try:
            has_cuda = hasattr(torch, 'cuda') and torch.cuda.is_available()
        except (RuntimeError, Exception) as e:
            print(f"ML_UTILS: torch.cuda.is_available() failed: {e}")
            return 'cpu'

        if has_cuda:
            try:
                # Try to initialize CUDA to ensure driver is present and working
                torch.cuda.init()
                print("ML_UTILS: CUDA initialized successfully.")
                return 'cuda'
            except (RuntimeError, Exception) as e:
                print(f"ML_UTILS: CUDA init failed (likely no driver): {e}")
                return 'cpu'
        
        # Apple Silicon check (MPS)
        # We skip MPS if running in a container, as standard Linux containers don't support it
        # even on Mac hosts, and it often leads to "not linked" errors.
        if not is_running_in_container():
            try:
                if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                    # Perform a "smoke test" to ensure it's actually linked and working
                    # This prevents the "PyTorch is not linked with support for mps devices" error
                    test_tensor = torch.zeros(1).to("mps")
                    print("ML_UTILS: MPS available and verified.")
                    return 'mps'
            except Exception as e:
                print(f"ML_UTILS: MPS detection/smoke-test failed: {e}")
            
        print("ML_UTILS: Defaulting to CPU.")
        return 'cpu'
    except Exception as e:
        # Catch all (ImportError, etc.) and fallback to CPU
        print(f"ML_UTILS: Device detection failed completely: {e}")
        return 'cpu'

def get_gpu_usage() -> float:
    """
    Return current GPU memory usage percentage (0.0 to 100.0).
    Returns 0.0 if no GPU is available or on error.
    """
    try:
        import torch
        if not torch.cuda.is_available():
            return 0.0
        
        # Get current device
        device = torch.cuda.current_device()
        
        # Get memory info
        t = torch.cuda.get_device_properties(device).total_memory
        r = torch.cuda.memory_reserved(device)
        a = torch.cuda.memory_allocated(device)
        # We care about total reserved/used memory on the device
        # nvidia-smi is more accurate for system-wide usage
        import subprocess
        try:
            cmd = "nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits"
            result = subprocess.check_output(cmd.split(), encoding='utf-8')
            used, total = map(int, result.strip().split(','))
            return (used / total) * 100.0
        except Exception:
            # Fallback to torch stats (less accurate for other processes)
            return (r / t) * 100.0
            
    except ImportError:
        return 0.0

def millisec_to_time(ms: int) -> str:
    """Convert milliseconds to HH:MM:SS string."""
    seconds = ms // 1000
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    return f"{h:02}:{m:02}:{s:02}"
