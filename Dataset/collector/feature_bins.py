def bin_cpu_load(load_pct: float) -> int:
    if load_pct < 30: return 0
    if load_pct < 70: return 1
    return 2

def bin_gpu_load(load_pct: float) -> int:
    if load_pct < 20: return 0
    if load_pct < 80: return 1
    return 2

def bin_gpu_mem(free_mb: int, total_mb: int) -> int:
    frac = free_mb / total_mb
    if frac > 0.7: return 2
    if frac > 0.3: return 1
    return 0

def bin_gpu_temp(temp_c: float) -> int:
    if temp_c < 70: return 0
    if temp_c < 85: return 1
    return 2
