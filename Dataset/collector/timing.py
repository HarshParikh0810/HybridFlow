import time, torch, numpy as np

def time_cpu(fn, warmup=3, reps=9):
    for _ in range(warmup): fn()
    t=[]
    for _ in range(reps):
        t0=time.perf_counter(); fn(); t1=time.perf_counter()
        t.append((t1-t0)*1000)
    return float(np.median(t))

def time_gpu_kernel(fn, warmup=3, reps=9):
    torch.cuda.synchronize()
    for _ in range(warmup): fn()
    times=[]
    for _ in range(reps):
        start=torch.cuda.Event(enable_timing=True); end=torch.cuda.Event(enable_timing=True)
        start.record(); fn(); end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    return float(np.median(times))

def time_h2d(t_cpu, warmup=2, reps=7):
    torch.cuda.synchronize()
    for _ in range(warmup): t_cpu.to("cuda"); torch.cuda.synchronize()
    ts=[]
    for _ in range(reps):
        torch.cuda.synchronize()
        t0=time.perf_counter(); t_cpu.to("cuda"); torch.cuda.synchronize()
        t1=time.perf_counter(); ts.append((t1-t0)*1000)
    return float(np.median(ts))

def time_d2h(t_gpu, warmup=2, reps=7):
    torch.cuda.synchronize()
    for _ in range(warmup): t_gpu.to("cpu"); torch.cuda.synchronize()
    ts=[]
    for _ in range(reps):
        torch.cuda.synchronize()
        t0=time.perf_counter(); t_gpu.to("cpu"); torch.cuda.synchronize()
        t1=time.perf_counter(); ts.append((t1-t0)*1000)
    return float(np.median(ts))
