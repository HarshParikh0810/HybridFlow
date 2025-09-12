import time
import torch
from collector.device_profile import get_device_profile
from collector.sys_state import get_system_state

def collect_one_conv2d(N, Cin, H, W, Cout, kernel=3, dtype="float32"):
    if isinstance(dtype, str):
        dtype = getattr(torch, dtype)

    A = torch.randn(N, Cin, H, W, dtype=dtype, device="cpu")
    Wt = torch.randn(Cout, Cin, kernel, kernel, dtype=dtype, device="cpu")

    # CPU run
    start = time.perf_counter()
    out_cpu = torch.nn.functional.conv2d(A, Wt)
    cpu_time = (time.perf_counter() - start) * 1000

    # GPU run
    if torch.cuda.is_available():
        A_gpu, Wt_gpu = A.cuda(), Wt.cuda()
        torch.cuda.synchronize()
        start = time.perf_counter()
        out_gpu = torch.nn.functional.conv2d(A_gpu, Wt_gpu)
        torch.cuda.synchronize()
        gpu_time = (time.perf_counter() - start) * 1000
        start = time.perf_counter()
        _ = A.cuda(); _ = Wt.cuda()
        torch.cuda.synchronize()
        transfer_time = (time.perf_counter() - start) * 1000
    else:
        gpu_time, transfer_time = float("inf"), float("inf")

    gpu_total = gpu_time + transfer_time
    winner = 0 if cpu_time < gpu_total else 1

    dev = get_device_profile()
    sys_state = get_system_state()

    return {
        "operation_type": "conv2d",
        "log_size": float(round(torch.log10(torch.tensor(N*Cin*H*W*Cout*kernel*kernel+1.0)).item(), 4)),
        "dtype": str(dtype).replace("torch.", ""),
        "cpu_runtime_ms": round(cpu_time, 4),
        "gpu_runtime_ms": round(gpu_time, 4),
        "transfer_time_ms": round(transfer_time, 4),
        "winner": winner,
        **sys_state,
        "device_name": dev["device_name"],
        "is_edge": dev["is_edge"],
    }
