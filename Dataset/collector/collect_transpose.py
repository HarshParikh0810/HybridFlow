import time
import torch
from collector.device_profile import get_device_profile
from collector.sys_state import get_system_state

def estimate_transfer_cost(tensor):
    bytes_size = tensor.element_size() * tensor.nelement()
    bandwidth = 12e9 if torch.cuda.is_available() else 1e9
    return (bytes_size / bandwidth) * 1000

def collect_one_transpose(M=1024, N=1024, dtype=torch.float32):
    A = torch.randn(M, N, dtype=dtype, device="cpu")

    # CPU
    start = time.perf_counter()
    _ = A.T
    cpu_time = (time.perf_counter() - start) * 1000

    # GPU
    if torch.cuda.is_available():
        A_gpu = A.cuda()
        torch.cuda.synchronize()
        start = time.perf_counter()
        _ = A_gpu.T
        torch.cuda.synchronize()
        gpu_time = (time.perf_counter() - start) * 1000

        transfer_time = estimate_transfer_cost(A)
    else:
        gpu_time, transfer_time = float("inf"), float("inf")

    winner = 0 if cpu_time < (gpu_time + transfer_time) else 1

    dev = get_device_profile()
    sys_state = get_system_state()

    return {
        "operation_type": "transpose",
        "log_size": float(round(torch.log10(torch.tensor(M*N+1.)).item(), 4)),
        "dtype": str(dtype).replace("torch.", ""),
        "cpu_runtime_ms": round(cpu_time, 4),
        "gpu_runtime_ms": round(gpu_time, 4),
        "transfer_time_ms": round(transfer_time, 4),
        "data_transfer_cost_ms": round(transfer_time, 4),
        "winner": winner,
        **sys_state,
        "device_name": dev["device_name"],
        "is_edge": dev["is_edge"],
    }
