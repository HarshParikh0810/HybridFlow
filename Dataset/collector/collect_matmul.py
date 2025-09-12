import time
import torch
from collector.device_profile import get_device_profile
from collector.sys_state import get_system_state

def estimate_transfer_cost(tensor):
    """Rough estimate: size / bandwidth (GB/s)."""
    bytes_size = tensor.element_size() * tensor.nelement()
    if torch.cuda.is_available():
        bandwidth = 12e9 if "GeForce" in torch.cuda.get_device_name(0) else 20e9
    else:
        bandwidth = 1e9
    return (bytes_size / bandwidth) * 1000  # ms

def collect_one_matmul(m, n, k, dtype=torch.float32):
    A = torch.randn(m, k, dtype=dtype, device="cpu")
    B = torch.randn(k, n, dtype=dtype, device="cpu")

    # CPU run
    start = time.perf_counter()
    _ = A @ B
    cpu_time = (time.perf_counter() - start) * 1000

    # GPU run
    if torch.cuda.is_available():
        A_gpu, B_gpu = A.cuda(), B.cuda()
        torch.cuda.synchronize()
        start = time.perf_counter()
        _ = A_gpu @ B_gpu
        torch.cuda.synchronize()
        gpu_time = (time.perf_counter() - start) * 1000

        transfer_time = estimate_transfer_cost(A) + estimate_transfer_cost(B)
    else:
        gpu_time, transfer_time = float("inf"), float("inf")

    gpu_total = gpu_time + transfer_time
    winner = 0 if cpu_time < gpu_total else 1

    dev = get_device_profile()
    sys_state = get_system_state()

    return {
        "operation_type": "matmul",
        "log_size": float(round(torch.log10(torch.tensor(m*n*k+1.)).item(), 4)),
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
