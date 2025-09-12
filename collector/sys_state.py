import psutil
import torch
import os

def get_system_state():
    """
    Collect real-time system stats:
    - CPU load %
    - Battery / power mode
    - GPU load %, temperature, memory usage & pressure
    - Number of concurrent GPU tasks
    """
    state = {}

    state["cpu_load_pct"] = psutil.cpu_percent(interval=None)

    if hasattr(psutil, "sensors_battery"):
        battery = psutil.sensors_battery()
        state["is_battery_powered"] = int(battery and battery.power_plugged)
    else:
        state["is_battery_powered"] = 0

    try:
        if os.path.exists("/etc/nvpmodel.conf"):
            cmd = "sudo nvpmodel -q --verbose | grep 'Power Mode' | awk '{print $3}'"
            mode = os.popen(cmd).read().strip()
            state["power_mode"] = int(mode) if mode.isdigit() else 0
        else:
            state["power_mode"] = 0
    except:
        state["power_mode"] = 0

    if torch.cuda.is_available():
        if NVML_AVAILABLE:
            h = pynvml.nvmlDeviceGetHandleByIndex(0)
            util = pynvml.nvmlDeviceGetUtilizationRates(h)
            mem = pynvml.nvmlDeviceGetMemoryInfo(h)
            temp = pynvml.nvmlDeviceGetTemperature(h, pynvml.NVML_TEMPERATURE_GPU)
            procs = pynvml.nvmlDeviceGetComputeRunningProcesses(h)

            state.update({
                "gpu_load_pct": util.gpu,
                "gpu_mem_used_mb": mem.used // (1024 ** 2),
                "gpu_mem_total_mb": mem.total // (1024 ** 2),
                "gpu_mem_pressure": round(mem.used / mem.total, 4),
                "gpu_temp_C": temp,
                "thermal_headroom": max(0, 85 - temp),  
                "concurrent_gpu_tasks": len(procs)
            })
        else:
            state.update({
                "gpu_load_pct": -1,
                "gpu_mem_used_mb": -1,
                "gpu_mem_total_mb": -1,
                "gpu_mem_pressure": -1,
                "gpu_temp_C": -1,
                "thermal_headroom": -1,
                "concurrent_gpu_tasks": -1
            })
    else:
        state.update({
            "gpu_load_pct": 0,
            "gpu_mem_used_mb": 0,
            "gpu_mem_total_mb": 0,
            "gpu_mem_pressure": 0,
            "gpu_temp_C": 0,
            "thermal_headroom": 0,
            "concurrent_gpu_tasks": 0
        })

    return state


if __name__ == "__main__":
    print(get_system_state())
