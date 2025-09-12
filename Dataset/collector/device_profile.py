import platform
import psutil

def get_device_profile():
    name = "Unknown"
    mem_total_mb = 0
    bus_type = "Unknown"
    source = "none"

    try:
        import pynvml
        pynvml.nvmlInit()
        h = pynvml.nvmlDeviceGetHandleByIndex(0)
        name = pynvml.nvmlDeviceGetName(h)
        if isinstance(name, bytes):
            name = name.decode("utf-8")
        mem_total_mb = pynvml.nvmlDeviceGetMemoryInfo(h).total // (1024**2)
        bus_type = "PCIe"
        source = "nvml"
    except Exception:
        try:
            from jtop import jtop
            with jtop() as jetson:
                name = "Jetson " + jetson.board['model']
                mem_total_mb = jetson.memory['RAM']['tot'] // 1024  # MB
                bus_type = "SoC"
                source = "jtop"
        except Exception:
            name = platform.node()
            mem_total_mb = psutil.virtual_memory().total // (1024**2)
            bus_type = "CPU-only"
            source = "cpu"

    if "Jetson" in name or "Tegra" in name or bus_type == "SoC":
        is_edge = 1
    elif mem_total_mb < 4096:
        is_edge = 1
    else:
        is_edge = 0

    return {
        "device_name": name,
        "gpu_mem_total_mb": int(mem_total_mb),
        "bus_type": bus_type,
        "is_edge": is_edge,
        "source": source
    }
