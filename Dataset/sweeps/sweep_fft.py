import os
import pandas as pd
import torch
from collector.collect_fft import collect_one_fft

os.makedirs("datasets", exist_ok=True)

# 🔹 Diverse FFT sizes (small → very large)
sizes = [
    256, 384, 512, 768, 1024, 1536,
    2048, 3072, 4096, 5120, 6144, 7168,
    8192, 9216, 10240, 12288, 14336,
    16384, 20480, 24576, 28672, 32768,
    40960, 49152, 65536
]

# 🔹 Batch sizes
batches = [1, 4, 16]   # lighter to heavier workloads

# 🔹 Only float32 supported for FFT
dtypes = [torch.float32]

rows = []
for d in dtypes:
    for N in sizes:
        for b in batches:
            rows.append(collect_one_fft(N=N, batch=b, dtype=d))

df = pd.DataFrame(rows)
df.to_parquet("datasets/fft.parquet", index=False)
print("✅ Saved fft dataset:", df.shape)
print(df[["cpu_runtime_ms","gpu_runtime_ms"]])