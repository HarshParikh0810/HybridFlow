import os, pandas as pd
import torch
from collector.collect_elementwise import collect_one_elementwise

os.makedirs("datasets", exist_ok=True)

sizes = [64, 128, 256, 384, 512, 640, 768, 896, 1024, 1152, 1280,
    1408, 1536, 1664, 1792, 1920, 2048, 2304, 2560,
    2816, 3072, 3328, 3584, 3840, 4096, 4608, 5120,
    5632, 6144, 6656, 7168, 7680, 8192, 9216, 10240,
    11264, 12288, 13312, 14336, 15360, 16384, 18432, 20480, 32768, 65536]
dtypes = [torch.float32, torch.float16]

rows = []
for d in dtypes:
    for n in sizes:
        rows.append(collect_one_elementwise(N=n, dtype=d))

df = pd.DataFrame(rows)
df.to_parquet("datasets/elementwise.parquet", index=False)
print("✅ Saved elementwise dataset:", df.shape)
print(df[["cpu_runtime_ms","gpu_runtime_ms"]])



