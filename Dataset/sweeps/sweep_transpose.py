import os, pandas as pd
import torch
from collector.collect_transpose import collect_one_transpose

os.makedirs("datasets", exist_ok=True)

sizes = [
    # Small
    (128, 128), (192, 192), (256, 256), (384, 384), (512, 512),

    # Medium
    (640, 640), (768, 768), (896, 896), (1024, 1024), (1280, 1280),

    # Large
    (1536, 1536), (1792, 1792), (2048, 2048), (2304, 2304), (2560, 2560),

    # Very Large (but still safe on laptop GPU/CPU)
    (3072, 3072), (3584, 3584), (4096, 4096), (4608, 4608), (5120, 5120),

    # Rectangular — realistic for memory layout stress
    (1024, 2048), (2048, 4096), (1536, 3072), (2560, 4096), (3072, 5120)
]

dtypes = [torch.float32, torch.float16]

rows = []
for d in dtypes:
    for H, W in sizes:
        rows.append(collect_one_transpose(H, W, dtype=d))

df = pd.DataFrame(rows)
df.to_parquet("datasets/transpose.parquet", index=False)
print("✅ Saved transpose dataset:", df.shape)
print(df[["cpu_runtime_ms","gpu_runtime_ms"]])