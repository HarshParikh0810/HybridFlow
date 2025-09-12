import os, pandas as pd, torch
from collector.collect_matmul import collect_one_matmul

os.makedirs("datasets", exist_ok=True)

# Expanded sizes: small → medium → large (but not insane)
small_med = [16, 24, 32, 40, 48, 56, 64, 80, 96, 112, 128,
             160, 192, 224, 256, 288, 320, 352, 384, 416, 448,
             480, 512, 576, 640, 704, 768, 832, 896, 960, 1024]

# Sparse sampling large → very large
large = [1152, 1280, 1408, 1536, 1792, 2048, 2560, 3072]

sizes = small_med + large
dtypes = [torch.float32, torch.float16]

rows = []
for d in dtypes:
    for n in sizes:
        rows.append(collect_one_matmul(n, n, n, dtype=d))

df = pd.DataFrame(rows)
df.to_parquet("datasets/matmul.parquet", index=False)
print("✅ Saved matmul dataset:", df.shape)
print(df[["cpu_runtime_ms", "gpu_runtime_ms"]])
