import os, pandas as pd
from collector.collect_conv2d import collect_one_conv2d

os.makedirs("datasets", exist_ok=True)

configs = [
    # Small
    (1, 3, 32, 32, 16, 3),
    (1, 3, 64, 64, 32, 3),
    (1, 8, 64, 64, 32, 5),

    # Medium (dense sampling)
    (1, 16, 128, 128, 32, 3),
    (1, 16, 128, 128, 32, 5),
    (1, 32, 128, 128, 64, 3),
    (1, 32, 128, 128, 64, 5),
    (1, 32, 160, 160, 64, 3),
    (1, 32, 160, 160, 64, 5),
    (1, 32, 192, 192, 64, 3),
    (1, 32, 192, 192, 64, 5),

    # Large (sparser)
    (1, 32, 224, 224, 64, 3),
    (1, 32, 224, 224, 64, 5),
    (1, 64, 224, 224, 128, 3),
    (1, 64, 224, 224, 128, 5),

    # Very large (but not insane)
    (1, 32, 256, 256, 64, 3),
    (1, 32, 256, 256, 64, 5),
    (1, 64, 256, 256, 128, 3),
    (1, 64, 256, 256, 128, 5),
    (1, 32, 320, 320, 64, 3),
    (1, 32, 320, 320, 64, 5),
    (1, 64, 320, 320, 128, 3),
    (1, 64, 320, 320, 128, 5),
]

dtypes = ["float32", "float16"]

rows = []
for d in dtypes:
    for cfg in configs:
        rows.append(collect_one_conv2d(*cfg, dtype=d))

df = pd.DataFrame(rows)
df.to_parquet("datasets/conv2d.parquet", index=False)
print("✅ Saved conv2d dataset:", df.shape)
print(df[["cpu_runtime_ms", "gpu_runtime_ms"]])
