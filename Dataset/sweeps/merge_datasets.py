import pandas as pd
import glob, os

def merge_datasets(input_folder="datasets", output_file="datasets/merged_dataset.csv"):
    # Find all parquet files inside datasets/
    parquet_files = glob.glob(os.path.join(input_folder, "*.parquet"))

    if not parquet_files:
        print("❌ No parquet files found in", input_folder)
        return

    dfs = []
    for f in parquet_files:
        try:
            df = pd.read_parquet(f)
            df["source_file"] = os.path.basename(f)
            dfs.append(df)
            print(f"✅ Loaded {f} with {len(df)} rows")
        except Exception as e:
            print(f"⚠️ Could not read {f}: {e}")

    if not dfs:
        print("❌ No datasets loaded.")
        return

    merged = pd.concat(dfs, ignore_index=True)

    # Drop duplicates if run_id accidentally added twice
    if "run_id" in merged.columns:
        merged = merged.drop_duplicates(subset=["run_id"])

    # Save as CSV for accessibility
    merged.to_csv(output_file, index=False)
    print(f"\n🎉 Merged dataset saved at {output_file} with {len(merged)} rows")

    # ✅ Print rows per operation_type
    if "operation_type" in merged.columns:
        print("\n📊 Rows per operation_type:")
        print(merged["operation_type"].value_counts())

if __name__ == "__main__":
    merge_datasets()
