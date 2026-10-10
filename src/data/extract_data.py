import os
import shutil
import kagglehub

def fetch_datasets():
    """
    Downloads multiple datasets from Kaggle and moves them
    to the project's raw data directory for further processing.
    """
    datasets_to_download = [
        "akouaorsot/nietzsches-bibliography",
        # "another-author/dataset-name-2",
        # "another-author/dataset-name-3"
    ]

    project_raw_dir = os.path.join("artifacts", "raw")
    os.makedirs(project_raw_dir, exist_ok=True)

    for dataset_id in datasets_to_download:
        print(f"[*] Starting download for dataset: {dataset_id}")

        # 1. Download to Kaggle local cache
        cache_path = kagglehub.dataset_download(dataset_id)
        print(f"    - Cached at: {cache_path}")

        # 2. Move files from cache to project directory
        for filename in os.listdir(cache_path):
            source_file = os.path.join(cache_path, filename)
            dest_file = os.path.join(project_raw_dir, filename)

            if os.path.isfile(source_file):
                shutil.copy2(source_file, dest_file)
                print(f"    - Copied {filename} to {project_raw_dir}/")

        print(f"[*] Successfully extracted: {dataset_id}\n")

if __name__ == "__main__":
    fetch_datasets()
    print("All data extraction completed successfully.")