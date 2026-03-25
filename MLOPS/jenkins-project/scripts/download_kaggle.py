"""Download dataset from Kaggle using Kaggle API.
Dataset: aliiihussain/amazon-sales-dataset
Requires a valid kaggle.json in ~/.kaggle or provided via environment.
"""
import os
from pathlib import Path

def download_dataset():
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except Exception as e:
        raise RuntimeError('kaggle package not installed. Install via pip install kaggle') from e

    out_dir = Path(__file__).resolve().parent.parent / 'data'
    out_dir.mkdir(parents=True, exist_ok=True)

    api = KaggleApi()
    api.authenticate()

    dataset = 'aliiihussain/amazon-sales-dataset'
    print(f'Downloading {dataset} to {out_dir} ...')
    api.dataset_download_files(dataset, path=str(out_dir), unzip=True)
    print('Download complete')


if __name__ == '__main__':
    download_dataset()
