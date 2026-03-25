"""Wrapper script for data preprocessing.
Uses functions from download.py to load and clear the local dataset and write df_clear.csv
"""
from pathlib import Path
from download import load_local_data, clear_data


def main():
    base = Path(__file__).resolve().parent.parent
    # working directory is jenkins-project
    print('Loading local dataset...')
    df = load_local_data()
    print('Running cleaning and feature engineering...')
    df_clean = clear_data(df=df)
    print(f'df_clear.csv saved with shape {df_clean.shape}')


if __name__ == '__main__':
    main()
