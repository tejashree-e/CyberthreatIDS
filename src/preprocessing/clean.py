import pandas as pd
import numpy as np
import glob
import os

RAW_DIR = 'data/raw'
OUT_PATH = 'data/processed/cleaned.csv'


def load_all_csvs():
    files = glob.glob(os.path.join(RAW_DIR, '*.csv'))
    print(f'Found {len(files)} CSV files')

    frames = []
    for f in files:
        df = pd.read_csv(f, encoding='latin-1')
        df.columns = df.columns.str.strip()  # remove whitespace from headers
        frames.append(df)

    return pd.concat(frames, ignore_index=True)


def clean(df):
    print(f'Raw shape: {df.shape}')

    # Drop rows with infinity or NaN
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)

    # Drop constant columns
    df = df.loc[:, df.nunique() > 1]

    # Standardize label column
    label_col = ' Label' if ' Label' in df.columns else 'Label'
    df.rename(columns={label_col: 'Label'}, inplace=True)
    df['Label'] = df['Label'].str.strip()

    print(f'Clean shape: {df.shape}')
    print(df['Label'].value_counts())

    return df


if __name__ == '__main__':
    df = load_all_csvs()
    df = clean(df)

    os.makedirs('data/processed', exist_ok=True)
    df.to_csv(OUT_PATH, index=False)

    print(f'Saved to {OUT_PATH}')