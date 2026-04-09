import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, f_classif

import joblib
import os

CLEAN_PATH = 'data/processed/cleaned.csv'
FEAT_PATH = 'data/processed/features.csv'
N_FEATURES = 30  # top features to keep


def prepare():
    df = pd.read_csv(CLEAN_PATH)

    # Encode labels
    le = LabelEncoder()
    df['Label_enc'] = le.fit_transform(df['Label'])
    joblib.dump(le, 'data/models/label_encoder.pkl')

    X = df.drop(columns=['Label', 'Label_enc'])
    y = df['Label_enc']

    # SPLIT FIRST (prevents data leakage)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=42
    )

    # Feature selection ONLY on training data
    selector = SelectKBest(f_classif, k=N_FEATURES)
    X_train = selector.fit_transform(X_train, y_train)
    X_test = selector.transform(X_test)

    selected_cols = X.columns[selector.get_support()].tolist()
    print('Selected features:', selected_cols)

    # Save selected feature names
    joblib.dump(selected_cols, 'data/models/feature_cols.pkl')

    # Scaling (fit only on training data)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    joblib.dump(scaler, 'data/models/scaler.pkl')

    print(f'Training data shape: {X_train.shape}')
    print(f'Test data shape: {X_test.shape}')

    return X_train, X_test, y_train, y_test, le


if __name__ == '__main__':
    os.makedirs('data/models', exist_ok=True)
    prepare()