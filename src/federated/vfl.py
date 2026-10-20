# src/federated/vfl.py
"""
Vertical Federated Learning (VFL) simulation, implemented as a Split Neural
Network: the feature set is split between two "parties" that each hold
different columns for the SAME samples (e.g. Party A = timing features,
Party B = packet-size/flag features). Each party runs a small local "bottom
model" that turns its own features into an embedding; only the embeddings
(not raw features) are sent to a coordinator, which concatenates them and
runs the final classification layer. That's the actual privacy property of
VFL, and it's worth stating explicitly in your report.

Run:
    python -m src.federated.vfl
"""

import numpy as np
import joblib
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, Concatenate
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import accuracy_score, classification_report

from src.preprocessing.feature_engineer import prepare

EMBED_DIM = 32
EPOCHS = 15
BATCH_SIZE = 512


def split_features(X, feature_cols):
    """Splits columns roughly in half between the two parties.
    Customize this split to be semantically meaningful, e.g.:
      party_a_cols = [c for c in feature_cols if 'IAT' in c or 'Duration' in c]
      party_b_cols = [c for c in feature_cols if c not in party_a_cols]
    """
    half = len(feature_cols) // 2
    return X[:, :half], X[:, half:], feature_cols[:half], feature_cols[half:]


def build_party_model(input_dim: int, name: str):
    """A party's local 'bottom model' — produces an embedding, never raw
    features leave this function's output."""
    inp = Input(shape=(input_dim,), name=f"{name}_input")
    x = Dense(64, activation="relu")(inp)
    x = Dropout(0.2)(x)
    embedding = Dense(EMBED_DIM, activation="relu", name=f"{name}_embedding")(x)
    return inp, embedding


def build_vfl_model(dim_a: int, dim_b: int, n_classes: int):
    inp_a, emb_a = build_party_model(dim_a, "party_a")
    inp_b, emb_b = build_party_model(dim_b, "party_b")

    # Coordinator: only ever sees embeddings, not raw party features
    merged = Concatenate()([emb_a, emb_b])
    x = Dense(64, activation="relu")(merged)
    x = Dropout(0.2)(x)
    out = Dense(n_classes, activation="softmax")(x)

    model = Model(inputs=[inp_a, inp_b], outputs=out)
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return model


def run_vfl():
    X_train, X_test, y_train, y_test, le = prepare()
    feature_cols = joblib.load("data/models/feature_cols.pkl")
    n_classes = len(le.classes_)

    Xa_train, Xb_train, cols_a, cols_b = split_features(X_train, feature_cols)
    Xa_test, Xb_test, _, _ = split_features(X_test, feature_cols)
    print(f"Party A features ({len(cols_a)}): {cols_a}")
    print(f"Party B features ({len(cols_b)}): {cols_b}")

    model = build_vfl_model(Xa_train.shape[1], Xb_train.shape[1], n_classes)
    model.summary()

    y_train_cat = to_categorical(y_train, n_classes)
    y_test_cat = to_categorical(y_test, n_classes)

    model.fit([Xa_train, Xb_train], y_train_cat,
              validation_data=([Xa_test, Xb_test], y_test_cat),
              epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=1)

    y_pred = np.argmax(model.predict([Xa_test, Xb_test], verbose=0), axis=1)
    acc = accuracy_score(y_test, y_pred)
    print(f"\n=== VFL FINAL — accuracy: {acc:.4f} ===")
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    model.save("data/models/vfl_model.h5")
    print("Saved: data/models/vfl_model.h5")
    return model


if __name__ == "__main__":
    run_vfl()