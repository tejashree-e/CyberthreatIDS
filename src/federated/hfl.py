# src/federated/hfl.py
"""
Horizontal Federated Learning (HFL) simulation.

Same feature schema, samples split across N simulated "clients" (think:
N campus network sensors each seeing different traffic). Each client trains
locally on its own shard; a coordinator aggregates weights with FedAvg.
No raw data ever leaves a client's shard — only model weights are shared,
which is the actual privacy property you should mention in your report.

Run:
    python -m src.federated.hfl
"""

import numpy as np
import joblib
from tensorflow.keras.models import Sequential, clone_model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import accuracy_score, classification_report

from src.preprocessing.feature_engineer import prepare

N_CLIENTS = 5
N_ROUNDS = 15
LOCAL_EPOCHS = 2
BATCH_SIZE = 512


def build_model(input_dim: int, n_classes: int):
    model = Sequential([
        Dense(128, activation="relu", input_shape=(input_dim,)),
        Dropout(0.3),
        Dense(64, activation="relu"),
        Dropout(0.2),
        Dense(n_classes, activation="softmax"),
    ])
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return model


def split_clients(X, y, n_clients: int):
    """IID split for simplicity. Swap for a non-IID split (e.g. by attack type)
    if you want to show FL handling harder, more realistic conditions."""
    idx = np.random.permutation(len(X))
    shards = np.array_split(idx, n_clients)
    return [(X[s], y[s]) for s in shards]


def federated_average(weight_list, sample_counts):
    total = sum(sample_counts)
    avg = []
    for layer_idx in range(len(weight_list[0])):
        layer_stack = np.array([w[layer_idx] for w in weight_list])
        weights = np.array(sample_counts) / total
        avg_layer = np.tensordot(weights, layer_stack, axes=(0, 0))
        avg.append(avg_layer)
    return avg


def run_hfl():
    X_train, X_test, y_train, y_test, le = prepare()
    X_train, y_train = np.asarray(X_train), np.asarray(y_train)
    X_test, y_test = np.asarray(X_test), np.asarray(y_test)
    n_classes = len(le.classes_)
    input_dim = X_train.shape[1]

    clients = split_clients(X_train, y_train, N_CLIENTS)
    global_model = build_model(input_dim, n_classes)
    global_weights = global_model.get_weights()

    for rnd in range(1, N_ROUNDS + 1):
        local_weights, sample_counts = [], []
        for X_c, y_c in clients:
            local_model = clone_model(global_model)
            local_model.set_weights(global_weights)
            local_model.compile(optimizer="adam", loss="categorical_crossentropy",
                                 metrics=["accuracy"])
            y_c_cat = to_categorical(y_c, n_classes)
            local_model.fit(X_c, y_c_cat, epochs=LOCAL_EPOCHS,
                             batch_size=BATCH_SIZE, verbose=0)
            local_weights.append(local_model.get_weights())
            sample_counts.append(len(X_c))

        global_weights = federated_average(local_weights, sample_counts)
        global_model.set_weights(global_weights)

        if rnd % 5 == 0 or rnd == N_ROUNDS:
            y_pred = np.argmax(global_model.predict(X_test, verbose=0), axis=1)
            acc = accuracy_score(y_test, y_pred)
            print(f"[HFL] Round {rnd}/{N_ROUNDS} — test accuracy: {acc:.4f}")

    y_pred = np.argmax(global_model.predict(X_test, verbose=0), axis=1)
    print("\n=== HFL FINAL REPORT ===")
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    global_model.save("data/models/hfl_model.h5")
    print("Saved: data/models/hfl_model.h5")
    return global_model


if __name__ == "__main__":
    run_hfl()