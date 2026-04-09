import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import to_categorical
import joblib

from src.preprocessing.feature_engineer import prepare


def train():
    X_train, X_test, y_train, y_test, le = prepare()
    n_classes = len(le.classes_)

    # Reshape for LSTM: (samples, timesteps=1, features)
    X_tr_l = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
    X_te_l = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])

    y_tr_c = to_categorical(y_train, n_classes)
    y_te_c = to_categorical(y_test, n_classes)

    model = Sequential([
        LSTM(
            128,
            input_shape=(1, X_train.shape[1]),
            return_sequences=True
        ),
        BatchNormalization(),
        Dropout(0.3),

        LSTM(64),
        BatchNormalization(),
        Dropout(0.3),

        Dense(64, activation='relu'),
        Dropout(0.2),

        Dense(n_classes, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    model.summary()

    callbacks = [
        EarlyStopping(patience=5, restore_best_weights=True),
        ModelCheckpoint(
            'data/models/lstm_model.h5',
            save_best_only=True
        )
    ]

    model.fit(
        X_tr_l,
        y_tr_c,
        validation_data=(X_te_l, y_te_c),
        epochs=30,
        batch_size=1024,
        callbacks=callbacks
    )


if __name__ == '__main__':
    train()