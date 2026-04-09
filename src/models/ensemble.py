import numpy as np
import joblib
from tensorflow.keras.models import load_model


class EnsemblePredictor:
    def __init__(self):
        self.rf = joblib.load('data/models/rf_model.pkl')
        self.xgb = joblib.load('data/models/xgb_model.pkl')
        self.lstm = load_model('data/models/lstm_model.h5')
        self.scaler = joblib.load('data/models/scaler.pkl')
        self.le = joblib.load('data/models/label_encoder.pkl')
        self.feature_cols = joblib.load('data/models/feature_cols.pkl')

    def predict(self, X_scaled: np.ndarray):
        """
        X_scaled: 2D numpy array, shape (n_samples, n_features)
        Returns: list of dicts with label, confidence, risk_level
        """
        p_rf = self.rf.predict_proba(X_scaled)
        p_xgb = self.xgb.predict_proba(X_scaled)

        X_lstm = X_scaled.reshape(X_scaled.shape[0], 1, X_scaled.shape[1])
        p_lstm = self.lstm.predict(X_lstm)

        # Soft vote: average probabilities with weights
        p_avg = (0.35 * p_rf + 0.35 * p_xgb + 0.30 * p_lstm)

        predicted_idx = np.argmax(p_avg, axis=1)
        confidence = np.max(p_avg, axis=1)
        labels = self.le.inverse_transform(predicted_idx)

        results = []
        for label, conf in zip(labels, confidence):
            results.append({
                'label': label,
                'confidence': round(float(conf) * 100, 2),
                'risk_level': _risk(label),
            })

        return results


def _risk(label):
    label = label.upper()

    if 'BENIGN' in label:
        return 'NONE'
    if 'DDOS' in label:
        return 'CRITICAL'
    if 'BOT' in label:
        return 'HIGH'
    if 'BRUTE' in label:
        return 'HIGH'
    if 'INFILTR' in label:
        return 'HIGH'
    if 'PORT' in label:
        return 'MEDIUM'
    if 'SQL' in label:
        return 'MEDIUM'
    if 'XSS' in label:
        return 'MEDIUM'
    if 'WEB' in label:
        return 'MEDIUM'

    return 'LOW'