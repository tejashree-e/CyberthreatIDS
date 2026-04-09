import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score
)

from src.preprocessing.feature_engineer import prepare
from src.models.ensemble import EnsemblePredictor


def full_evaluation():
    _, X_test, _, y_test, le = prepare()

    predictor = EnsemblePredictor()
    results = predictor.predict(X_test)

    y_pred_labels = [r['label'] for r in results]
    y_pred_enc = le.transform(y_pred_labels)

    print('=== ENSEMBLE CLASSIFICATION REPORT ===')
    print(classification_report(y_test, y_pred_enc, target_names=le.classes_))

    cm = confusion_matrix(y_test, y_pred_enc)

    plt.figure(figsize=(12, 10))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=le.classes_,
        yticklabels=le.classes_
    )

    plt.title('Ensemble Confusion Matrix')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=150)

    print('Confusion matrix saved.')


if __name__ == '__main__':
    full_evaluation()