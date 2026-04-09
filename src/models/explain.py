import shap
import joblib
import numpy as np
import matplotlib.pyplot as plt


def build_shap_explainer():
    rf = joblib.load('data/models/rf_model.pkl')

    # TreeExplainer is fast for Random Forest / XGBoost
    explainer = shap.TreeExplainer(rf)

    joblib.dump(explainer, 'data/models/shap_explainer.pkl')
    print('SHAP explainer saved.')

    return explainer


def explain_single(X_sample: np.ndarray, feature_names: list):
    """
    X_sample: 1D array of one flow's features
    Returns: dict of {feature: shap_value}
    """
    explainer = joblib.load('data/models/shap_explainer.pkl')

    shap_values = explainer.shap_values(X_sample.reshape(1, -1))

    # Take values for the predicted class
    pred_class = np.argmax(explainer.expected_value)
    sv = shap_values[pred_class][0]

    return dict(
        sorted(
            zip(feature_names, sv),
            key=lambda x: abs(x[1]),
            reverse=True
        )[:10]
    )


def plot_summary(X_test, feature_names):
    explainer = joblib.load('data/models/shap_explainer.pkl')

    shap_values = explainer.shap_values(X_test[:500])

    shap.summary_plot(
        shap_values[0],
        X_test[:500],
        feature_names=feature_names,
        show=True
    )


if __name__ == '__main__':
    build_shap_explainer()