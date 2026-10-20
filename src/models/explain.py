
import os
import joblib
import numpy as np
import shap
import matplotlib.pyplot as plt

MODEL_PATH = "data/models/rf_model.pkl"
EXPLAINER_PATH = "data/models/shap_explainer.pkl"


def build_shap_explainer(model_path: str = MODEL_PATH,
                          out_path: str = EXPLAINER_PATH):
    """Builds and saves a TreeExplainer for the Random Forest model."""
    model = joblib.load(model_path)
    explainer = shap.TreeExplainer(model)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    joblib.dump(explainer, out_path)
    print(f"SHAP explainer saved to {out_path}")
    return explainer


def _extract_class_values(shap_values, predicted_class: int):
    """
    Handles both SHAP API shapes:
      - old API: list of arrays, one per class -> shap_values[class][sample]
      - new API: single array shaped (n_samples, n_features, n_classes)
    Returns a 1D array of per-feature SHAP values for the predicted class.
    """
    if isinstance(shap_values, list):
        return np.array(shap_values[predicted_class][0])
    arr = np.array(shap_values)
    if arr.ndim == 3:
        return arr[0, :, predicted_class]
    return arr[0]


def explain_single(x_scaled_row: np.ndarray,
                    feature_names: list,
                    predicted_class: int,
                    top_n: int = 10,
                    explainer_path: str = EXPLAINER_PATH) -> dict:
    """
    x_scaled_row: 1D array, already scaled, matching feature_names order.
    predicted_class: the integer class index the ensemble predicted
                      (pass this in from ensemble.py's output so the
                      explanation matches what was actually shown to the user).
    Returns: {feature_name: shap_value}, sorted by |impact|, top_n entries.
    """
    explainer = joblib.load(explainer_path)
    shap_values = explainer.shap_values(x_scaled_row.reshape(1, -1))
    values = _extract_class_values(shap_values, predicted_class)

    pairs = sorted(zip(feature_names, values), key=lambda p: abs(p[1]), reverse=True)
    return {name: round(float(val), 5) for name, val in pairs[:top_n]}


def plot_summary(X_test: np.ndarray, feature_names: list,
                  sample_size: int = 500, save_path: str = "shap_summary.png",
                  explainer_path: str = EXPLAINER_PATH):
    """Global feature-importance plot for your presentation/report."""
    explainer = joblib.load(explainer_path)
    subset = X_test[:sample_size]
    shap_values = explainer.shap_values(subset)

    values = shap_values[1] if isinstance(shap_values, list) else shap_values
    shap.summary_plot(values, subset, feature_names=feature_names, show=False)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved SHAP summary plot to {save_path}")


if __name__ == "__main__":
    build_shap_explainer()