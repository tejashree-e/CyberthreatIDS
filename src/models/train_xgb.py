from xgboost import XGBClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import joblib

from src.preprocessing.feature_engineer import prepare


def train():
    X_train, X_test, y_train, y_test, le = prepare()

    print('Training XGBoost...')

    # 🔥 STRATIFIED SAMPLING (keeps all classes, avoids label issues)
    X_train, _, y_train, _ = train_test_split(
        X_train,
        y_train,
        train_size=500000,
        stratify=y_train,
        random_state=42
    )

    clf = XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric='mlogloss',
        early_stopping_rounds=20,
        n_jobs=-1,
        random_state=42
    )

    clf.fit(
        X_train,
        y_train,
        eval_set=[(X_test, y_test)],
        verbose=50
    )

    y_pred = clf.predict(X_test)

    print(classification_report(
        y_test,
        y_pred,
        target_names=le.classes_
    ))

    joblib.dump(clf, 'data/models/xgb_model.pkl')
    print('Saved: data/models/xgb_model.pkl')


if __name__ == '__main__':
    train()