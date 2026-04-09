from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib

from src.preprocessing.feature_engineer import prepare


def train():
    X_train, X_test, y_train, y_test, le = prepare()

    print('Training Random Forest...')

    clf = RandomForestClassifier(
        n_estimators=50,
        max_depth=None,
        min_samples_split=5,
        class_weight='balanced_subsample',
        n_jobs=-1,
        random_state=42
    )

    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    print(classification_report(
        y_test,
        y_pred,
        target_names=le.classes_
    ))

    joblib.dump(clf, 'data/models/rf_model.pkl')
    print('Saved: data/models/rf_model.pkl')


if __name__ == '__main__':
    train()