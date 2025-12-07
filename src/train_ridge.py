from pipeline import run_pipeline
from sklearn.linear_model import Ridge
from evaluation import evaluate_model
import joblib


def run_ridge():
    # Run the preprocessing pipeline
    X_train, y_train, X_val, y_val, _ = run_pipeline(processor=2, use_all_features=False)

    # Initialize and train the Ridge regression model
    model = Ridge(alpha=1)
    model.fit(X_train, y_train)

    # Predict on validation set
    y_pred = model.predict(X_val)

    # Evaluate the model
    results = evaluate_model(y_val, y_pred, model_name='Ridge α=1')
    print(results)

    # Save the model
    joblib.dump(model, 'models/ridge.pkl')
    print('Ridge model saved')


if __name__ == '__main__':
    run_ridge()
