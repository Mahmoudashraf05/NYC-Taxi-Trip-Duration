import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from evaluation import evaluate_model
from pipeline import run_pipeline

def run_feature_selection():
    # Run the preprocessing pipeline
    X_train, y_train, X_val, y_val, feature_names  = run_pipeline(processor=0, use_all_features=True)

    # Initialize and train the Random Forest model
    model = RandomForestRegressor(
        n_estimators=80, max_depth=20, random_state=42, n_jobs=-1
    )
    model.fit(X_train, y_train)

    # Predict on validation set
    y_pred = model.predict(X_val)

    # Evaluate the model
    results = evaluate_model(y_val, y_pred, model_name='Random Forest (Feature Selection)')
    print(results)

    # Feature importance
    fi = pd.DataFrame(
        {'Feature': feature_names, 'Importance': model.feature_importances_}
    ).sort_values('Importance', ascending=False)
    print(fi.head(30))

if __name__ == '__main__':
    run_feature_selection()