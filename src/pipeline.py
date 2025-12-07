import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler

from data_load import load_data, Outliers_handling
from feature_engineering import prepare_data


def encoding(train, validation, use_all_features):
    """
    One-hot encode categoricals + select final feature set including target.
    """
    # 1) Define categorical columns to be one-hot encoded
    categorical_features = [
        'passenger_count',
        'vendor_id',
        'store_and_fwd_flag',
        'Working_days',
        'rush_hour',
        'month',
        'hour',
        'day',
        'day_of_week',
    ]

    # 2) Create a OneHotEncoder
    enc = OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False)

    # 3) Fit encoder on train and transform train/validation
    cat_train = pd.DataFrame(
        enc.fit_transform(train[categorical_features]),
        columns=enc.get_feature_names_out(categorical_features),
        index=train.index,
    )
    cat_val = pd.DataFrame(
        enc.transform(validation[categorical_features]),
        columns=enc.get_feature_names_out(categorical_features),
        index=validation.index,
    )

    # 4) Concatenate original numeric + engineered features
    train_full = pd.concat([train, cat_train], axis=1)
    val_full = pd.concat([validation, cat_val], axis=1)

    # 5) If we want to keep ALL features (for feature selection / experiments),
    if use_all_features:
        num_features_train = train_full.select_dtypes(include='number').columns
        train_full_numeric = train_full[num_features_train].copy()
        train_target = train_full_numeric.pop('log_trip_duration')
        train_full_numeric.loc[:, 'log_trip_duration'] = train_target  # put target last

        num_features_val = val_full.select_dtypes(include='number').columns
        val_full_numeric = val_full[num_features_val].copy()
        val_target = val_full_numeric.pop('log_trip_duration')
        val_full_numeric.loc[:, 'log_trip_duration'] = val_target  # put target last

        return train_full_numeric, val_full_numeric

    # Cleaned feature set after feature selection
    features = [
        'pickup_longitude',
        'pickup_latitude',
        'dropoff_longitude',
        'dropoff_latitude',
        'latitude_distance',
        'longitude_distance',
        'manhattan_distance',
        'haversine_distance',
        'distance_ratio',
        'midpoint_latitude',
        'midpoint_longitude',
        'bearing',
        'vendor_id_2',
        'Working_days_1',
        'rush_hour_1',
        'month_2',
        'month_3',
        'month_4',
        'month_5',
        'month_6',
        'hour_sin',
        'hour_cos',
        'day_of_week_sin',
        'day_of_week_cos',
        'day_sin',
        'day_cos',
        'bearing_sin',
        'bearing_cos',
        'log_trip_duration',
    ]

    df_train = train_full[features]
    df_val = val_full[features]

    return df_train, df_val


def preparedata(train, test, processor):
    """
    Convert to numpy, split into X/y and optionally scale.
    processor:
        0 -> no scaling (tree models)
        1 -> StandardScaler
        2 -> MinMaxScaler
    """
    # 1) Convert to NumPy for faster ML model running
    train = train.to_numpy()
    test = test.to_numpy()

    # 2) Split into features (X) and target (y)
    x_train = train[:, :-1]
    t_train = train[:, -1]

    x_test = test[:, :-1]
    t_test = test[:, -1]

    # 3) Optionally apply scaling:
    scaler = None
    if processor == 1:
        scaler = StandardScaler()
        x_train = scaler.fit_transform(x_train)
        x_test = scaler.transform(x_test)
    elif processor == 2:
        scaler = MinMaxScaler()
        x_train = scaler.fit_transform(x_train)
        x_test = scaler.transform(x_test)

    return x_train, t_train, x_test, t_test


def run_pipeline(processor, use_all_features):
    """
    Full pipeline:
      load → outliers → feature engineering → encoding → X/y split
    Returns:
      X_train, y_train, X_val, y_val, feature_names
    """
    # 1) Load raw training and validation datasets
    train, validation = load_data()

    print(f'Training data before processed: {train.shape}')
    print(f'Validation data before processed: {validation.shape}')

    # 2) Handle outliers
    train = Outliers_handling(train)
    validation = Outliers_handling(validation)

    # 3) Apply feature engineering
    train = prepare_data(train)
    validation = prepare_data(validation)

    # 4) Encode categorical variables and build the final feature set
    train_encoded, val_encoded = encoding(train, validation, use_all_features)

    feature_names = list(train_encoded.columns[:-1])

    # splitting the data
    X_train, y_train, X_val, y_val = preparedata(train_encoded, val_encoded, processor=processor)

    print(f'Training data after processed: {train_encoded.shape}')
    print(f'Validation data after processed: {val_encoded.shape}')

    return X_train, y_train, X_val, y_val, feature_names
