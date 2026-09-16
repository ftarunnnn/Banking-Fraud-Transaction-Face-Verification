import os
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

def train_models(input_csv='data/engineered_transactions.csv', model_dir='models'):
    if not os.path.exists(input_csv):
        input_csv = 'data/processed_transactions.csv'
        
    df = pd.read_csv(input_csv)
    
    feature_cols = [
        'amount', 'distance_from_home_km', 'user_avg_amount', 'amount_ratio_to_avg',
        'amount_zscore', 'time_diff_minutes', 'is_rapid_tx', 'high_distance_flag',
        'is_night_tx', 'hour', 'day_of_week', 'is_weekend',
        'transaction_type_encoded', 'merchant_category_encoded',
        'location_city_encoded', 'device_id_encoded'
    ]
    
    X = df[feature_cols]
    y = df['is_fraud']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    os.makedirs(model_dir, exist_ok=True)
    
    # 1. Random Forest Classifier
    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=12,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    rf_model.fit(X_train_scaled, y_train)
    
    # 2. XGBoost Classifier
    scale_pos_weight = (len(y_train) - sum(y_train)) / sum(y_train)
    xgb_model = XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.05,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        eval_metric='logloss'
    )
    xgb_model.fit(X_train_scaled, y_train)
    
    # Save artifacts
    joblib.dump(rf_model, os.path.join(model_dir, 'fraud_model_rf.pkl'))
    joblib.dump(xgb_model, os.path.join(model_dir, 'fraud_model_xgb.pkl'))
    joblib.dump(scaler, os.path.join(model_dir, 'scaler.pkl'))
    joblib.dump(feature_cols, os.path.join(model_dir, 'feature_names.pkl'))
    
    # Save test data for evaluation in Phase 6
    np.save(os.path.join(model_dir, 'X_test_scaled.npy'), X_test_scaled)
    np.save(os.path.join(model_dir, 'y_test.npy'), y_test.values)
    
    print(f"[OK] Phase 5: Models trained successfully.")
    print(f"     Train samples: {len(X_train)}, Test samples: {len(X_test)}")
    print(f"     Saved: fraud_model_rf.pkl, fraud_model_xgb.pkl, scaler.pkl, feature_names.pkl")

if __name__ == '__main__':
    train_models()
