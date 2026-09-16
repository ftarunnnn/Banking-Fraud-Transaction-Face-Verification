import os
import numpy as np
import pandas as pd

def engineer_features(input_csv='data/processed_transactions.csv', output_csv='data/engineered_transactions.csv'):
    if not os.path.exists(input_csv):
        input_csv = 'data/raw_transactions.csv'
        
    df = pd.read_csv(input_csv)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values(by=['user_id', 'timestamp']).reset_index(drop=True)
    
    # 1. Historical User Transaction Aggregates
    user_avg_amount = df.groupby('user_id')['amount'].transform('mean')
    user_std_amount = df.groupby('user_id')['amount'].transform('std').fillna(1.0)
    
    df['user_avg_amount'] = user_avg_amount
    df['amount_ratio_to_avg'] = df['amount'] / (user_avg_amount + 1e-5)
    df['amount_zscore'] = (df['amount'] - user_avg_amount) / (user_std_amount + 1e-5)
    
    # 2. Transaction Frequency & Velocity
    # Transaction count per user in the dataset
    df['user_tx_count'] = df.groupby('user_id')['transaction_id'].transform('count')
    
    # Time diff between consecutive transactions per user (in minutes)
    df['time_diff_minutes'] = df.groupby('user_id')['timestamp'].diff().dt.total_seconds() / 60.0
    df['time_diff_minutes'] = df['time_diff_minutes'].fillna(9999.0) # first transaction default
    
    # Rapid transaction flag (< 5 minutes)
    df['is_rapid_tx'] = (df['time_diff_minutes'] < 5.0).astype(int)
    
    # 3. Location & Device Anomaly Features
    # Distance threshold flag (> 100 km from home)
    df['high_distance_flag'] = (df['distance_from_home_km'] > 100.0).astype(int)
    
    # 4. Time-based Patterns
    df['hour'] = df['timestamp'].dt.hour
    # Late night transaction (11 PM [23] to 5 AM [5])
    df['is_night_tx'] = df['hour'].apply(lambda h: 1 if (h >= 23 or h <= 5) else 0)
    
    # Save engineered dataset
    df.to_csv(output_csv, index=False)
    print(f"[OK] Phase 4: Feature engineering completed. Shape: {df.shape}")
    print(f"     New Features Added: user_avg_amount, amount_ratio_to_avg, amount_zscore, time_diff_minutes, is_rapid_tx, high_distance_flag, is_night_tx")
    return df

if __name__ == '__main__':
    engineer_features()
