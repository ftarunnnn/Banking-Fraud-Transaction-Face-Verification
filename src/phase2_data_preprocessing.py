import os
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

def preprocess_transaction_data(input_csv='data/raw_transactions.csv', output_csv='data/processed_transactions.csv'):
    if not os.path.exists(input_csv):
        raise FileNotFoundError(f"Input file {input_csv} not found. Run Phase 1 first.")
        
    df = pd.read_csv(input_csv)
    initial_shape = df.shape
    
    # 1. Handle Duplicates & Missing Values
    df = df.drop_duplicates()
    df = df.dropna()
    
    # Ensure timestamp is datetime format
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # 2. Extract basic time components
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['is_weekend'] = df['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)
    
    # 3. Categorical Feature Encoding
    encoders = {}
    cat_columns = ['transaction_type', 'merchant_category', 'location_city', 'device_id']
    
    for col in cat_columns:
        le = LabelEncoder()
        df[f'{col}_encoded'] = le.fit_transform(df[col].astype(str))
        encoders[col] = le
        
    os.makedirs('models', exist_ok=True)
    encoder_path = os.path.join('models', 'encoders.pkl')
    joblib.dump(encoders, encoder_path)
    
    # Save processed dataframe
    df.to_csv(output_csv, index=False)
    print(f"[OK] Phase 2: Transaction data preprocessed. Shape: {initial_shape} -> {df.shape}")
    print(f"[OK] Encoders saved to {encoder_path}")
    return df

def preprocess_face_image_numpy(img_np, target_size=(160, 160)):
    """
    Resize face numpy array/image and normalize pixels to [-1, 1] range.
    """
    from PIL import Image
    if isinstance(img_np, np.ndarray):
        img = Image.fromarray(img_np.astype('uint8'))
    else:
        img = img_np
        
    img_resized = img.resize(target_size)
    img_arr = np.asarray(img_resized, dtype=np.float32)
    # Normalize to [-1, 1]
    img_norm = (img_arr - 127.5) / 128.0
    return img_norm

if __name__ == '__main__':
    preprocess_transaction_data()
