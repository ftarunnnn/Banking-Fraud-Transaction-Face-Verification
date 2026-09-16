import os
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

def generate_transaction_dataset(num_samples=10000, seed=42):
    np.random.seed(seed)
    os.makedirs('data', exist_ok=True)
    
    user_ids = [f'USR_{i:04d}' for i in range(1, 101)] # 100 registered users
    cities = ['New York', 'London', 'Tokyo', 'Mumbai', 'Paris', 'Berlin', 'Sydney', 'Toronto']
    tx_types = ['Online', 'POS', 'ATM', 'Wire Transfer']
    merchant_cats = ['Grocery', 'Electronics', 'Luxury', 'Gaming', 'Transfer', 'Utilities', 'Travel']
    devices = [f'DEV_{i:03d}' for i in range(1, 150)]
    
    data = []
    start_time = pd.Timestamp('2026-01-01 00:00:00')
    
    for i in range(num_samples):
        tx_id = f'TXN_{i+1:06d}'
        user_id = np.random.choice(user_ids)
        
        # User home city index
        home_city_idx = int(user_id.split('_')[1]) % len(cities)
        home_city = cities[home_city_idx]
        home_device = f"DEV_{int(user_id.split('_')[1]):03d}"
        
        # Generate timestamp over 90 days
        time_offset = np.random.randint(0, 90 * 24 * 3600)
        timestamp = start_time + pd.Timedelta(seconds=time_offset)
        hour = timestamp.hour
        
        # Fraud probability logic
        # 1. High amount
        # 2. Unusual location (high distance)
        # 3. New device
        # 4. Late night (11 PM - 5 AM)
        
        is_fraud_trigger = np.random.rand() < 0.04  # ~4% base fraud
        
        if is_fraud_trigger:
            is_fraud = 1
            amount = np.round(np.random.exponential(scale=1200) + 400, 2)
            distance = np.round(np.random.uniform(200, 5000), 2)
            city = np.random.choice([c for c in cities if c != home_city])
            device = np.random.choice(devices)
            tx_type = np.random.choice(['Online', 'Wire Transfer', 'ATM'])
            merchant = np.random.choice(['Luxury', 'Gaming', 'Transfer', 'Electronics'])
        else:
            is_fraud = 0
            amount = np.round(np.random.exponential(scale=65) + 10, 2)
            distance = np.round(np.random.exponential(scale=5), 2)
            city = home_city if np.random.rand() > 0.1 else np.random.choice(cities)
            device = home_device if np.random.rand() > 0.15 else np.random.choice(devices)
            tx_type = np.random.choice(tx_types, p=[0.4, 0.4, 0.15, 0.05])
            merchant = np.random.choice(merchant_cats)
            
        data.append({
            'transaction_id': tx_id,
            'user_id': user_id,
            'timestamp': timestamp,
            'amount': amount,
            'location_city': city,
            'distance_from_home_km': distance,
            'device_id': device,
            'transaction_type': tx_type,
            'merchant_category': merchant,
            'is_fraud': is_fraud
        })
        
    df = pd.DataFrame(data)
    csv_path = os.path.join('data', 'raw_transactions.csv')
    df.to_csv(csv_path, index=False)
    print(f"[OK] Phase 1: Transaction dataset generated with {len(df)} rows. Fraud count: {df['is_fraud'].sum()} ({df['is_fraud'].mean()*100:.2f}%)")
    return df

def generate_face_dataset():
    base_dir = os.path.join('data', 'face_dataset')
    os.makedirs(base_dir, exist_ok=True)
    
    users = ['user_001', 'user_002', 'user_003', 'user_004', 'user_005']
    colors = [
        ((235, 190, 160), (40, 30, 20), (50, 100, 200)),  # User 1 skin, hair, eyes
        ((220, 170, 140), (20, 20, 20), (100, 60, 30)),   # User 2
        ((240, 200, 170), (150, 80, 40), (30, 120, 80)),  # User 3
        ((190, 130, 100), (10, 10, 10), (40, 40, 40)),    # User 4
        ((245, 210, 180), (200, 160, 50), (60, 140, 220)) # User 5
    ]
    
    for idx, user in enumerate(users):
        user_dir = os.path.join(base_dir, user)
        os.makedirs(user_dir, exist_ok=True)
        skin, hair, eye = colors[idx % len(colors)]
        
        # Generate 3 registered photos per user with minor pose variations
        for img_i in range(1, 4):
            img = Image.new('RGB', (256, 256), color=(240, 242, 245))
            draw = ImageDraw.Draw(img)
            
            # Head background / Hair
            draw.ellipse([50 - img_i*2, 30, 206 + img_i*2, 220], fill=hair)
            # Face contour
            draw.ellipse([60, 60, 196, 215], fill=skin)
            # Eyes
            draw.ellipse([85 + img_i, 105, 115 + img_i, 130], fill=(255, 255, 255))
            draw.ellipse([141 - img_i, 105, 171 - img_i, 130], fill=(255, 255, 255))
            draw.ellipse([95 + img_i, 112, 105 + img_i, 123], fill=eye)
            draw.ellipse([151 - img_i, 112, 161 - img_i, 123], fill=eye)
            # Nose
            draw.line([(128, 125), (124, 155), (132, 155)], fill=(160, 110, 90), width=3)
            # Mouth
            draw.arc([100, 160, 156, 185], start=0, end=180, fill=(180, 60, 60), width=4)
            
            file_path = os.path.join(user_dir, f'face_{img_i}.jpg')
            img.save(file_path)
            
    print(f"[OK] Phase 1: Face dataset created for {len(users)} users in {base_dir}")

if __name__ == '__main__':
    generate_transaction_dataset()
    generate_face_dataset()
