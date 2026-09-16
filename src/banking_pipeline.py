import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import joblib
import numpy as np
import pandas as pd
from src.phase8_face_embedding import FaceVerifier

class BankingSecuritySystem:
    def __init__(self, model_dir='models', fraud_threshold=0.30):
        self.fraud_threshold = fraud_threshold
        self.model_dir = model_dir
        
        # Load ML Fraud detection artifacts
        self.rf_model = joblib.load(os.path.join(model_dir, 'fraud_model_rf.pkl'))
        self.xgb_model = joblib.load(os.path.join(model_dir, 'fraud_model_xgb.pkl'))
        self.scaler = joblib.load(os.path.join(model_dir, 'scaler.pkl'))
        self.feature_cols = joblib.load(os.path.join(model_dir, 'feature_names.pkl'))
        self.encoders = joblib.load(os.path.join(model_dir, 'encoders.pkl'))
        
        # Load DL Face Verifier
        self.face_verifier = FaceVerifier()

    def process_transaction(self, transaction_payload, submitted_face_image=None, registered_face_image=None):
        """
        Processes a raw transaction payload and optional facial image.
        
        transaction_payload dict:
            user_id, amount, distance_from_home_km, transaction_type, merchant_category, location_city, device_id, timestamp
        """
        # 1. Feature Preprocessing & Encoding
        payload = transaction_payload.copy()
        
        # User historical defaults if not provided
        user_avg_amount = payload.get('user_avg_amount', payload['amount'] * 0.8)
        user_std_amount = payload.get('user_std_amount', 50.0)
        
        amount_ratio = payload['amount'] / (user_avg_amount + 1e-5)
        amount_zscore = (payload['amount'] - user_avg_amount) / (user_std_amount + 1e-5)
        
        # Time features
        ts = pd.to_datetime(payload.get('timestamp', pd.Timestamp.now()))
        hour = ts.hour
        day_of_week = ts.dayofweek
        is_weekend = 1 if day_of_week >= 5 else 0
        is_night = 1 if (hour >= 23 or hour <= 5) else 0
        
        # Categorical Encoding
        tx_type_enc = self.encoders['transaction_type'].transform([payload['transaction_type']])[0] if payload['transaction_type'] in self.encoders['transaction_type'].classes_ else 0
        merchant_enc = self.encoders['merchant_category'].transform([payload['merchant_category']])[0] if payload['merchant_category'] in self.encoders['merchant_category'].classes_ else 0
        city_enc = self.encoders['location_city'].transform([payload['location_city']])[0] if payload['location_city'] in self.encoders['location_city'].classes_ else 0
        device_enc = self.encoders['device_id'].transform([payload['device_id']])[0] if payload['device_id'] in self.encoders['device_id'].classes_ else 0
        
        feat_dict = {
            'amount': payload['amount'],
            'distance_from_home_km': payload['distance_from_home_km'],
            'user_avg_amount': user_avg_amount,
            'amount_ratio_to_avg': amount_ratio,
            'amount_zscore': amount_zscore,
            'time_diff_minutes': payload.get('time_diff_minutes', 60.0),
            'is_rapid_tx': 1 if payload.get('time_diff_minutes', 60.0) < 5.0 else 0,
            'high_distance_flag': 1 if payload['distance_from_home_km'] > 100.0 else 0,
            'is_night_tx': is_night,
            'hour': hour,
            'day_of_week': day_of_week,
            'is_weekend': is_weekend,
            'transaction_type_encoded': tx_type_enc,
            'merchant_category_encoded': merchant_enc,
            'location_city_encoded': city_enc,
            'device_id_encoded': device_enc
        }
        
        X_df = pd.DataFrame([feat_dict])[self.feature_cols]
        X_scaled = self.scaler.transform(X_df)
        
        # 2. Fraud Model Predictions
        xgb_prob = float(self.xgb_model.predict_proba(X_scaled)[0, 1])
        rf_prob = float(self.rf_model.predict_proba(X_scaled)[0, 1])
        fraud_risk_score = (xgb_prob * 0.6) + (rf_prob * 0.4) # Weighted ensemble
        
        is_fraud_predicted = fraud_risk_score >= self.fraud_threshold
        
        # 3. Decision & Biometric Trigger
        face_verification_result = None
        face_required = is_fraud_predicted or payload['distance_from_home_km'] > 200.0 or payload['amount'] > 1000.0
        
        if face_required:
            if submitted_face_image is not None and registered_face_image is not None:
                face_verification_result = self.face_verifier.verify_faces(registered_face_image, submitted_face_image)
            else:
                face_verification_result = {
                    'verified': False,
                    'similarity': 0.0,
                    'l2_distance': 999.0,
                    'confidence_pct': 0.0,
                    'status': '[REQUIRED] Biometric Face Verification Pending'
                }
                
        # Final Decision Logic
        if not face_required:
            final_status = 'APPROVED'
            decision_msg = '[APPROVED] Transaction Approved (Low Fraud Risk)'
        else:
            if face_verification_result and face_verification_result.get('verified', False):
                final_status = 'APPROVED_WITH_BIOMETRIC'
                decision_msg = '[APPROVED] Transaction Approved (Biometrically Authenticated)'
            elif face_verification_result and not face_verification_result.get('verified', False) and submitted_face_image is not None:
                final_status = 'BLOCKED_IDENTITY_MISMATCH'
                decision_msg = '[BLOCKED] Transaction Blocked (Biometric Verification Failed)'
            else:
                final_status = 'REQUIRES_FACE_VERIFICATION'
                decision_msg = '[REQUIRED] Biometric Verification Required'
                
        return {
            'final_status': final_status,
            'decision_message': decision_msg,
            'fraud_risk_score': fraud_risk_score,
            'fraud_risk_pct': fraud_risk_score * 100.0,
            'is_suspicious': is_fraud_predicted,
            'face_required': face_required,
            'face_verification_result': face_verification_result,
            'features': feat_dict
        }

if __name__ == '__main__':
    system = BankingSecuritySystem()
    
    # Test normal transaction
    sample_normal = {
        'user_id': 'USR_0001',
        'amount': 45.50,
        'distance_from_home_km': 2.3,
        'transaction_type': 'POS',
        'merchant_category': 'Grocery',
        'location_city': 'New York',
        'device_id': 'DEV_001',
        'timestamp': '2026-09-16 14:30:00'
    }
    
    # Test high-risk suspicious transaction
    sample_suspicious = {
        'user_id': 'USR_0001',
        'amount': 2500.00,
        'distance_from_home_km': 1500.0,
        'transaction_type': 'Wire Transfer',
        'merchant_category': 'Luxury',
        'location_city': 'Tokyo',
        'device_id': 'DEV_999',
        'timestamp': '2026-09-16 02:15:00'
    }
    
    res1 = system.process_transaction(sample_normal)
    res2 = system.process_transaction(
        sample_suspicious,
        submitted_face_image='data/face_dataset/user_001/face_2.jpg',
        registered_face_image='data/face_dataset/user_001/face_1.jpg'
    )
    
    print("[OK] Phase 9: Integrated Banking Pipeline Test")
    print(f"     Normal Transaction Decision:     {res1['decision_message']} (Risk: {res1['fraud_risk_pct']:.1f}%)")
    print(f"     Suspicious Transaction Decision: {res2['decision_message']} (Risk: {res2['fraud_risk_pct']:.1f}%)")
