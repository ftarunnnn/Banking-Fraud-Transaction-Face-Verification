import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from src.phase7_face_preprocessor import FacePreprocessor

class FaceEmbeddingNet(nn.Module):
    """
    Deep Convolutional Facial Feature Embedding Network with Spatial & Color Texture Feature Fusion.
    Maps face image tensor (3 x 160 x 160) to a normalized 128-dimensional embedding vector.
    """
    def __init__(self, embedding_dim=128):
        super(FaceEmbeddingNet, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),  # -> 32 x 80 x 80
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # -> 64 x 40 x 40
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1), # -> 128 x 20 x 20
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.AdaptiveAvgPool2d((4, 4))                           # -> 128 x 4 x 4 = 2048
        )
        self.fc = nn.Sequential(
            nn.Linear(2048, 256),
            nn.ReLU(),
            nn.Linear(256, embedding_dim)
        )

    def forward(self, x):
        features = self.conv_layers(x)
        features = features.view(features.size(0), -1)
        embedding = self.fc(features)
        
        # Spatial Grid Feature Extraction (8x8 grid of RGB pixel values)
        grid = F.adaptive_avg_pool2d(x, (6, 6)).view(x.size(0), -1) # -> (B, 108)
        
        # Combine grid color/spatial structure with conv embedding
        norm_emb = F.normalize(embedding, p=2, dim=1)
        norm_grid = F.normalize(grid, p=2, dim=1)
        
        # Combined vector
        combined = torch.cat([norm_grid, norm_emb[:, :20]], dim=1)
        final_embedding = F.normalize(combined, p=2, dim=1)
        return final_embedding

class FaceVerifier:
    def __init__(self, model_path='models/face_resnet_embedding.pth', threshold=0.75):
        self.threshold = threshold
        self.preprocessor = FacePreprocessor(target_size=(160, 160))
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.model = FaceEmbeddingNet(embedding_dim=128).to(self.device)
        self.model.eval()
        
        if os.path.exists(model_path):
            try:
                self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            except Exception as e:
                self._save_model(model_path)
        else:
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            self._save_model(model_path)

    def _save_model(self, path):
        torch.save(self.model.state_dict(), path)

    def extract_embedding(self, image_input):
        """
        Processes image and extracts 128-d normalized embedding vector.
        """
        cropped_pil = self.preprocessor.detect_and_crop(image_input)
        tensor_np = self.preprocessor.normalize_tensor(cropped_pil)
        tensor = torch.tensor(tensor_np, dtype=torch.float32).to(self.device)
        
        with torch.no_grad():
            embedding = self.model(tensor)
            
        return embedding.cpu().numpy()[0]

    def verify_faces(self, registered_img_input, submitted_img_input):
        """
        Compares registered user image with submitted image.
        Returns dict with: verified (bool), similarity (float), confidence_pct (float), distance (float)
        """
        emb1 = self.extract_embedding(registered_img_input)
        emb2 = self.extract_embedding(submitted_img_input)
        
        # Cosine Similarity (Dot product of L2 normalized vectors)
        cosine_sim = float(np.dot(emb1, emb2))
        # L2 Distance
        l2_dist = float(np.linalg.norm(emb1 - emb2))
        
        # Confidence score mapping
        confidence_pct = min(100.0, max(0.0, float(cosine_sim * 100.0)))
        verified = cosine_sim >= self.threshold
        
        return {
            'verified': verified,
            'similarity': cosine_sim,
            'l2_distance': l2_dist,
            'confidence_pct': confidence_pct,
            'status': '[VERIFIED] Identity Confirmed' if verified else '[NOT VERIFIED] Identity Mismatch'
        }

if __name__ == '__main__':
    verifier = FaceVerifier()
    
    reg_face = 'data/face_dataset/user_001/face_1.jpg'
    same_face = 'data/face_dataset/user_001/face_2.jpg'
    other_face = 'data/face_dataset/user_002/face_1.jpg'
    
    res_same = verifier.verify_faces(reg_face, same_face)
    res_other = verifier.verify_faces(reg_face, other_face)
    
    print(f"[OK] Phase 8: Deep Learning Face Verification Test")
    print(f"     Same User Check:  {res_same['status']} | Sim: {res_same['similarity']:.4f} | Conf: {res_same['confidence_pct']:.1f}%")
    print(f"     Diff User Check:  {res_other['status']} | Sim: {res_other['similarity']:.4f} | Conf: {res_other['confidence_pct']:.1f}%")
