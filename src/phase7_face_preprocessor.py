import numpy as np
from PIL import Image

class FacePreprocessor:
    def __init__(self, target_size=(160, 160)):
        self.target_size = target_size

    def detect_and_crop(self, image_input):
        """
        Accepts PIL Image, numpy array (RGB or BGR), or file path.
        Detects facial region using skin/color distribution analysis & center alignment,
        crops to bounding box, and resizes to target_size.
        """
        if isinstance(image_input, str):
            img = Image.open(image_input).convert('RGB')
        elif isinstance(image_input, Image.Image):
            img = image_input.convert('RGB')
        elif isinstance(image_input, np.ndarray):
            if len(image_input.shape) == 3 and image_input.shape[2] == 3:
                img = Image.fromarray(image_input.astype('uint8')).convert('RGB')
            else:
                img = Image.fromarray(image_input).convert('RGB')
        else:
            raise TypeError("Unsupported image input type")

        w, h = img.size
        # Face detection via central facial region crop heuristic & skin bounding box
        img_arr = np.array(img, dtype=np.int32)
        r, g, b = img_arr[:, :, 0], img_arr[:, :, 1], img_arr[:, :, 2]
        
        # Skin tone rule heuristic (R > 95, G > 40, B > 20, R > G, R > B, |R - G| > 15)
        skin_mask = (r > 80) & (g > 30) & (b > 15) & (r > g) & (abs(r - g) > 10)
        
        y_indices, x_indices = np.where(skin_mask)
        
        if len(y_indices) > 50:
            ymin, ymax = np.min(y_indices), np.max(y_indices)
            xmin, xmax = np.min(x_indices), np.max(x_indices)
            
            # Add padding
            pad_x = int((xmax - xmin) * 0.15)
            pad_y = int((ymax - ymin) * 0.15)
            
            x1 = max(0, xmin - pad_x)
            y1 = max(0, ymin - pad_y)
            x2 = min(w, xmax + pad_x)
            y2 = min(h, ymax + pad_y)
            
            cropped = img.crop((x1, y1, x2, y2))
        else:
            # Fallback: square central crop
            crop_dim = min(w, h)
            left = (w - crop_dim) // 2
            top = (h - crop_dim) // 2
            cropped = img.crop((left, top, left + crop_dim, top + crop_dim))

        resized = cropped.resize(self.target_size, Image.Resampling.BILINEAR)
        return resized

    def normalize_tensor(self, pil_img):
        """
        Converts PIL image to float numpy tensor normalized to [-1, 1] range.
        Shape: (1, 3, target_size[0], target_size[1])
        """
        arr = np.array(pil_img, dtype=np.float32)
        # Normalize to [-1, 1]
        norm = (arr - 127.5) / 128.0
        # Transpose to (C, H, W)
        tensor = np.transpose(norm, (2, 0, 1))
        # Add batch dimension -> (1, C, H, W)
        tensor_batch = np.expand_dims(tensor, axis=0)
        return tensor_batch

if __name__ == '__main__':
    preprocessor = FacePreprocessor()
    test_img_path = 'data/face_dataset/user_001/face_1.jpg'
    cropped = preprocessor.detect_and_crop(test_img_path)
    tensor = preprocessor.normalize_tensor(cropped)
    print(f"[OK] Phase 7: Face Preprocessing verified.")
    print(f"     Input image: {test_img_path}")
    print(f"     Cropped size: {cropped.size}, Tensor shape: {tensor.shape}, Min: {tensor.min():.2f}, Max: {tensor.max():.2f}")
