import os
import cv2
import numpy as np
from pathlib import Path
from datetime import datetime

class FaceReferenceManager:
    """مدیریت تصاویر مرجع چهره‌ها برای شناسایی بهتر"""
    
    def __init__(self, reference_dir="FaceRecognation/reference_faces"):
        self.reference_dir = Path(reference_dir)
        self.reference_dir.mkdir(parents=True, exist_ok=True)
        self.reference_images = {}  # {face_id: image_path}
        self.reference_embeddings = {}  # {face_id: embedding}
        self.clear_all_references()  # Clear old snapshots on startup
        
    def save_reference_image(self, face_id, face_image, face_embedding):
        """ذخیره تصویر مرجع برای یک چهره"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        image_path = self.reference_dir / f"face_{face_id}_{timestamp}.jpg"
        
        # ذخیره تصویر
        cv2.imwrite(str(image_path), face_image)
        
        # ذخیره اطلاعات
        self.reference_images[face_id] = str(image_path)
        self.reference_embeddings[face_id] = face_embedding.copy()
        
        return str(image_path)
    
    def load_reference_image(self, face_id):
        """بارگذاری تصویر مرجع یک چهره"""
        if face_id in self.reference_images:
            image_path = self.reference_images[face_id]
            if os.path.exists(image_path):
                return cv2.imread(image_path)
        return None
    
    def get_reference_embedding(self, face_id):
        """دریافت embedding تصویر مرجع"""
        return self.reference_embeddings.get(face_id, None)
    
    def compare_with_references(self, face_embedding, threshold=0.5):
        """مقایسه embedding با تصاویر مرجع و برگرداندن نزدیک‌ترین ID"""
        if len(self.reference_embeddings) == 0:
            return None, float('inf')
        
        min_distance = float('inf')
        closest_id = None
        
        for ref_id, ref_embedding in self.reference_embeddings.items():
            # محاسبه فاصله اقلیدسی
            distance = np.sqrt(np.sum((face_embedding - ref_embedding) ** 2))
            
            if distance < min_distance:
                min_distance = distance
                closest_id = ref_id
        
        # اگر فاصله کمتر از threshold باشد, همان ID را برمی‌گردانیم
        if min_distance <= threshold:
            return closest_id, min_distance
        
        return None, min_distance
    
    def has_reference(self, face_id):
        """بررسی وجود تصویر مرجع برای یک ID"""
        return face_id in self.reference_images
    
    def get_all_reference_ids(self):
        """دریافت لیست تمام IDهای دارای تصویر مرجع"""
        return list(self.reference_images.keys())
    
    def remove_reference(self, face_id):
        """حذف تصویر مرجع یک چهره"""
        if face_id in self.reference_images:
            image_path = self.reference_images[face_id]
            if os.path.exists(image_path):
                os.remove(image_path)
            del self.reference_images[face_id]
            if face_id in self.reference_embeddings:
                del self.reference_embeddings[face_id]

    def clear_all_references(self):
        """حذف تمام تصاویر مرجع از پوشه و پاک کردن memory"""
        deleted_count = 0
        if self.reference_dir.exists():
            for file_path in self.reference_dir.iterdir():
                if file_path.is_file() and file_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                    try:
                        os.remove(file_path)
                        deleted_count += 1
                    except Exception as e:
                        print(f"[-] Error deleting file {file_path}: {e}")
        self.reference_images.clear()
        self.reference_embeddings.clear()
        print(f"[✓] Cleared {deleted_count} old snapshot files from {self.reference_dir}")

