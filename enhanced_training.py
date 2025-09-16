# enhanced_training.py
import os
import yaml
from ultralytics import YOLO
import cv2
import numpy as np
from sklearn.model_selection import train_test_split

class EnhancedTrainer:
    def __init__(self):
        self.dataset_path = "/content/combined_dataset"
        self.model_size = "m"  # Switch to medium model for better performance
        
    def verify_labels(self):
        """Verify that all training images have proper labels"""
        print("🔍 Verifying label consistency...")
        
        images_dir = os.path.join(self.dataset_path, "train", "images")
        labels_dir = os.path.join(self.dataset_path, "train", "labels")
        
        issues_found = 0
        for img_file in os.listdir(images_dir):
            if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                label_file = os.path.splitext(img_file)[0] + '.txt'
                label_path = os.path.join(labels_dir, label_file)
                
                if not os.path.exists(label_path):
                    print(f"❌ Missing label for: {img_file}")
                    issues_found += 1
                else:
                    # Check if label file is empty
                    with open(label_path, 'r') as f:
                        content = f.read().strip()
                        if not content:
                            print(f"❌ Empty label file: {label_file}")
                            issues_found += 1
        
        if issues_found == 0:
            print("✅ All labels are consistent")
        else:
            print(f"⚠️  Found {issues_found} label issues")
        
        return issues_found
    
    def analyze_difficult_images(self):
        """Identify and analyze difficult crowd images"""
        difficult_images = []
        labels_dir = os.path.join(self.dataset_path, "train", "labels")
        
        for label_file in os.listdir(labels_dir):
            if label_file.endswith('.txt'):
                label_path = os.path.join(labels_dir, label_file)
                
                with open(label_path, 'r') as f:
                    lines = f.readlines()
                    
                if len(lines) >= 10:  # Images with many people (crowd scenes)
                    difficult_images.append({
                        'file': label_file.replace('.txt', '.jpg'),
                        'person_count': len(lines),
                        'small_objects': sum(1 for line in lines if float(line.split()[3]) < 0.05)  # Small bboxes
                    })
        
        # Sort by difficulty (most people first)
        difficult_images.sort(key=lambda x: x['person_count'], reverse=True)
        
        print(f"📊 Found {len(difficult_images)} difficult crowd images")
        for img in difficult_images[:5]:  # Show top 5 most difficult
            print(f"   {img['file']}: {img['person_count']} people, {img['small_objects']} small objects")
        
        return difficult_images
    
    def create_augmented_dataset(self, difficult_images):
        """Create augmented versions of difficult images"""
        print("🔄 Creating augmented dataset for difficult images...")
        
        aug_dir = os.path.join(self.dataset_path, "augmented")
        os.makedirs(aug_dir, exist_ok=True)
        
        # Select top 20 most difficult images for augmentation
        selected_images = difficult_images[:20]
        
        for img_info in selected_images:
            img_name = img_info['file']
            img_path = os.path.join(self.dataset_path, "train", "images", img_name)
            label_path = os.path.join(self.dataset_path, "train", "labels", img_name.replace('.jpg', '.txt'))
            
            if os.path.exists(img_path) and os.path.exists(label_path):
                # Read image
                img = cv2.imread(img_path)
                
                # Apply augmentations
                augmentations = self._generate_augmentations(img, img_path, label_path)
                
                for aug_name, aug_img, aug_labels in augmentations:
                    aug_img_path = os.path.join(aug_dir, f"aug_{aug_name}_{img_name}")
                    aug_label_path = os.path.join(aug_dir, f"aug_{aug_name}_{img_name.replace('.jpg', '.txt')}")
                    
                    cv2.imwrite(aug_img_path, aug_img)
                    with open(aug_label_path, 'w') as f:
                        f.write(aug_labels)
        
        print(f"✅ Created {len(selected_images) * 5} augmented samples")
    
    def _generate_augmentations(self, img, img_path, label_path):
        """Generate augmented versions of an image"""
        augmentations = []
        h, w = img.shape[:2]
        
        # Read original labels
        with open(label_path, 'r') as f:
            original_labels = f.read()
        
        # 1. Brightness adjustment
        bright_img = cv2.convertScaleAbs(img, alpha=1.2, beta=20)
        augmentations.append(('bright', bright_img, original_labels))
        
        # 2. Contrast adjustment
        contrast_img = cv2.convertScaleAbs(img, alpha=1.3, beta=0)
        augmentations.append(('contrast', contrast_img, original_labels))
        
        # 3. Gaussian blur
        blur_img = cv2.GaussianBlur(img, (5, 5), 0)
        augmentations.append(('blur', blur_img, original_labels))
        
        # 4. Horizontal flip (need to adjust labels)
        flip_img = cv2.flip(img, 1)
        flip_labels = self._flip_labels(original_labels, w)
        augmentations.append(('flip', flip_img, flip_labels))
        
        # 5. Random crop (simulate different viewpoints)
        crop_percent = 0.8
        crop_h, crop_w = int(h * crop_percent), int(w * crop_percent)
        start_y, start_x = np.random.randint(0, h - crop_h), np.random.randint(0, w - crop_w)
        crop_img = img[start_y:start_y+crop_h, start_x:start_x+crop_w]
        crop_labels = self._crop_labels(original_labels, start_x/w, start_y/h, crop_w/w, crop_h/h)
        augmentations.append(('crop', crop_img, crop_labels))
        
        return augmentations
    
    def _flip_labels(self, labels, img_width):
        """Flip labels horizontally"""
        new_labels = []
        for line in labels.strip().split('\n'):
            if line:
                cls, x_center, y_center, width, height = map(float, line.split())
                x_center = 1.0 - x_center  # Flip horizontally
                new_labels.append(f"{int(cls)} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
        return '\n'.join(new_labels)
    
    def _crop_labels(self, labels, start_x, start_y, crop_w, crop_h):
        """Adjust labels for cropped image"""
        new_labels = []
        for line in labels.strip().split('\n'):
            if line:
                cls, x_center, y_center, width, height = map(float, line.split())
                
                # Adjust coordinates for crop
                x_center = (x_center - start_x) / crop_w
                y_center = (y_center - start_y) / crop_h
                width = width / crop_w
                height = height / crop_h
                
                # Only keep labels that are mostly within the crop
                if (0 <= x_center <= 1 and 0 <= y_center <= 1 and 
                    width > 0.02 and height > 0.02):  # Minimum size threshold
                    new_labels.append(f"{int(cls)} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
        
        return '\n'.join(new_labels)
    
    def train_enhanced_model(self):
        """Train an enhanced model with better settings"""
        print("🚀 Starting enhanced training...")
        
        # Use medium model for better performance
        model = YOLO(f"yolov8{self.model_size}.pt")
        
        # Training configuration
        results = model.train(
            data=os.path.join(self.dataset_path, "combined.yaml"),
            epochs=150,  # More epochs for better learning
            imgsz=640,
            batch=16,
            device=0,
            patience=30,  # Longer patience for complex learning
            lr0=0.01,     # Learning rate
            augment=True,  # Enable built-in augmentation
            mosaic=0.5,   # Mosaic augmentation probability
            mixup=0.1,    # Mixup augmentation
            hsv_h=0.015,  # HSV augmentation - Hue
            hsv_s=0.7,    # HSV augmentation - Saturation
            hsv_v=0.4,    # HSV augmentation - Value
            degrees=10.0, # Rotation augmentation
            translate=0.1, # Translation augmentation
            scale=0.5,    # Scale augmentation
            shear=2.0,    # Shear augmentation
            perspective=0.0005, # Perspective augmentation
            fliplr=0.5,   # Horizontal flip probability
            name=f"enhanced_{self.model_size}_model"
        )
        
        return results

# Main execution
if __name__ == "__main__":
    trainer = EnhancedTrainer()
    
    # Step 1: Verify labels
    trainer.verify_labels()
    
    # Step 2: Analyze difficult images
    difficult_images = trainer.analyze_difficult_images()
    
    # Step 3: Create augmented dataset for difficult images
    trainer.create_augmented_dataset(difficult_images)
    
    # Step 4: Train enhanced model
    trainer.train_enhanced_model()