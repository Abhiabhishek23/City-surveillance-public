# tools/prepare_dataset.py
import os
import json
import shutil
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from PIL import Image
import argparse

def organize_dataset(raw_data_dir, output_dir, train_ratio=0.8):
    """
    Organize raw data into training and validation sets
    """
    # Create directory structure
    directories = [
        os.path.join(output_dir, 'train', 'images'),
        os.path.join(output_dir, 'train', 'annotations'),
        os.path.join(output_dir, 'val', 'images'),
        os.path.join(output_dir, 'val', 'annotations'),
        os.path.join(output_dir, 'test', 'images'),
        os.path.join(output_dir, 'test', 'annotations')
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    
    # Collect all image-annotation pairs
    image_files = []
    annotation_files = []
    
    for file in os.listdir(raw_data_dir):
        if file.lower().endswith(('.jpg', '.jpeg', '.png')):
            base_name = os.path.splitext(file)[0]
            annotation_file = f"{base_name}.json"
            
            if os.path.exists(os.path.join(raw_data_dir, annotation_file)):
                image_files.append(file)
                annotation_files.append(annotation_file)
    
    # Split dataset
    train_images, test_images, train_anns, test_anns = train_test_split(
        image_files, annotation_files, test_size=1-train_ratio, random_state=42
    )
    
    val_images, test_images, val_anns, test_anns = train_test_split(
        test_images, test_anns, test_size=0.5, random_state=42
    )
    
    # Copy files to appropriate directories
    def copy_files(files, ann_files, dest_dir):
        for img_file, ann_file in zip(files, ann_files):
            shutil.copy2(os.path.join(raw_data_dir, img_file), 
                        os.path.join(dest_dir, 'images', img_file))
            shutil.copy2(os.path.join(raw_data_dir, ann_file), 
                        os.path.join(dest_dir, 'annotations', ann_file))
    
    copy_files(train_images, train_anns, os.path.join(output_dir, 'train'))
    copy_files(val_images, val_anns, os.path.join(output_dir, 'val'))
    copy_files(test_images, test_anns, os.path.join(output_dir, 'test'))
    
    print(f"Dataset organized:")
    print(f"  Training: {len(train_images)} images")
    print(f"  Validation: {len(val_images)} images")
    print(f"  Test: {len(test_images)} images")
    print(f"Dataset saved to: {output_dir}")

def convert_labelme_to_coco(labelme_dir, output_file):
    """
    Convert LabelMe annotations to COCO format
    """
    coco_data = {
        "info": {
            "description": "Construction Detection Dataset",
            "version": "1.0",
            "year": 2023,
            "contributor": "City Surveillance System",
            "date_created": "2023-01-01"
        },
        "licenses": [{
            "id": 1,
            "name": "Academic Use",
            "url": ""
        }],
        "categories": [
            {"id": 1, "name": "building", "supercategory": "construction"},
            {"id": 2, "name": "construction", "supercategory": "construction"},
            {"id": 3, "name": "house", "supercategory": "construction"}
        ],
        "images": [],
        "annotations": []
    }
    
    annotation_id = 1
    image_id = 1
    
    for ann_file in os.listdir(labelme_dir):
        if ann_file.endswith('.json'):
            with open(os.path.join(labelme_dir, ann_file)) as f:
                data = json.load(f)
            
            # Add image info
            image_info = {
                "id": image_id,
                "width": data['imageWidth'],
                "height": data['imageHeight'],
                "file_name": data['imagePath'].split('/')[-1] if 'imagePath' in data else ann_file.replace('.json', '.jpg'),
                "license": 1,
                "date_captured": "2023-01-01"
            }
            coco_data['images'].append(image_info)
            
            # Add annotations
            for shape in data['shapes']:
                if shape['label'] in ['building', 'construction', 'house']:
                    # Convert polygon to COCO format
                    points = np.array(shape['points']).flatten().tolist()
                    
                    # Calculate bounding box
                    x_coords = points[::2]
                    y_coords = points[1::2]
                    x_min, x_max = min(x_coords), max(x_coords)
                    y_min, y_max = min(y_coords), max(y_coords)
                    width = x_max - x_min
                    height = y_max - y_min
                    
                    # Map label to category ID
                    if shape['label'] == 'building':
                        category_id = 1
                    elif shape['label'] == 'construction':
                        category_id = 2
                    elif shape['label'] == 'house':
                        category_id = 3
                    
                    # Calculate area
                    area = width * height
                    
                    coco_data['annotations'].append({
                        "id": annotation_id,
                        "image_id": image_id,
                        "category_id": category_id,
                        "segmentation": [points],
                        "area": area,
                        "bbox": [x_min, y_min, width, height],
                        "iscrowd": 0
                    })
                    
                    annotation_id += 1
            
            image_id += 1
    
    # Save COCO format annotations
    with open(output_file, 'w') as f:
        json.dump(coco_data, f, indent=2)
    
    print(f"COCO format annotations saved to: {output_file}")
    print(f"  Images: {len(coco_data['images'])}")
    print(f"  Annotations: {len(coco_data['annotations'])}")

def create_augmented_dataset(original_dir, output_dir, augmentations=5):
    """
    Create augmented dataset for training
    """
    from albumentations import (
        HorizontalFlip, VerticalFlip, RandomRotate90, ShiftScaleRotate,
        RandomBrightnessContrast, Blur, OpticalDistortion, GridDistortion,
        ElasticTransform, Compose
    )
    
    # Create output directories
    os.makedirs(os.path.join(output_dir, 'images'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'annotations'), exist_ok=True)
    
    # Define augmentation pipeline
    augment_pipeline = Compose([
        HorizontalFlip(p=0.5),
        VerticalFlip(p=0.2),
        RandomRotate90(p=0.3),
        ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5),
        RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        Blur(blur_limit=3, p=0.3),
    ], bbox_params={'format': 'pascal_voc', 'label_fields': ['category_ids']})
    
    # Process each image
    for img_file in os.listdir(os.path.join(original_dir, 'images')):
        if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
            base_name = os.path.splitext(img_file)[0]
            ann_file = f"{base_name}.json"
            
            # Load original image and annotation
            img_path = os.path.join(original_dir, 'images', img_file)
            ann_path = os.path.join(original_dir, 'annotations', ann_file)
            
            if not os.path.exists(ann_path):
                continue
            
            image = cv2.imread(img_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            with open(ann_path) as f:
                annotation = json.load(f)
            
            # Extract bounding boxes and labels
            bboxes = []
            category_ids = []
            
            for shape in annotation['shapes']:
                if shape['label'] in ['building', 'construction', 'house']:
                    points = np.array(shape['points'])
                    x_min, y_min = np.min(points, axis=0)
                    x_max, y_max = np.max(points, axis=0)
                    bboxes.append([x_min, y_min, x_max, y_max])
                    
                    if shape['label'] == 'building':
                        category_ids.append(1)
                    elif shape['label'] == 'construction':
                        category_ids.append(2)
                    elif shape['label'] == 'house':
                        category_ids.append(3)
            
            # Create augmented versions
            for i in range(augmentations):
                augmented = augment_pipeline(
                    image=image,
                    bboxes=bboxes,
                    category_ids=category_ids
                )
                
                # Save augmented image
                aug_img_name = f"{base_name}_aug{i}.jpg"
                aug_img_path = os.path.join(output_dir, 'images', aug_img_name)
                cv2.imwrite(aug_img_path, cv2.cvtColor(augmented['image'], cv2.COLOR_RGB2BGR))
                
                # Create augmented annotation
                aug_annotation = annotation.copy()
                aug_annotation['shapes'] = []
                
                for j, bbox in enumerate(augmented['bboxes']):
                    x_min, y_min, x_max, y_max = bbox
                    category_id = augmented['category_ids'][j]
                    
                    # Map category ID back to label
                    if category_id == 1:
                        label = "building"
                    elif category_id == 2:
                        label = "construction"
                    elif category_id == 3:
                        label = "house"
                    else:
                        continue
                    
                    # Create polygon from bounding box
                    polygon = [
                        [x_min, y_min],
                        [x_max, y_min],
                        [x_max, y_max],
                        [x_min, y_max]
                    ]
                    
                    aug_annotation['shapes'].append({
                        "label": label,
                        "points": polygon,
                        "shape_type": "polygon"
                    })
                
                # Save augmented annotation
                aug_ann_name = f"{base_name}_aug{i}.json"
                aug_ann_path = os.path.join(output_dir, 'annotations', aug_ann_name)
                
                with open(aug_ann_path, 'w') as f:
                    json.dump(aug_annotation, f, indent=2)
    
    print(f"Augmented dataset created in: {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare dataset for construction detection")
    parser.add_argument("--raw-dir", required=True, help="Directory with raw images and annotations")
    parser.add_argument("--output-dir", required=True, help="Output directory for organized dataset")
    parser.add_argument("--coco-output", help="Output file for COCO format annotations")
    parser.add_argument("--augment", action="store_true", help="Create augmented dataset")
    parser.add_argument("--train-ratio", type=float, default=0.8, help="Ratio of training data")
    
    args = parser.parse_args()
    
    # Organize dataset
    organize_dataset(args.raw_dir, args.output_dir, args.train_ratio)
    
    # Convert to COCO format if requested
    if args.coco_output:
        convert_labelme_to_coco(os.path.join(args.output_dir, 'train', 'annotations'), args.coco_output)
    
    # Create augmented dataset if requested
    if args.augment:
        augmented_dir = os.path.join(args.output_dir, 'augmented')
        create_augmented_dataset(os.path.join(args.output_dir, 'train'), augmented_dir)
        
        # Merge augmented dataset with original training data
        merged_dir = os.path.join(args.output_dir, 'train_augmented')
        os.makedirs(merged_dir, exist_ok=True)
        os.makedirs(os.path.join(merged_dir, 'images'), exist_ok=True)
        os.makedirs(os.path.join(merged_dir, 'annotations'), exist_ok=True)
        
        # Copy original training files
        for file_type in ['images', 'annotations']:
            src_dir = os.path.join(args.output_dir, 'train', file_type)
            dst_dir = os.path.join(merged_dir, file_type)
            
            for file in os.listdir(src_dir):
                shutil.copy2(os.path.join(src_dir, file), os.path.join(dst_dir, file))
        
        # Copy augmented files
        for file_type in ['images', 'annotations']:
            src_dir = os.path.join(augmented_dir, file_type)
            dst_dir = os.path.join(merged_dir, file_type)
            
            for file in os.listdir(src_dir):
                shutil.copy2(os.path.join(src_dir, file), os.path.join(dst_dir, file))
        
        print(f"Merged dataset created in: {merged_dir}")