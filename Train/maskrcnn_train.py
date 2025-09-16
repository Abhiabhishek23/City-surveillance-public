# train/maskrcnn_enhanced.py
import torch
import torchvision
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torch.utils.data import DataLoader
from torch.optim import SGD
from torch.optim.lr_scheduler import StepLR
import os
import time
from datetime import datetime
from .dataset import ConstructionDataset, get_train_transform, get_valid_transform

def get_model(num_classes, pretrained=True):
    """Initialize Mask R-CNN model with custom classifier"""
    model = maskrcnn_resnet50_fpn(pretrained=pretrained)
    
    # Replace the box predictor
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(
        in_features, num_classes
    )
    
    # Replace the mask predictor
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = torchvision.models.detection.mask_rcnn.MaskRCNNPredictor(
        in_features_mask, 256, num_classes
    )
    
    return model

def train_model():
    # Configuration
    num_classes = 2  # background + construction
    batch_size = 2
    num_epochs = 25
    learning_rate = 0.005
    momentum = 0.9
    weight_decay = 0.0005
    step_size = 7
    gamma = 0.1
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on device: {device}")
    
    # Create datasets
    train_dataset = ConstructionDataset("data/train", transforms=get_train_transform())
    valid_dataset = ConstructionDataset("data/valid", transforms=get_valid_transform())
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        collate_fn=lambda x: tuple(zip(*x)), num_workers=2
    )
    
    valid_loader = DataLoader(
        valid_dataset, batch_size=1, shuffle=False,
        collate_fn=lambda x: tuple(zip(*x)), num_workers=2
    )
    
    # Initialize model
    model = get_model(num_classes)
    model.to(device)
    
    # Define optimizer and scheduler
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = SGD(params, lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
    scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)
    
    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'learning_rate': []
    }
    
    # Create models directory
    os.makedirs("models", exist_ok=True)
    
    # Training loop
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        
        # Training phase
        model.train()
        train_loss = 0.0
        
        for batch_idx, (images, targets) in enumerate(train_loader):
            # Move images and targets to device
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            
            # Forward pass
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            
            # Backward pass and optimize
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            
            train_loss += losses.item()
            
            if batch_idx % 10 == 0:
                print(f"Batch {batch_idx}/{len(train_loader)}, Loss: {losses.item():.4f}")
        
        # Update learning rate
        scheduler.step()
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for images, targets in valid_loader:
                images = [img.to(device) for img in images]
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                
                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
                val_loss += losses.item()
        
        # Calculate average losses
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(valid_loader)
        
        # Update history
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['learning_rate'].append(scheduler.get_last_lr()[0])
        
        print(f"Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, LR: {scheduler.get_last_lr()[0]:.6f}")
        
        # Save model checkpoint
        if (epoch + 1) % 5 == 0:
            checkpoint_path = f"models/maskrcnn_epoch_{epoch+1}.pth"
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Checkpoint saved: {checkpoint_path}")
    
    # Save final model
    final_model_path = "models/construction_detection_model.pth"
    torch.save(model.state_dict(), final_model_path)
    print(f"Training completed. Final model saved: {final_model_path}")
    
    # Save training history
    history_path = "models/training_history.json"
    with open(history_path, 'w') as f:
        import json
        json.dump(history, f, indent=2)
    
    return model, history

if __name__ == "__main__":
    train_model()