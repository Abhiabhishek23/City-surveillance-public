# train/complete_training.py
import torch
import torchvision
from torch.utils.data import DataLoader
from torch.optim import SGD
from torch.optim.lr_scheduler import StepLR
from sklearn.metrics import precision_score, recall_score, f1_score
import numpy as np
from dataset import ConstructionDataset, get_train_transform, get_valid_transform

def train_model_with_validation():
    # Configuration
    num_classes = 2  # background + construction
    batch_size = 2
    num_epochs = 30
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Datasets
    train_dataset = ConstructionDataset("data/train", transforms=get_train_transform())
    val_dataset = ConstructionDataset("data/val", transforms=get_valid_transform())
    
    # Data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                            collate_fn=lambda x: tuple(zip(*x)))
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False,
                          collate_fn=lambda x: tuple(zip(*x)))
    
    # Model
    model = maskrcnn_resnet50_fpn(pretrained=True)
    model.to(device)
    
    # Optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.1)
    
    # Training loop
    best_val_loss = float('inf')
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0.0
        for images, targets in train_loader:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            
            train_loss += losses.item()
        
        # Validation
        model.eval()
        val_loss = 0.0
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for images, targets in val_loader:
                images = [img.to(device) for img in images]
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                
                # Get predictions
                predictions = model(images)
                
                # Calculate loss
                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
                val_loss += losses.item()
                
                # Calculate metrics
                for i, pred in enumerate(predictions):
                    # Convert predictions to binary labels
                    pred_labels = [1 if score > 0.5 else 0 for score in pred['scores']]
                    true_labels = [1] * len(targets[i]['labels'])  # All are construction
                    
                    all_preds.extend(pred_labels)
                    all_targets.extend(true_labels)
        
        # Calculate metrics
        precision = precision_score(all_targets, all_preds, zero_division=0)
        recall = recall_score(all_targets, all_preds, zero_division=0)
        f1 = f1_score(all_targets, all_preds, zero_division=0)
        
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), "models/best_construction_model.pth")
            print("Best model saved!")
        
        scheduler.step()
    
    print("Training completed!")
    return model

if __name__ == "__main__":
    train_model_with_validation()