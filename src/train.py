import torch
import torch.nn as nn
from transformers import CLIPModel
import argparse
from datetime import datetime
import wandb
from torch.utils.data import DataLoader
from tqdm import tqdm
import boto3

class CLIPBasedDetector(nn.Module):
    """
    Transfer learning using CLIP for AI image detection
    """
    def __init__(self, freeze_encoder=True):
        super().__init__()
        # Load pre-trained CLIP
        self.encoder = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
        
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
                
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 2)
        )
        
    def forward(self, images):
        # Extract image features
        features = self.encoder.get_image_features(pixel_values=images)
        # Classify
        return self.classifier(features)

def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, (images, labels) in enumerate(tqdm(dataloader)):
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
    return total_loss / len(dataloader), 100. * correct / total

def validate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in tqdm(dataloader):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
    return total_loss / len(dataloader), 100. * correct / total

def train_model(args):
    """
    Main training loop with AWS integration
    """
    # Initialize wandb
    wandb.init(
        project="ai-image-detection",
        config=vars(args)
    )
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Initialize model
    model = CLIPBasedDetector(freeze_encoder=True).to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.learning_rate,
        weight_decay=0.01
    )
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs
    )
    
    # Training loop
    best_acc = 0
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device
        )
        
        # Validate
        val_loss, val_acc = validate(
            model, val_loader, criterion, device
        )
        
        # Update scheduler
        scheduler.step()
        
        # Log metrics
        wandb.log({
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_acc': val_acc,
            'learning_rate': optimizer.param_groups[0]['lr']
        })
        
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_acc': best_acc,
            }, f'checkpoints/best_model.pt')
            
            # Upload to S3
            if args.s3_bucket:
                s3 = boto3.client('s3')
                s3.upload_file(
                    'checkpoints/best_model.pt',
                    args.s3_bucket,
                    f'models/{args.run_name}/best_model.pt'
                )
    
    print(f"\nTraining complete! Best accuracy: {best_acc:.2f}%")
    wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--data_path', type=str, default='./data')
    parser.add_argument('--s3_bucket', type=str, default=None)
    parser.add_argument('--run_name', type=str, default=datetime.now().strftime('%Y%m%d_%H%M%S'))
    
    args = parser.parse_args()
    train_model(args)