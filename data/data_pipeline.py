import boto3
from torch.utils.data import Dataset, DataLoader
import io
from PIL import Image
import torch
from torchvision import transforms
from typing import Optional, List, Tuple
import concurrent.futures
from functools import lru_cache

class S3ImageDataset(Dataset):
    """
    Efficient data loading from S3 with caching and parallel downloads
    """
    def __init__(
        self, 
        bucket: str, 
        prefix: str, 
        transform: Optional[transforms.Compose] = None,
        cache_size: int = 1000,
        num_workers: int = 4
    ):
        self.s3 = boto3.client('s3')
        self.bucket = bucket
        self.prefix = prefix
        self.transform = transform or self.get_default_transforms()
        self.cache_size = cache_size
        self.num_workers = num_workers
        
        # List all images in S3
        self.image_keys = self._list_images()
        print(f"Found {len(self.image_keys)} images in s3://{bucket}/{prefix}")
        
    def get_default_transforms(self) -> transforms.Compose:
        """
        Default image transformations for CLIP model
        """
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.48145466, 0.4578275, 0.40821073],
                std=[0.26862954, 0.26130258, 0.27577711]
            )
        ])
        
    def _list_images(self) -> List[str]:
        """
        Retrieve all image paths from S3 bucket
        """
        paginator = self.s3.get_paginator('list_objects_v2')
        pages = paginator.paginate(Bucket=self.bucket, Prefix=self.prefix)
        
        image_keys = []
        valid_extensions = {'.jpg', '.jpeg', '.png', '.webp'}
        
        for page in pages:
            for obj in page.get('Contents', []):
                key = obj['Key']
                if any(key.lower().endswith(ext) for ext in valid_extensions):
                    image_keys.append(key)
                    
        return sorted(image_keys)  # Sort for reproducibility
    
    @lru_cache(maxsize=1000)
    def _load_image_from_s3(self, key: str) -> Image.Image:
        """
        Load and cache image from S3
        """
        obj = self.s3.get_object(Bucket=self.bucket, Key=key)
        img_bytes = obj['Body'].read()
        return Image.open(io.BytesIO(img_bytes)).convert('RGB')
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Load image and label
        """
        key = self.image_keys[idx]
        
        # Load image with caching
        img = self._load_image_from_s3(key)
        
        # Apply transforms
        if self.transform:
            img = self.transform(img)
            
        # Extract label from path structure
        # Assumes structure: prefix/ai_generated/... or prefix/real/...
        label = 1 if 'ai_generated' in key or 'synthetic' in key else 0
        
        return img, label
    
    def __len__(self) -> int:
        return len(self.image_keys)
    
    def get_labels_distribution(self) -> dict:
        """
        Get distribution of labels in dataset
        """
        ai_count = sum(1 for key in self.image_keys 
                      if 'ai_generated' in key or 'synthetic' in key)
        real_count = len(self.image_keys) - ai_count
        
        return {
            'ai_generated': ai_count,
            'real': real_count,
            'total': len(self.image_keys),
            'ai_percentage': (ai_count / len(self.image_keys)) * 100
        }

class LocalDataset(Dataset):
    """
    Dataset for local files (useful for development/testing)
    """
    def __init__(
        self,
        root_dir: str,
        transform: Optional[transforms.Compose] = None
    ):
        import os
        from pathlib import Path
        
        self.root_dir = Path(root_dir)
        self.transform = transform or S3ImageDataset.get_default_transforms(None)
        
        # Find all images
        self.image_paths = []
        self.labels = []
        
        # Assume directory structure: root_dir/ai_generated/* and root_dir/real/*
        for label, folder in enumerate(['real', 'ai_generated']):
            folder_path = self.root_dir / folder
            if folder_path.exists():
                for img_path in folder_path.glob('*'):
                    if img_path.suffix.lower() in {'.jpg', '.jpeg', '.png', '.webp'}:
                        self.image_paths.append(img_path)
                        self.labels.append(label)
        
        print(f"Found {len(self.image_paths)} local images")
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        img = Image.open(img_path).convert('RGB')
        
        if self.transform:
            img = self.transform(img)
            
        return img, label
    
    def __len__(self) -> int:
        return len(self.image_paths)

def create_data_loaders(
    bucket: str = None,
    train_prefix: str = 'train/',
    val_prefix: str = 'val/',
    local_root: str = None,
    batch_size: int = 32,
    num_workers: int = 4
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and validation data loaders
    """
    if bucket:
        # S3 datasets
        train_dataset = S3ImageDataset(bucket, train_prefix)
        val_dataset = S3ImageDataset(bucket, val_prefix)
    else:
        # Local datasets
        train_dataset = LocalDataset(f"{local_root}/train")
        val_dataset = LocalDataset(f"{local_root}/val")
    
    # Print dataset statistics
    if hasattr(train_dataset, 'get_labels_distribution'):
        print("\nTraining set distribution:")
        for k, v in train_dataset.get_labels_distribution().items():
            print(f"  {k}: {v}")
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader

if __name__ == "__main__":
    # Test data loading
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--bucket', type=str, help='S3 bucket name')
    parser.add_argument('--local', type=str, help='Local data path')
    
    args = parser.parse_args()
    
    if args.bucket:
        dataset = S3ImageDataset(args.bucket, 'train/')
    else:
        dataset = LocalDataset(args.local or './data')
    
    # Test loading a few images
    print("\nTesting data loading...")
    for i in range(min(3, len(dataset))):
        img, label = dataset[i]
        print(f"Image {i}: shape={img.shape}, label={label}")