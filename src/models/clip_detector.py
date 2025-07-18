import torch
import torch.nn as nn
from transformers import CLIPModel, CLIPProcessor
from typing import Optional, Dict, List
import numpy as np

class CLIPBasedDetector(nn.Module):
    """
    Advanced CLIP-based detector with multiple detection strategies
    """
    def __init__(
        self,
        model_name: str = "openai/clip-vit-large-patch14",
        freeze_encoder: bool = True,
        use_attention: bool = True
    ):
        super().__init__()
        
        # Load pre-trained CLIP
        self.encoder = CLIPModel.from_pretrained(model_name)
        self.hidden_size = self.encoder.config.projection_dim
        
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
        
        # Multi-head attention for feature refinement
        self.use_attention = use_attention
        if use_attention:
            self.attention = nn.MultiheadAttention(
                embed_dim=self.hidden_size,
                num_heads=8,
                dropout=0.1
            )
            
        # Classification head with residual connections
        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_size, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 2)
        )
        
        # Auxiliary heads for multi-task learning
        self.artifact_detector = nn.Linear(self.hidden_size, 1)  # Detect specific artifacts
        self.quality_assessor = nn.Linear(self.hidden_size, 1)   # Assess image quality
        
    def forward(
        self,
        pixel_values: torch.Tensor,
        return_features: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with multiple outputs
        """
        # Extract CLIP features
        outputs = self.encoder.vision_model(pixel_values=pixel_values)
        features = outputs.last_hidden_state  # [batch, seq_len, hidden_size]
        
        # Global pooling
        pooled_features = features.mean(dim=1)  # [batch, hidden_size]
        
        # Apply attention if enabled
        if self.use_attention:
            # Self-attention over spatial features
            features_t = features.transpose(0, 1)  # [seq_len, batch, hidden_size]
            attended_features, _ = self.attention(
                features_t, features_t, features_t
            )
            attended_features = attended_features.transpose(0, 1)
            pooled_features = attended_features.mean(dim=1)
        
        # Main classification
        logits = self.classifier(pooled_features)
        
        # Auxiliary outputs
        artifact_score = self.artifact_detector(pooled_features)
        quality_score = self.quality_assessor(pooled_features)
        
        outputs = {
            'logits': logits,
            'artifact_score': artifact_score,
            'quality_score': quality_score
        }
        
        if return_features:
            outputs['features'] = pooled_features
            
        return outputs

class EnsembleDetector(nn.Module):
    """
    Ensemble of multiple detection models
    """
    def __init__(self, models: List[nn.Module]):
        super().__init__()
        self.models = nn.ModuleList(models)
        self.fusion = nn.Linear(len(models) * 2, 2)
        
    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Ensemble prediction with learned fusion
        """
        all_outputs = []
        
        for model in self.models:
            with torch.no_grad():
                output = model(pixel_values)
                all_outputs.append(output['logits'])
        
        # Concatenate all predictions
        combined = torch.cat(all_outputs, dim=1)
        
        # Learned fusion
        final_logits = self.fusion(combined)
        
        return {'logits': final_logits}

class FrequencyAnalyzer(nn.Module):
    """
    Analyze frequency domain for AI artifacts
    """
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(128, 2)
        
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Analyze frequency patterns
        """
        # Convert to grayscale
        gray = images.mean(dim=1, keepdim=True)
        
        # Apply FFT
        fft = torch.fft.fft2(gray)
        magnitude = torch.abs(fft)
        
        # Log scale for better feature extraction
        log_magnitude = torch.log1p(magnitude)
        
        # CNN on frequency domain
        x = torch.relu(self.conv1(log_magnitude))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = self.pool(x).squeeze(-1).squeeze(-1)
        
        return self.fc(x)

if __name__ == "__main__":
    # Test model
    model = CLIPBasedDetector()
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # Test forward pass
    dummy_input = torch.randn(4, 3, 224, 224)
    outputs = model(dummy_input)
    
    print("\nOutput shapes:")
    for key, value in outputs.items():
        print(f"  {key}: {value.shape}")