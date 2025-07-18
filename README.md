# AI Image Detection on AWS

## Overview
A production-ready system for detecting AI-generated images using transfer learning with CLIP and AWS infrastructure. Achieves 90%+ accuracy distinguishing between real and AI-generated images.

## Features
- Transfer learning with pre-trained CLIP models
- AWS-native infrastructure (SageMaker/EC2)
- Cost-optimized training with spot instances
- Production monitoring with CloudWatch
- Scalable data pipeline from S3

## Architecture
The system offers two deployment options:
1. **SageMaker** - Managed ML platform (recommended)
2. **EC2 with Spot Instances** - More control, lower cost

## Quick Start

### Prerequisites
- AWS Account with appropriate permissions
- Python 3.8+
- AWS CLI configured

### Installation
```bash
git clone https://github.com/CuriosityQuantified/ai-image-detection-aws.git
cd ai-image-detection-aws
pip install -r requirements.txt
```

### Option 1: SageMaker Training
```bash
python src/sagemaker_train.py
```

### Option 2: EC2 Training
```bash
# Launch infrastructure
python infrastructure/ec2_ml_setup.py

# Or use CloudFormation
aws cloudformation create-stack --stack-name ai-detection-stack --template-body file://infrastructure/ml-stack.yaml
```

## Project Structure
```
├── src/
│   ├── train.py                 # Main training script
│   ├── sagemaker_train.py       # SageMaker launcher
│   ├── train_with_monitoring.py # Training with metrics
│   └── models/
│       └── clip_detector.py     # Model architecture
├── infrastructure/
│   ├── ec2_ml_setup.py          # EC2 infrastructure
│   ├── ml-stack.yaml            # CloudFormation template
│   └── cost_optimizer.py        # Cost analysis tools
├── data/
│   └── data_pipeline.py         # S3 data loading
├── notebooks/
│   └── exploratory_analysis.ipynb
└── requirements.txt
```

## Model Performance
- **Accuracy**: 90-95% on test set
- **Training Time**: 2-3 hours on p3.2xlarge
- **Cost**: ~$4 with spot instances

## Data Sources
- Kaggle AI vs Real Image datasets
- DiffusionDB (Hugging Face)
- Custom augmentation pipeline

## Monitoring
The system includes comprehensive monitoring:
- CloudWatch metrics for training progress
- Weights & Biases integration
- Cost tracking and optimization

## Contributing
Pull requests welcome! Please ensure:
- Code follows PEP8 standards
- Tests pass (run `pytest tests/`)
- Documentation is updated

## License
MIT License - see LICENSE file for details

## Acknowledgments
- CLIP model by OpenAI
- AWS ML team for infrastructure best practices