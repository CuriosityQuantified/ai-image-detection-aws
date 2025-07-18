import torch
import boto3
import json
from datetime import datetime
import wandb
from torch.utils.tensorboard import SummaryWriter
from typing import Dict, Any
import os

class AIDetectorTrainer:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.s3 = boto3.client('s3')
        self.cloudwatch = boto3.client('cloudwatch')
        
        # Initialize experiment tracking
        wandb.init(project="ai-image-detection", config=config)
        self.writer = SummaryWriter(f"runs/{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        
        # Cost tracking
        self.start_time = datetime.now()
        self.instance_type = os.environ.get('INSTANCE_TYPE', 'ml.p3.2xlarge')
        
    def log_metrics_to_cloudwatch(self, metrics: Dict[str, float], epoch: int):
        """
        Push metrics to CloudWatch for monitoring and alerting
        """
        metric_data = []
        for metric_name, value in metrics.items():
            metric_data.append({
                'MetricName': metric_name,
                'Value': value,
                'Unit': 'None',
                'Dimensions': [
                    {'Name': 'Model', 'Value': self.config.get('model', 'clip')},
                    {'Name': 'Epoch', 'Value': str(epoch)},
                    {'Name': 'RunID', 'Value': self.config.get('run_id', 'default')}
                ]
            })
        
        # Batch send metrics
        if metric_data:
            self.cloudwatch.put_metric_data(
                Namespace='ML/ImageDetection',
                MetricData=metric_data
            )
    
    def log_cost_metrics(self):
        """
        Track and log training costs
        """
        elapsed_hours = (datetime.now() - self.start_time).total_seconds() / 3600
        
        # Approximate costs based on instance type
        hourly_costs = {
            'ml.p3.2xlarge': 3.06,
            'ml.g4dn.xlarge': 0.736,
            'ml.p3.8xlarge': 12.24
        }
        
        spot_discount = 0.3  # 70% savings with spot
        cost = elapsed_hours * hourly_costs.get(self.instance_type, 3.06) * spot_discount
        
        self.cloudwatch.put_metric_data(
            Namespace='ML/Costs',
            MetricData=[{
                'MetricName': 'TrainingCost',
                'Value': cost,
                'Unit': 'None',
                'Dimensions': [
                    {'Name': 'Project', 'Value': 'AIDetection'},
                    {'Name': 'InstanceType', 'Value': self.instance_type}
                ]
            }]
        )
        
        return cost
    
    def save_checkpoint_to_s3(self, model, epoch: int, metrics: Dict[str, float]):
        """
        Save model checkpoints with versioning and metadata
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'metrics': metrics,
            'timestamp': datetime.now().isoformat(),
            'config': self.config,
            'training_cost': self.log_cost_metrics()
        }
        
        # Save locally first
        local_path = f"/tmp/checkpoint_epoch_{epoch}.pt"
        torch.save(checkpoint, local_path)
        
        # Upload to S3 with metadata
        s3_key = f"models/ai-detection/{self.config['run_id']}/checkpoint_epoch_{epoch}.pt"
        
        metadata = {
            'accuracy': str(metrics.get('accuracy', 0)),
            'loss': str(metrics.get('loss', 0)),
            'model_type': self.config.get('model', 'clip'),
            'training_date': datetime.now().isoformat(),
            'epoch': str(epoch)
        }
        
        self.s3.upload_file(
            local_path, 
            'ml-model-artifacts',
            s3_key,
            ExtraArgs={
                'Metadata': metadata,
                'ServerSideEncryption': 'AES256'
            }
        )
        
        # Clean up local file
        os.remove(local_path)
        
        print(f"Checkpoint saved to s3://ml-model-artifacts/{s3_key}")
    
    def create_model_registry_entry(self, model_path: str, metrics: Dict[str, float]):
        """
        Register model in a simple S3-based registry
        """
        registry_entry = {
            'model_id': f"{self.config['run_id']}_final",
            'created_at': datetime.now().isoformat(),
            'model_path': model_path,
            'metrics': metrics,
            'config': self.config,
            'framework': 'pytorch',
            'framework_version': torch.__version__,
            'status': 'ready'
        }
        
        # Save registry entry
        registry_key = f"model-registry/ai-detection/{self.config['run_id']}.json"
        
        self.s3.put_object(
            Bucket='ml-model-artifacts',
            Key=registry_key,
            Body=json.dumps(registry_entry, indent=2),
            ContentType='application/json'
        )
    
    def log_training_summary(self, total_epochs: int, best_metrics: Dict[str, float]):
        """
        Log final training summary
        """
        total_cost = self.log_cost_metrics()
        
        summary = {
            'run_id': self.config['run_id'],
            'total_epochs': total_epochs,
            'best_metrics': best_metrics,
            'total_training_time': str(datetime.now() - self.start_time),
            'total_cost': f"${total_cost:.2f}",
            'instance_type': self.instance_type,
            'completed_at': datetime.now().isoformat()
        }
        
        # Log to wandb
        wandb.log(summary)
        
        # Save summary to S3
        summary_key = f"training-summaries/ai-detection/{self.config['run_id']}_summary.json"
        self.s3.put_object(
            Bucket='ml-model-artifacts',
            Key=summary_key,
            Body=json.dumps(summary, indent=2),
            ContentType='application/json'
        )
        
        print("\n" + "="*50)
        print("TRAINING SUMMARY")
        print("="*50)
        for key, value in summary.items():
            print(f"{key}: {value}")
        print("="*50)