import sagemaker
from sagemaker.pytorch import PyTorch
from sagemaker.inputs import TrainingInput
import boto3
from datetime import datetime

def create_training_job():
    """
    Set up distributed training on AWS infrastructure
    """
    role = sagemaker.get_execution_role()
    sess = sagemaker.Session()
    
    # Configure training with cost optimization
    estimator = PyTorch(
        entry_point='train.py',
        source_dir='src',
        role=role,
        instance_count=1,
        instance_type='ml.p3.2xlarge',  # V100 GPU instance
        framework_version='2.0',
        py_version='py310',
        
        # Use spot instances for 70% cost savings
        use_spot_instances=True,
        max_wait=7200,  # 2 hour max wait time
        
        hyperparameters={
            'epochs': 20,
            'batch_size': 32,
            'model': 'clip-vit-large',
            'learning_rate': 1e-4
        },
        
        # Metrics for automatic model tuning
        metric_definitions=[
            {'Name': 'validation:accuracy', 'Regex': 'Val Acc: ([0-9\\.]+)'},
            {'Name': 'train:loss', 'Regex': 'Train Loss: ([0-9\\.]+)'}
        ]
    )
    
    # S3 data configuration
    inputs = {
        'train': TrainingInput(
            s3_data=f's3://ai-detection-data/train/',
            content_type='application/x-image',
            s3_data_type='S3Prefix',
            input_mode='FastFile'
        ),
        'validation': TrainingInput(
            s3_data=f's3://ai-detection-data/val/',
            content_type='application/x-image',
            s3_data_type='S3Prefix',
            input_mode='FastFile'
        )
    }
    
    # Create a unique job name
    job_name = f"ai-detection-{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}"
    
    # Start training
    estimator.fit(inputs, job_name=job_name, wait=True)
    
    # Deploy model endpoint (optional)
    # predictor = estimator.deploy(
    #     initial_instance_count=1,
    #     instance_type='ml.m5.xlarge'
    # )
    
    return estimator

def hyperparameter_tuning():
    """
    Optional: Set up hyperparameter tuning job
    """
    from sagemaker.tuner import HyperparameterTuner, IntegerParameter, ContinuousParameter
    
    hyperparameter_ranges = {
        'learning_rate': ContinuousParameter(1e-5, 1e-3),
        'batch_size': IntegerParameter(16, 64),
        'dropout': ContinuousParameter(0.1, 0.5)
    }
    
    objective_metric_name = 'validation:accuracy'
    objective_type = 'Maximize'
    
    tuner = HyperparameterTuner(
        estimator,
        objective_metric_name,
        hyperparameter_ranges,
        max_jobs=10,
        max_parallel_jobs=2,
        objective_type=objective_type
    )
    
    return tuner

if __name__ == "__main__":
    print("Starting SageMaker training job...")
    estimator = create_training_job()
    print("Training complete!")
    print(f"Model artifacts saved to: {estimator.model_data}")