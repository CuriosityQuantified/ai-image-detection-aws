import boto3
from typing import Dict
import json

class MLInfrastructure:
    """
    EC2-based ML infrastructure with enterprise-grade setup
    """
    def __init__(self):
        self.ec2 = boto3.client('ec2')
        self.s3 = boto3.client('s3')
        
    def launch_training_instance(self) -> str:
        """
        Launch optimized spot instance for deep learning
        """
        user_data = """#!/bin/bash
        # Configure ML environment
        sudo apt-get update
        sudo apt-get install -y python3-pip nvidia-driver-470
        
        # Install ML stack
        pip3 install torch torchvision transformers
        pip3 install boto3 wandb scikit-learn
        
        # Mount high-performance storage for datasets
        sudo mkfs -t xfs /dev/nvme1n1
        sudo mount /dev/nvme1n1 /data
        
        # Retrieve training code
        aws s3 cp s3://ml-code-bucket/training/ /home/ubuntu/training/ --recursive
        
        # Execute training with monitoring
        cd /home/ubuntu/training
        python3 train_with_monitoring.py
        """
        
        response = self.ec2.request_spot_instances(
            SpotPrice='0.90',  # Max price for spot instance
            InstanceCount=1,
            Type='one-time',
            LaunchSpecification={
                'ImageId': 'ami-0123456789abcdef0',  # AWS Deep Learning AMI
                'InstanceType': 'p3.2xlarge',
                'KeyName': 'ml-training-key',
                'SecurityGroups': ['ml-security-group'],
                'BlockDeviceMappings': [
                    {
                        'DeviceName': '/dev/sda1',
                        'Ebs': {
                            'VolumeSize': 100,
                            'VolumeType': 'gp3',  # Latest generation SSD
                            'DeleteOnTermination': True
                        }
                    },
                    {
                        'DeviceName': '/dev/sdb',
                        'Ebs': {
                            'VolumeSize': 500,  # Data volume
                            'VolumeType': 'st1',  # Throughput optimized
                            'DeleteOnTermination': True
                        }
                    }
                ],
                'IamInstanceProfile': {
                    'Name': 'ml-instance-role'
                },
                'UserData': user_data,
                'TagSpecifications': [
                    {
                        'ResourceType': 'instance',
                        'Tags': [
                            {'Key': 'Project', 'Value': 'AI-Detection'},
                            {'Key': 'Environment', 'Value': 'Training'},
                            {'Key': 'CostCenter', 'Value': 'ML-Research'}
                        ]
                    }
                ]
            }
        )
        return response['SpotInstanceRequests'][0]['SpotInstanceRequestId']
    
    def create_security_group(self):
        """
        Create security group for ML instances
        """
        response = self.ec2.create_security_group(
            GroupName='ml-security-group',
            Description='Security group for ML training instances'
        )
        
        # Allow SSH and TensorBoard
        self.ec2.authorize_security_group_ingress(
            GroupId=response['GroupId'],
            IpPermissions=[
                {
                    'IpProtocol': 'tcp',
                    'FromPort': 22,
                    'ToPort': 22,
                    'IpRanges': [{'CidrIp': '0.0.0.0/0'}]
                },
                {
                    'IpProtocol': 'tcp',
                    'FromPort': 6006,
                    'ToPort': 6006,
                    'IpRanges': [{'CidrIp': '0.0.0.0/0'}]
                }
            ]
        )
        
        return response['GroupId']

if __name__ == "__main__":
    infra = MLInfrastructure()
    instance_id = infra.launch_training_instance()
    print(f"Launched spot instance request: {instance_id}")