import boto3
from datetime import datetime, timedelta
from typing import Dict, List

class AWSCostOptimizer:
    """
    Optimize ML workload costs on AWS
    """
    def __init__(self):
        self.pricing = boto3.client('pricing', region_name='us-east-1')
        self.ec2 = boto3.client('ec2')
        
    def _get_spot_price(self, instance_type: str, region: str = 'us-west-2') -> float:
        """
        Get current spot price for instance type
        """
        response = self.ec2.describe_spot_price_history(
            InstanceTypes=[instance_type],
            MaxResults=1,
            ProductDescriptions=['Linux/UNIX'],
            StartTime=datetime.now() - timedelta(hours=1)
        )
        
        if response['SpotPriceHistory']:
            return float(response['SpotPriceHistory'][0]['SpotPrice'])
        return 0.0
    
    def get_cheapest_gpu_instance(self, region='us-west-2') -> tuple:
        """
        Find the most cost-effective GPU instance
        """
        # Common GPU instances for ML workloads
        instance_types = ['p3.2xlarge', 'g4dn.xlarge', 'p3.8xlarge', 'g5.xlarge']
        
        cheapest = None
        min_price = float('inf')
        
        for instance in instance_types:
            price = self._get_spot_price(instance, region)
            if price > 0 and price < min_price:
                min_price = price
                cheapest = instance
                
        return cheapest, min_price
    
    def estimate_training_cost(self, instance_type: str, hours: float) -> Dict[str, float]:
        """
        Calculate total cost including compute, storage, and transfer
        """
        spot_price = self._get_spot_price(instance_type)
        
        # Storage costs
        storage_gb = 500
        storage_cost = 0.10 * storage_gb / 730 * hours  # $0.10/GB/month for gp3
        
        # Data transfer costs (assuming 20GB)
        data_transfer_gb = 20
        data_transfer = 0.09 * data_transfer_gb  # $0.09/GB
        
        # Compute cost
        compute_cost = spot_price * hours
        
        # On-demand prices for comparison
        on_demand_prices = {
            'p3.2xlarge': 3.06,
            'g4dn.xlarge': 0.736,
            'p3.8xlarge': 12.24,
            'g5.xlarge': 1.006
        }
        
        on_demand_cost = on_demand_prices.get(instance_type, 3.06) * hours
        savings_percent = ((on_demand_cost - compute_cost) / on_demand_cost) * 100
        
        return {
            'compute': compute_cost,
            'storage': storage_cost,
            'transfer': data_transfer,
            'total': compute_cost + storage_cost + data_transfer,
            'on_demand_cost': on_demand_cost,
            'savings_vs_ondemand': f'{savings_percent:.1f}%',
            'hourly_rate': spot_price
        }
    
    def optimize_instance_selection(self, target_memory_gb: int = 16) -> List[Dict]:
        """
        Recommend instances based on requirements
        """
        # Instance specifications
        instances = [
            {'type': 'p3.2xlarge', 'gpu': 1, 'gpu_memory': 16, 'cpu': 8, 'memory': 61},
            {'type': 'p3.8xlarge', 'gpu': 4, 'gpu_memory': 64, 'cpu': 32, 'memory': 244},
            {'type': 'g4dn.xlarge', 'gpu': 1, 'gpu_memory': 16, 'cpu': 4, 'memory': 16},
            {'type': 'g5.xlarge', 'gpu': 1, 'gpu_memory': 24, 'cpu': 4, 'memory': 16}
        ]
        
        recommendations = []
        
        for instance in instances:
            if instance['gpu_memory'] >= target_memory_gb:
                price = self._get_spot_price(instance['type'])
                if price > 0:
                    cost_per_gpu_hour = price / instance['gpu']
                    recommendations.append({
                        'instance_type': instance['type'],
                        'gpus': instance['gpu'],
                        'gpu_memory': instance['gpu_memory'],
                        'hourly_cost': price,
                        'cost_per_gpu_hour': cost_per_gpu_hour,
                        'specs': instance
                    })
        
        # Sort by cost per GPU hour
        recommendations.sort(key=lambda x: x['cost_per_gpu_hour'])
        return recommendations

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Estimate AWS training costs')
    parser.add_argument('--hours', type=float, default=3, help='Training hours')
    parser.add_argument('--instance', type=str, default='p3.2xlarge', help='Instance type')
    
    args = parser.parse_args()
    
    optimizer = AWSCostOptimizer()
    
    # Get cost estimate
    costs = optimizer.estimate_training_cost(args.instance, args.hours)
    
    print("\n" + "="*50)
    print(f"Cost Estimate for {args.instance} ({args.hours} hours)")
    print("="*50)
    for key, value in costs.items():
        if isinstance(value, float):
            print(f"{key}: ${value:.2f}")
        else:
            print(f"{key}: {value}")
    
    # Get cheapest option
    cheapest, price = optimizer.get_cheapest_gpu_instance()
    print(f"\nCheapest GPU instance: {cheapest} at ${price:.3f}/hour")
    
    # Get recommendations
    print("\nRecommended instances for AI training:")
    recommendations = optimizer.optimize_instance_selection()
    for rec in recommendations[:3]:
        print(f"- {rec['instance_type']}: ${rec['hourly_cost']:.3f}/hr ({rec['gpus']} GPU, {rec['gpu_memory']}GB VRAM)")