"""
GPU utilities for efficient training
"""
import torch
import numpy as np
from typing import Optional, Dict, Any, List
import gc
import psutil
import os

class GPUManager:
    """Manage GPU resources for efficient training"""
    
    def __init__(self, device_id: int = 0):
        self.device_id = device_id
        self.device = self._setup_device()
        self.memory_stats = {}
    
    def _setup_device(self) -> torch.device:
        """Setup GPU device"""
        if torch.cuda.is_available():
            device = torch.device(f'cuda:{self.device_id}')
            torch.cuda.set_device(device)
            
            # Print GPU info
            print(f"Using GPU: {torch.cuda.get_device_name(device)}")
            print(f"GPU Memory: {torch.cuda.get_device_properties(device).total_memory / 1e9:.2f} GB")
            
            # Set optimization flags
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            
            # Enable TF32 for Ampere GPUs
            if torch.cuda.get_device_capability()[0] >= 8:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
                print("TF32 enabled for faster training")
            
        else:
            device = torch.device('cpu')
            print("Using CPU")
        
        return device
    
    def get_memory_stats(self) -> Dict[str, float]:
        """Get GPU memory statistics"""
        if self.device.type == 'cuda':
            torch.cuda.synchronize()
            allocated = torch.cuda.memory_allocated(self.device) / 1e9
            reserved = torch.cuda.memory_reserved(self.device) / 1e9
            max_allocated = torch.cuda.max_memory_allocated(self.device) / 1e9
            
            self.memory_stats = {
                'allocated_gb': allocated,
                'reserved_gb': reserved,
                'max_allocated_gb': max_allocated,
                'utilization': (allocated / reserved) * 100 if reserved > 0 else 0
            }
        
        return self.memory_stats
    
    def clear_cache(self):
        """Clear GPU cache"""
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            gc.collect()
    
    def optimize_batch_size(self, model: torch.nn.Module, input_shape: tuple,
                          target_memory_usage: float = 0.8) -> int:
        """Find optimal batch size for given memory constraints"""
        
        if self.device.type != 'cuda':
            return 32  # Default for CPU
        
        # Get total GPU memory
        total_memory = torch.cuda.get_device_properties(self.device).total_memory / 1e9
        target_memory = total_memory * target_memory_usage
        
        # Try different batch sizes
        batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
        optimal_batch_size = 32
        
        model.train()
        
        for batch_size in batch_sizes:
            try:
                # Create dummy batch
                dummy_input = torch.randn(batch_size, *input_shape).to(self.device)
                
                # Forward pass to allocate memory
                output = model(dummy_input)
                
                # Backward pass
                if isinstance(output, torch.Tensor):
                    loss = output.sum()
                    loss.backward()
                
                # Get memory usage
                memory_stats = self.get_memory_stats()
                memory_used = memory_stats['allocated_gb']
                
                # Clear
                self.clear_cache()
                
                if memory_used < target_memory:
                    optimal_batch_size = batch_size
                else:
                    break
                    
            except RuntimeError as e:
                if 'out of memory' in str(e):
                    self.clear_cache()
                    break
                else:
                    raise e
        
        print(f"Optimal batch size: {optimal_batch_size}")
        return optimal_batch_size
    
    def mixed_precision_training(self, model: torch.nn.Module, optimizer: torch.optim.Optimizer,
                               scaler: Optional[torch.cuda.amp.GradScaler] = None):
        """Setup mixed precision training"""
        
        if self.device.type != 'cuda':
            return model, optimizer, None
        
        # Create gradient scaler for mixed precision
        if scaler is None:
            scaler = torch.cuda.amp.GradScaler()
        
        return model, optimizer, scaler
    
    def data_parallel(self, model: torch.nn.Module) -> torch.nn.Module:
        """Enable data parallel training across multiple GPUs"""
        
        if torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs for data parallel training")
            model = torch.nn.DataParallel(model)
        
        return model
    
    def gradient_accumulation(self, accumulation_steps: int, batch_size: int,
                            target_batch_size: int) -> int:
        """Calculate gradient accumulation steps"""
        
        if batch_size >= target_batch_size:
            return 1
        
        accumulation_steps = max(1, target_batch_size // batch_size)
        print(f"Using gradient accumulation with {accumulation_steps} steps")
        print(f"Effective batch size: {batch_size * accumulation_steps}")
        
        return accumulation_steps
    
    def profile_model(self, model: torch.nn.Module, input_shape: tuple,
                     iterations: int = 100):
        """Profile model performance"""
        
        if self.device.type != 'cuda':
            print("Profiling only available for GPU")
            return
        
        model.eval()
        model.to(self.device)
        
        # Warmup
        dummy_input = torch.randn(1, *input_shape).to(self.device)
        for _ in range(10):
            _ = model(dummy_input)
        
        # Profile
        torch.cuda.synchronize()
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        start_event.record()
        for _ in range(iterations):
            _ = model(dummy_input)
        end_event.record()
        
        torch.cuda.synchronize()
        elapsed_time = start_event.elapsed_time(end_event) / 1000  # Convert to seconds
        
        throughput = iterations / elapsed_time
        latency = elapsed_time / iterations * 1000  # Convert to milliseconds
        
        print(f"\nModel Profile:")
        print(f"  Throughput: {throughput:.2f} samples/sec")
        print(f"  Latency: {latency:.2f} ms")
        
        # Memory profile
        memory_stats = self.get_memory_stats()
        print(f"  Memory allocated: {memory_stats['allocated_gb']:.2f} GB")
        
        self.clear_cache()
        
        return {
            'throughput': throughput,
            'latency': latency,
            'memory_allocated': memory_stats['allocated_gb']
        }
    
    def check_system_resources(self) -> Dict[str, Any]:
        """Check system resources"""
        
        resources = {}
        
        # CPU
        resources['cpu_count'] = psutil.cpu_count()
        resources['cpu_percent'] = psutil.cpu_percent(interval=1)
        
        # Memory
        memory = psutil.virtual_memory()
        resources['memory_total_gb'] = memory.total / 1e9
        resources['memory_available_gb'] = memory.available / 1e9
        resources['memory_percent_used'] = memory.percent
        
        # Disk
        disk = psutil.disk_usage('/')
        resources['disk_total_gb'] = disk.total / 1e9
        resources['disk_free_gb'] = disk.free / 1e9
        resources['disk_percent_used'] = disk.percent
        
        # GPU
        if self.device.type == 'cuda':
            resources['gpu_count'] = torch.cuda.device_count()
            resources['gpu_name'] = torch.cuda.get_device_name(self.device)
            resources['gpu_memory_total_gb'] = torch.cuda.get_device_properties(self.device).total_memory / 1e9
            
            # Current GPU memory
            memory_stats = self.get_memory_stats()
            resources.update(memory_stats)
        
        return resources
    
    def print_resource_summary(self):
        """Print system resource summary"""
        
        resources = self.check_system_resources()
        
        print("\n" + "="*50)
        print("SYSTEM RESOURCE SUMMARY")
        print("="*50)
        
        print(f"\nCPU:")
        print(f"  Cores: {resources['cpu_count']}")
        print(f"  Usage: {resources['cpu_percent']:.1f}%")
        
        print(f"\nMemory:")
        print(f"  Total: {resources['memory_total_gb']:.2f} GB")
        print(f"  Available: {resources['memory_available_gb']:.2f} GB")
        print(f"  Used: {resources['memory_percent_used']:.1f}%")
        
        print(f"\nDisk:")
        print(f"  Total: {resources['disk_total_gb']:.2f} GB")
        print(f"  Free: {resources['disk_free_gb']:.2f} GB")
        print(f"  Used: {resources['disk_percent_used']:.1f}%")
        
        if self.device.type == 'cuda':
            print(f"\nGPU:")
            print(f"  Name: {resources['gpu_name']}")
            print(f"  Memory: {resources['gpu_memory_total_gb']:.2f} GB")
            print(f"  Allocated: {resources.get('allocated_gb', 0):.2f} GB")
            print(f"  Utilization: {resources.get('utilization', 0):.1f}%")
        
        print("="*50)