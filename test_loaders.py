import os
import time
import statistics
import psutil
import gc
from src.data_modules import LMDBRIRDataset, DiskRIRDataset
from typing import List, Dict, Any
import numpy as np


class DatasetBenchmark:
    def __init__(self, dataset_class, dataset_kwargs: Dict[str, Any]):
        self.dataset_class = dataset_class
        self.dataset_kwargs = dataset_kwargs
        self.results = {}
    
    def benchmark_loading(self, num_samples: int = 100, warmup_samples: int = 10) -> Dict[str, Any]:
        """Benchmark dataset loading performance"""
        print(f"Benchmarking {self.dataset_class.__name__} loading...")
        
        # Create dataset
        dataset = self.dataset_class(**self.dataset_kwargs)
        data_loader = iter(dataset)
        
        # Warmup phase
        print(f"Warming up with {warmup_samples} samples...")
        for _ in range(warmup_samples):
            try:
                next(data_loader)
            except StopIteration:
                data_loader = iter(dataset)  # Reset if exhausted
                next(data_loader)
        
        # Benchmark phase
        print(f"Benchmarking {num_samples} samples...")
        load_times = []
        memory_usage = []
        
        # Get initial memory
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        for i in range(num_samples):
            # Memory before loading
            mem_before = process.memory_info().rss / 1024 / 1024
            
            # Time the loading
            start_time = time.perf_counter()
            try:
                rir = next(data_loader)
                end_time = time.perf_counter()
                
                # Memory after loading
                mem_after = process.memory_info().rss / 1024 / 1024
                
                load_times.append(end_time - start_time)
                memory_usage.append(mem_after)
                
                if (i + 1) % 20 == 0:
                    print(f"Processed {i + 1}/{num_samples} samples")
                    
            except StopIteration:
                print(f"Dataset exhausted after {i} samples")
                break
        
        # Calculate statistics
        final_memory = process.memory_info().rss / 1024 / 1024
        
        results = {
            'dataset_class': self.dataset_class.__name__,
            'num_samples': len(load_times),
            'total_time': sum(load_times),
            'avg_time_per_sample': statistics.mean(load_times),
            'median_time_per_sample': statistics.median(load_times),
            'std_time_per_sample': statistics.stdev(load_times) if len(load_times) > 1 else 0,
            'min_time_per_sample': min(load_times),
            'max_time_per_sample': max(load_times),
            'samples_per_second': len(load_times) / sum(load_times),
            'initial_memory_mb': initial_memory,
            'final_memory_mb': final_memory,
            'peak_memory_mb': max(memory_usage),
            'avg_memory_mb': statistics.mean(memory_usage),
            'memory_increase_mb': final_memory - initial_memory,
            'load_times': load_times,
            'memory_usage': memory_usage
        }
        
        return results
    
    def benchmark_repeated_access(self, num_iterations: int = 5, samples_per_iteration: int = 50) -> Dict[str, Any]:
        """Benchmark repeated access patterns (useful for cache testing)"""
        print(f"Benchmarking repeated access patterns...")
        
        iteration_times = []
        
        for iteration in range(num_iterations):
            dataset = self.dataset_class(**self.dataset_kwargs)
            data_loader = iter(dataset)
            
            start_time = time.perf_counter()
            for i in range(samples_per_iteration):
                try:
                    next(data_loader)
                except StopIteration:
                    break
            end_time = time.perf_counter()
            
            iteration_time = end_time - start_time
            iteration_times.append(iteration_time)
            print(f"Iteration {iteration + 1}/{num_iterations}: {iteration_time:.3f}s")
        
        return {
            'num_iterations': num_iterations,
            'samples_per_iteration': samples_per_iteration,
            'iteration_times': iteration_times,
            'avg_iteration_time': statistics.mean(iteration_times),
            'std_iteration_time': statistics.stdev(iteration_times) if len(iteration_times) > 1 else 0,
            'first_iteration_time': iteration_times[0],
            'subsequent_avg_time': statistics.mean(iteration_times[1:]) if len(iteration_times) > 1 else 0
        }
    
    def print_results(self, results: Dict[str, Any]):
        """Print benchmark results in a formatted way"""
        print(f"\n{'='*60}")
        print(f"BENCHMARK RESULTS: {results['dataset_class']}")
        print(f"{'='*60}")
        print(f"Samples processed: {results['num_samples']}")
        print(f"Total time: {results['total_time']:.3f}s")
        print(f"Average time per sample: {results['avg_time_per_sample']*1000:.2f}ms")
        print(f"Median time per sample: {results['median_time_per_sample']*1000:.2f}ms")
        print(f"Std deviation: {results['std_time_per_sample']*1000:.2f}ms")
        print(f"Min time per sample: {results['min_time_per_sample']*1000:.2f}ms")
        print(f"Max time per sample: {results['max_time_per_sample']*1000:.2f}ms")
        print(f"Throughput: {results['samples_per_second']:.2f} samples/sec")
        print(f"\nMemory Usage:")
        print(f"Initial: {results['initial_memory_mb']:.1f}MB")
        print(f"Final: {results['final_memory_mb']:.1f}MB")
        print(f"Peak: {results['peak_memory_mb']:.1f}MB")
        print(f"Average: {results['avg_memory_mb']:.1f}MB")
        print(f"Increase: {results['memory_increase_mb']:.1f}MB")
        print(f"{'='*60}")
    
    def plot_timing_distribution(self, results: Dict[str, Any]):
        """Plot timing distribution (requires matplotlib)"""
        try:
            import matplotlib.pyplot as plt
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            
            # Timing histogram
            ax1.hist(np.array(results['load_times']) * 1000, bins=30, alpha=0.7, edgecolor='black')
            ax1.set_xlabel('Load Time (ms)')
            ax1.set_ylabel('Frequency')
            ax1.set_title(f'{results["dataset_class"]} - Load Time Distribution')
            ax1.axvline(results['avg_time_per_sample'] * 1000, color='red', linestyle='--', label='Mean')
            ax1.axvline(results['median_time_per_sample'] * 1000, color='green', linestyle='--', label='Median')
            ax1.legend()
            
            # Memory usage over time
            ax2.plot(results['memory_usage'], alpha=0.7)
            ax2.set_xlabel('Sample Index')
            ax2.set_ylabel('Memory Usage (MB)')
            ax2.set_title(f'{results["dataset_class"]} - Memory Usage Over Time')
            ax2.axhline(results['avg_memory_mb'], color='red', linestyle='--', label='Average')
            ax2.legend()
            
            plt.tight_layout()
            plt.show()
            
        except ImportError:
            print("Matplotlib not available. Skipping plots.")


def compare_datasets(dataset_configs: List[Dict[str, Any]], num_samples: int = 100):
    """Compare multiple dataset configurations"""
    results = []
    
    for config in dataset_configs:
        dataset_class = config.pop('dataset_class')
        benchmark = DatasetBenchmark(dataset_class, config)
        result = benchmark.benchmark_loading(num_samples)
        results.append(result)
        benchmark.print_results(result)
        
        # Clean up
        gc.collect()
        time.sleep(1)  # Brief pause between benchmarks
    
    # Comparison summary
    print(f"\n{'='*80}")
    print("COMPARISON SUMMARY")
    print(f"{'='*80}")
    print(f"{'Dataset':<20} {'Avg Time (ms)':<15} {'Throughput (sps)':<18} {'Memory (MB)':<12}")
    print(f"{'-'*80}")
    
    for result in results:
        print(f"{result['dataset_class']:<20} "
              f"{result['avg_time_per_sample']*1000:<15.2f} "
              f"{result['samples_per_second']:<18.2f} "
              f"{result['memory_increase_mb']:<12.1f}")
    
    return results


if __name__ == "__main__":
    # Configuration
    rir_data_dir = "/projects/0/prjs1261/BinauralRIRsUnpack/BinauralRIRs"
    rir_dir = "/projects/0/prjs1261/BinauralRIRsUnpack/Scenes"
    
    # Define dataset configurations to compare
    dataset_configs = [
        {
            'dataset_class': DiskRIRDataset,
            'rir_data_path': rir_data_dir,
            'rir_path': os.path.join(rir_dir, "train"),
            'sr': 32000,
            'ambisonic': False,
            'with_noise': True,
            'readonly': True,
            'cache_size': 1000
        },
        # Add LMDB configuration if you want to compare
        # {
        #     'dataset_class': LMDBRIRDataset,
        #     'rir_data_path': rir_data_dir,
        #     'rir_path': os.path.join(rir_dir, "train"),
        #     'sr': 32000,
        #     'ambisonic': False,
        #     'with_noise': True,
        #     'readonly': True
        # }
    ]
    
    # Run benchmark
    print("Starting comprehensive dataset benchmark...")
    results = compare_datasets(dataset_configs, num_samples=100)
    
    # Optional: Test repeated access patterns
    print("\nTesting repeated access patterns...")
    benchmark = DatasetBenchmark(DiskRIRDataset, dataset_configs[0])
    repeated_results = benchmark.benchmark_repeated_access(num_iterations=3, samples_per_iteration=30)
    
    print(f"\nRepeated Access Results:")
    print(f"First iteration: {repeated_results['first_iteration_time']:.3f}s")
    print(f"Subsequent average: {repeated_results['subsequent_avg_time']:.3f}s")
    print(f"Speedup after first: {repeated_results['first_iteration_time'] / repeated_results['subsequent_avg_time']:.2f}x")
