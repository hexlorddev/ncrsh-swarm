"""
Memory Profiling for ncrsh-Swarm
===============================

Advanced memory usage analysis and optimization recommendations.
"""

import asyncio
import psutil
import tracemalloc
import gc
import time
from typing import Dict, List, Any, Optional
import json
import matplotlib.pyplot as plt

from ..ncrsh_swarm.core.swarm_node import SwarmNode, SwarmNodeConfig
from ..ncrsh_swarm.models.transformer import TransformerConfig


class SwarmMemoryProfiler:
    """Memory profiling and optimization for swarm networks"""
    
    def __init__(self, output_dir: str = "./memory_profiles"):
        self.output_dir = output_dir
        self.profiles: List[Dict[str, Any]] = []
        
    async def profile_memory_usage(self, configurations: List[str]) -> Dict[str, Any]:
        """Profile memory usage across different configurations"""
        print("🧮 Profiling memory usage across configurations...")
        
        memory_results = {
            'test_type': 'memory_profiling',
            'configurations': {},
            'comparison': {},
            'recommendations': []
        }
        
        for config_name in configurations:
            print(f"  📊 Profiling {config_name} configuration...")
            
            profile_result = await self._profile_configuration(config_name)
            memory_results['configurations'][config_name] = profile_result
            
        # Generate comparison
        memory_results['comparison'] = self._compare_memory_profiles(
            memory_results['configurations']
        )
        
        # Generate recommendations
        memory_results['recommendations'] = self._generate_memory_recommendations(
            memory_results['configurations']
        )
        
        return memory_results
        
    async def _profile_configuration(self, config_name: str) -> Dict[str, Any]:
        """Profile memory usage for a specific configuration"""
        # Start memory tracing
        tracemalloc.start()
        
        # Get baseline memory
        process = psutil.Process()
        baseline_memory = process.memory_info().rss
        
        profile_data = {
            'config_name': config_name,
            'baseline_memory': baseline_memory,
            'phases': {},
            'peak_memory': baseline_memory,
            'memory_efficiency': 0.0
        }
        
        try:
            # Phase 1: Node creation
            phase1_start = process.memory_info().rss
            
            config = self._get_config_by_name(config_name)
            node = SwarmNode(config)
            
            phase1_end = process.memory_info().rss
            profile_data['phases']['node_creation'] = {
                'memory_before': phase1_start,
                'memory_after': phase1_end,
                'memory_increase': phase1_end - phase1_start
            }
            
            # Phase 2: Node startup
            phase2_start = process.memory_info().rss
            await node.start()
            phase2_end = process.memory_info().rss
            
            profile_data['phases']['node_startup'] = {
                'memory_before': phase2_start,
                'memory_after': phase2_end,
                'memory_increase': phase2_end - phase2_start
            }
            
            # Phase 3: Model operations
            phase3_start = process.memory_info().rss
            
            # Simulate training operations
            await self._simulate_memory_intensive_operations(node)
            
            phase3_end = process.memory_info().rss
            profile_data['phases']['model_operations'] = {
                'memory_before': phase3_start,
                'memory_after': phase3_end,
                'memory_increase': phase3_end - phase3_start
            }
            
            # Track peak memory
            profile_data['peak_memory'] = max(
                phase1_end, phase2_end, phase3_end
            )
            
            # Calculate efficiency
            model_params = sum(p.numel() for p in node.model.parameters())
            param_memory = model_params * 4  # 4 bytes per float32
            total_memory = profile_data['peak_memory'] - baseline_memory
            
            profile_data['memory_efficiency'] = param_memory / total_memory if total_memory > 0 else 0
            profile_data['model_parameters'] = model_params
            profile_data['parameter_memory'] = param_memory
            
            # Get detailed memory breakdown
            profile_data['memory_breakdown'] = await self._get_memory_breakdown(node)
            
            # Cleanup
            await node.stop()
            
        except Exception as e:
            profile_data['error'] = str(e)
            
        finally:
            # Stop tracing and get statistics
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            
            profile_data['tracemalloc'] = {
                'current': current,
                'peak': peak
            }
            
            # Force garbage collection
            gc.collect()
            
        return profile_data
        
    async def _simulate_memory_intensive_operations(self, node: SwarmNode) -> None:
        """Simulate memory-intensive operations"""
        # Create synthetic dataset
        dataset = self._create_large_dataset(size=100)
        
        try:
            # Short training to trigger memory allocation
            await node.train(dataset, epochs=1, batch_size=8)
        except Exception as e:
            print(f"Memory operation simulation failed: {e}")
            
    async def _get_memory_breakdown(self, node: SwarmNode) -> Dict[str, Any]:
        """Get detailed memory breakdown"""
        breakdown = {
            'model_parameters': 0,
            'model_buffers': 0,
            'optimizer_state': 0,
            'network_buffers': 0,
            'other': 0
        }
        
        try:
            # Model parameters
            breakdown['model_parameters'] = sum(
                p.numel() * p.element_size() for p in node.model.parameters()
            )
            
            # Model buffers
            breakdown['model_buffers'] = sum(
                b.numel() * b.element_size() for b in node.model.buffers()
            )
            
            # Trainer optimizer state (if available)
            if hasattr(node.trainer, 'optimizer'):
                breakdown['optimizer_state'] = breakdown['model_parameters'] * 2  # Estimate
                
        except Exception as e:
            breakdown['error'] = str(e)
            
        return breakdown
        
    def _get_config_by_name(self, config_name: str) -> SwarmNodeConfig:
        """Get configuration by preset name"""
        configs = {
            'tiny': SwarmNodeConfig(
                model_config=TransformerConfig(
                    hidden_size=64, num_layers=2, num_heads=2
                )
            ),
            'small': SwarmNodeConfig(
                model_config=TransformerConfig(
                    hidden_size=256, num_layers=4, num_heads=8
                )
            ),
            'medium': SwarmNodeConfig(
                model_config=TransformerConfig(
                    hidden_size=512, num_layers=8, num_heads=8
                )
            ),
            'large': SwarmNodeConfig(
                model_config=TransformerConfig(
                    hidden_size=768, num_layers=12, num_heads=12
                )
            )
        }
        
        return configs.get(config_name, configs['small'])
        
    def _create_large_dataset(self, size: int = 1000):
        """Create a dataset for memory testing"""
        import torch
        
        vocab_size = 1000
        seq_len = 128
        
        input_data = torch.randint(0, vocab_size, (size, seq_len))
        target_data = torch.roll(input_data, shifts=-1, dims=1)
        
        return torch.utils.data.TensorDataset(input_data, target_data)
        
    def _compare_memory_profiles(self, profiles: Dict[str, Dict]) -> Dict[str, Any]:
        """Compare memory profiles across configurations"""
        if not profiles:
            return {}
            
        comparison = {
            'memory_rankings': [],
            'efficiency_rankings': [],
            'memory_ratios': {},
            'efficiency_scores': {}
        }
        
        # Extract memory usage and efficiency
        memory_usage = {}
        efficiency_scores = {}
        
        for config_name, profile in profiles.items():
            if 'error' not in profile:
                total_memory = profile['peak_memory'] - profile['baseline_memory']
                memory_usage[config_name] = total_memory
                efficiency_scores[config_name] = profile.get('memory_efficiency', 0)
                
        # Rank by memory usage (lower is better)
        comparison['memory_rankings'] = sorted(
            memory_usage.keys(),
            key=lambda k: memory_usage[k]
        )
        
        # Rank by efficiency (higher is better)
        comparison['efficiency_rankings'] = sorted(
            efficiency_scores.keys(),
            key=lambda k: efficiency_scores[k],
            reverse=True
        )
        
        # Calculate ratios relative to smallest config
        if memory_usage:
            min_memory = min(memory_usage.values())
            comparison['memory_ratios'] = {
                k: v / min_memory for k, v in memory_usage.items()
            }
            
        comparison['efficiency_scores'] = efficiency_scores
        
        return comparison
        
    def _generate_memory_recommendations(self, profiles: Dict[str, Dict]) -> List[str]:
        """Generate memory optimization recommendations"""
        recommendations = []
        
        for config_name, profile in profiles.items():
            if 'error' in profile:
                continue
                
            efficiency = profile.get('memory_efficiency', 0)
            peak_memory = profile.get('peak_memory', 0)
            baseline = profile.get('baseline_memory', 0)
            total_memory = peak_memory - baseline
            
            # Low efficiency recommendations
            if efficiency < 0.3:
                recommendations.append(
                    f"{config_name}: Low memory efficiency ({efficiency:.1%}). "
                    "Consider enabling gradient checkpointing or reducing batch size."
                )
                
            # High memory usage recommendations
            if total_memory > 1024 * 1024 * 1024:  # > 1GB
                recommendations.append(
                    f"{config_name}: High memory usage ({total_memory / (1024**3):.1f}GB). "
                    "Consider model parallelism or mixed precision training."
                )
                
            # Check for memory leaks
            phases = profile.get('phases', {})
            startup_memory = phases.get('node_startup', {}).get('memory_increase', 0)
            operation_memory = phases.get('model_operations', {}).get('memory_increase', 0)
            
            if operation_memory > startup_memory * 2:
                recommendations.append(
                    f"{config_name}: Potential memory leak detected during operations. "
                    "Monitor garbage collection and tensor lifecycle."
                )
                
        if not recommendations:
            recommendations.append("All configurations show healthy memory usage patterns.")
            
        return recommendations
        
    def generate_memory_report(self, profiles: Dict[str, Any]) -> str:
        """Generate detailed memory analysis report"""
        report = f"""# Memory Profiling Report

## Overview
This report analyzes memory usage patterns across different ncrsh-Swarm configurations.

## Configuration Analysis

"""
        
        for config_name, profile in profiles.get('configurations', {}).items():
            if 'error' in profile:
                report += f"### {config_name.title()} Configuration\n**Error:** {profile['error']}\n\n"
                continue
                
            baseline = profile.get('baseline_memory', 0)
            peak = profile.get('peak_memory', 0)
            total = peak - baseline
            efficiency = profile.get('memory_efficiency', 0)
            
            report += f"""### {config_name.title()} Configuration

**Memory Usage:**
- Baseline: {baseline / (1024**2):.1f} MB
- Peak: {peak / (1024**2):.1f} MB  
- Total Increase: {total / (1024**2):.1f} MB
- Memory Efficiency: {efficiency:.1%}

**Phase Breakdown:**
"""
            
            phases = profile.get('phases', {})
            for phase_name, phase_data in phases.items():
                increase = phase_data.get('memory_increase', 0)
                report += f"- {phase_name.replace('_', ' ').title()}: +{increase / (1024**2):.1f} MB\n"
                
            report += "\n"
            
        # Add comparison section
        comparison = profiles.get('comparison', {})
        if comparison:
            report += "## Memory Comparison\n\n"
            
            rankings = comparison.get('memory_rankings', [])
            if rankings:
                report += "**Memory Usage Rankings (Best to Worst):**\n"
                for i, config in enumerate(rankings, 1):
                    ratio = comparison.get('memory_ratios', {}).get(config, 1)
                    report += f"{i}. {config.title()} ({ratio:.1f}x baseline)\n"
                    
                report += "\n"
                
        # Add recommendations
        recommendations = profiles.get('recommendations', [])
        if recommendations:
            report += "## Recommendations\n\n"
            for rec in recommendations:
                report += f"- {rec}\n"
                
        return report
        
    def plot_memory_usage(self, profiles: Dict[str, Any], save_path: str = None) -> None:
        """Generate memory usage visualization"""
        try:
            import matplotlib.pyplot as plt
            
            configs = list(profiles.get('configurations', {}).keys())
            memory_data = []
            
            for config in configs:
                profile = profiles['configurations'][config]
                if 'error' not in profile:
                    baseline = profile.get('baseline_memory', 0)
                    peak = profile.get('peak_memory', 0)
                    memory_data.append((peak - baseline) / (1024**2))  # MB
                else:
                    memory_data.append(0)
                    
            plt.figure(figsize=(10, 6))
            bars = plt.bar(configs, memory_data, color='skyblue', alpha=0.7)
            
            plt.title('Memory Usage by Configuration')
            plt.xlabel('Configuration')
            plt.ylabel('Memory Usage (MB)')
            plt.xticks(rotation=45)
            
            # Add value labels on bars
            for bar, value in zip(bars, memory_data):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                        f'{value:.1f}MB', ha='center', va='bottom')
                        
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"Memory usage plot saved to: {save_path}")
            else:
                plt.show()
                
        except ImportError:
            print("Matplotlib not available for plotting")
        except Exception as e:
            print(f"Failed to generate plot: {e}")


# Example usage
async def main():
    """Example memory profiling execution"""
    profiler = SwarmMemoryProfiler()
    
    print("🧮 Starting Memory Profiling")
    
    # Profile different configurations
    results = await profiler.profile_memory_usage(['small', 'medium'])
    
    # Generate report
    report = profiler.generate_memory_report(results)
    print("\n" + "="*60)
    print(report)
    
    # Generate visualization
    profiler.plot_memory_usage(results)


if __name__ == "__main__":
    asyncio.run(main())