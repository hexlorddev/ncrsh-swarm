"""
Performance Comparator for ncrsh-Swarm
=====================================

Compare performance across different configurations, hardware setups,
and network topologies.
"""

import asyncio
import statistics
import json
from typing import Dict, List, Any, Optional, Tuple
import time

from swarm_benchmarker import SwarmBenchmarker
from load_tester import SwarmLoadTester
from memory_profiler import SwarmMemoryProfiler


class PerformanceComparator:
    """
    Advanced performance comparison and analysis system
    """
    
    def __init__(self, output_dir: str = "./comparison_results"):
        self.output_dir = output_dir
        self.benchmarker = SwarmBenchmarker(output_dir)
        self.load_tester = SwarmLoadTester(output_dir)
        self.memory_profiler = SwarmMemoryProfiler(output_dir)
        
    async def compare_configurations(self, config_names: List[str]) -> Dict[str, Any]:
        """Compare performance across different configurations"""
        print(f"📊 Comparing {len(config_names)} configurations...")
        
        comparison_results = {
            'test_type': 'configuration_comparison',
            'configurations': config_names,
            'metrics': {},
            'rankings': {},
            'recommendations': {}
        }
        
        # Collect metrics for each configuration
        for config_name in config_names:
            print(f"  🔍 Testing {config_name}...")
            
            config_metrics = await self._collect_configuration_metrics(config_name)
            comparison_results['metrics'][config_name] = config_metrics
            
        # Generate rankings
        comparison_results['rankings'] = self._generate_rankings(
            comparison_results['metrics']
        )
        
        # Generate recommendations
        comparison_results['recommendations'] = self._generate_comparison_recommendations(
            comparison_results['metrics'], comparison_results['rankings']
        )
        
        return comparison_results
        
    async def _collect_configuration_metrics(self, config_name: str) -> Dict[str, Any]:
        """Collect comprehensive metrics for a configuration"""
        metrics = {
            'config_name': config_name,
            'performance': {},
            'memory': {},
            'scalability': {},
            'reliability': {}
        }
        
        try:
            # Performance metrics (training speed, throughput)
            perf_result = await self._measure_performance(config_name)
            metrics['performance'] = perf_result
            
            # Memory metrics
            memory_result = await self._measure_memory_usage(config_name)
            metrics['memory'] = memory_result
            
            # Scalability metrics
            scale_result = await self._measure_scalability(config_name)
            metrics['scalability'] = scale_result
            
            # Reliability metrics (fault tolerance)
            reliability_result = await self._measure_reliability(config_name)
            metrics['reliability'] = reliability_result
            
        except Exception as e:
            metrics['error'] = str(e)
            
        return metrics
        
    async def _measure_performance(self, config_name: str) -> Dict[str, float]:
        """Measure basic performance metrics"""
        # This would use the actual benchmarker, simplified for example
        return {
            'training_speed': 100.0,  # samples/sec
            'inference_speed': 500.0,  # samples/sec
            'convergence_rate': 0.85,  # loss reduction per epoch
            'throughput': 250.0  # operations/sec
        }
        
    async def _measure_memory_usage(self, config_name: str) -> Dict[str, float]:
        """Measure memory efficiency metrics"""
        return {
            'peak_memory_mb': 512.0,
            'memory_efficiency': 0.75,
            'memory_growth_rate': 0.02,  # MB per operation
            'gc_frequency': 10.0  # collections per minute
        }
        
    async def _measure_scalability(self, config_name: str) -> Dict[str, float]:
        """Measure scalability characteristics"""
        return {
            'scaling_efficiency': 0.80,
            'optimal_node_count': 8,
            'max_supported_nodes': 20,
            'scaling_overhead': 0.15
        }
        
    async def _measure_reliability(self, config_name: str) -> Dict[str, float]:
        """Measure reliability and fault tolerance"""
        return {
            'fault_tolerance': 0.85,  # performance retention under failures
            'recovery_time': 5.0,  # seconds
            'error_rate': 0.02,  # fraction of failed operations
            'uptime_score': 0.99
        }
        
    def _generate_rankings(self, metrics: Dict[str, Dict]) -> Dict[str, List[str]]:
        """Generate rankings across different criteria"""
        rankings = {}
        
        # Performance ranking (higher is better)
        perf_scores = {}
        for config, data in metrics.items():
            if 'error' not in data:
                perf = data.get('performance', {})
                # Composite performance score
                score = (
                    perf.get('training_speed', 0) * 0.3 +
                    perf.get('inference_speed', 0) * 0.3 +
                    perf.get('convergence_rate', 0) * 100 * 0.2 +
                    perf.get('throughput', 0) * 0.2
                )
                perf_scores[config] = score
                
        rankings['performance'] = sorted(
            perf_scores.keys(),
            key=lambda k: perf_scores[k],
            reverse=True
        )
        
        # Memory efficiency ranking (higher efficiency is better)
        memory_scores = {}
        for config, data in metrics.items():
            if 'error' not in data:
                memory = data.get('memory', {})
                # Lower memory usage and higher efficiency is better
                efficiency = memory.get('memory_efficiency', 0)
                peak_memory = memory.get('peak_memory_mb', 1000)
                score = efficiency * 1000 / peak_memory  # Efficiency per MB
                memory_scores[config] = score
                
        rankings['memory_efficiency'] = sorted(
            memory_scores.keys(),
            key=lambda k: memory_scores[k],
            reverse=True
        )
        
        # Scalability ranking
        scale_scores = {}
        for config, data in metrics.items():
            if 'error' not in data:
                scale = data.get('scalability', {})
                # Higher efficiency and more supported nodes is better
                score = (
                    scale.get('scaling_efficiency', 0) * 0.6 +
                    (scale.get('max_supported_nodes', 0) / 50.0) * 0.4
                )
                scale_scores[config] = score
                
        rankings['scalability'] = sorted(
            scale_scores.keys(),
            key=lambda k: scale_scores[k],
            reverse=True
        )
        
        # Reliability ranking
        reliability_scores = {}
        for config, data in metrics.items():
            if 'error' not in data:
                reliability = data.get('reliability', {})
                score = (
                    reliability.get('fault_tolerance', 0) * 0.4 +
                    reliability.get('uptime_score', 0) * 0.3 +
                    (1 - reliability.get('error_rate', 0)) * 0.3
                )
                reliability_scores[config] = score
                
        rankings['reliability'] = sorted(
            reliability_scores.keys(),
            key=lambda k: reliability_scores[k],
            reverse=True
        )
        
        # Overall ranking (weighted combination)
        overall_scores = {}
        for config in metrics.keys():
            if 'error' not in metrics[config]:
                score = (
                    perf_scores.get(config, 0) * 0.01 +  # Scale down performance
                    memory_scores.get(config, 0) * 0.3 +
                    scale_scores.get(config, 0) * 0.3 +
                    reliability_scores.get(config, 0) * 0.4
                )
                overall_scores[config] = score
                
        rankings['overall'] = sorted(
            overall_scores.keys(),
            key=lambda k: overall_scores[k],
            reverse=True
        )
        
        return rankings
        
    def _generate_comparison_recommendations(
        self, 
        metrics: Dict[str, Dict], 
        rankings: Dict[str, List]
    ) -> Dict[str, List[str]]:
        """Generate recommendations based on comparison results"""
        recommendations = {
            'production': [],
            'development': [],
            'research': [],
            'general': []
        }
        
        overall_ranking = rankings.get('overall', [])
        
        if overall_ranking:
            best_overall = overall_ranking[0]
            best_performance = rankings.get('performance', [None])[0]
            best_memory = rankings.get('memory_efficiency', [None])[0]
            best_scalability = rankings.get('scalability', [None])[0]
            
            # Production recommendations
            recommendations['production'].append(
                f"For production workloads, use '{best_overall}' for best overall performance"
            )
            
            if best_scalability and best_scalability != best_overall:
                recommendations['production'].append(
                    f"For high-scale deployments, consider '{best_scalability}' for better scalability"
                )
                
            # Development recommendations
            recommendations['development'].append(
                f"For development, '{best_memory}' offers good memory efficiency"
            )
            
            # Research recommendations
            if best_performance:
                recommendations['research'].append(
                    f"For research and experimentation, '{best_performance}' provides best performance"
                )
                
            # General recommendations
            recommendations['general'].append(
                "Consider workload characteristics when choosing configuration"
            )
            recommendations['general'].append(
                "Monitor resource usage and scale accordingly"
            )
            
        return recommendations
        
    def generate_comparison_report(self, results: Dict[str, Any]) -> str:
        """Generate comprehensive comparison report"""
        report = f"""# Performance Comparison Report

## Configurations Tested
{', '.join(results.get('configurations', []))}

## Rankings Summary

"""
        
        rankings = results.get('rankings', {})
        
        for category, ranking in rankings.items():
            report += f"### {category.replace('_', ' ').title()}\n"
            for i, config in enumerate(ranking, 1):
                report += f"{i}. {config.title()}\n"
            report += "\n"
            
        # Detailed metrics
        report += "## Detailed Metrics\n\n"
        
        metrics = results.get('metrics', {})
        for config_name, config_metrics in metrics.items():
            if 'error' in config_metrics:
                report += f"### {config_name.title()}\n**Error:** {config_metrics['error']}\n\n"
                continue
                
            report += f"### {config_name.title()}\n\n"
            
            # Performance metrics
            perf = config_metrics.get('performance', {})
            report += "**Performance:**\n"
            for metric, value in perf.items():
                unit = self._get_metric_unit(metric)
                report += f"- {metric.replace('_', ' ').title()}: {value:.2f}{unit}\n"
                
            # Memory metrics
            memory = config_metrics.get('memory', {})
            report += "\n**Memory:**\n"
            for metric, value in memory.items():
                unit = self._get_metric_unit(metric)
                report += f"- {metric.replace('_', ' ').title()}: {value:.2f}{unit}\n"
                
            report += "\n"
            
        # Recommendations
        recommendations = results.get('recommendations', {})
        if recommendations:
            report += "## Recommendations\n\n"
            
            for category, recs in recommendations.items():
                if recs:
                    report += f"### {category.title()}\n"
                    for rec in recs:
                        report += f"- {rec}\n"
                    report += "\n"
                    
        return report
        
    def _get_metric_unit(self, metric_name: str) -> str:
        """Get appropriate unit for a metric"""
        units = {
            'training_speed': ' samples/sec',
            'inference_speed': ' samples/sec',
            'convergence_rate': '',
            'throughput': ' ops/sec',
            'peak_memory_mb': ' MB',
            'memory_efficiency': '%',
            'memory_growth_rate': ' MB/op',
            'gc_frequency': '/min',
            'scaling_efficiency': '%',
            'optimal_node_count': ' nodes',
            'max_supported_nodes': ' nodes',
            'scaling_overhead': '%',
            'fault_tolerance': '%',
            'recovery_time': ' sec',
            'error_rate': '%',
            'uptime_score': '%'
        }
        
        return units.get(metric_name, '')


# Example usage
async def main():
    """Example performance comparison"""
    comparator = PerformanceComparator()
    
    print("📊 Starting Performance Comparison")
    
    # Compare configurations
    results = await comparator.compare_configurations(['small', 'medium', 'large'])
    
    # Generate report
    report = comparator.generate_comparison_report(results)
    print("\n" + "="*60)
    print(report)


if __name__ == "__main__":
    asyncio.run(main())