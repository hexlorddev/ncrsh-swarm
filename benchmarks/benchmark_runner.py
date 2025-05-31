#!/usr/bin/env python3
"""
Benchmark Runner for ncrsh-Swarm
================================

Run comprehensive performance benchmarks and generate detailed reports.
"""

import asyncio
import argparse
import json
import sys
import time
from pathlib import Path

from swarm_benchmarker import SwarmBenchmarker, SwarmBenchmarkSuite
from load_tester import SwarmLoadTester
from memory_profiler import SwarmMemoryProfiler
from performance_comparator import PerformanceComparator


class BenchmarkRunner:
    """Main benchmark runner orchestrating all tests"""
    
    def __init__(self, output_dir: str = "./benchmark_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.benchmarker = SwarmBenchmarker(str(self.output_dir))
        self.load_tester = SwarmLoadTester(str(self.output_dir))
        self.memory_profiler = SwarmMemoryProfiler(str(self.output_dir))
        self.comparator = PerformanceComparator(str(self.output_dir))
        
    async def run_full_benchmark_suite(self) -> Dict[str, Any]:
        """Run the complete benchmark suite"""
        print("🚀 Starting ncrsh-Swarm Full Benchmark Suite")
        print("=" * 60)
        
        start_time = time.time()
        results = {
            'start_time': start_time,
            'benchmarks': {},
            'system_info': self._get_system_info()
        }
        
        # 1. Scalability Benchmarks
        print("\n📈 Phase 1: Scalability Testing")
        scalability_results = await self.benchmarker.benchmark_scalability(max_nodes=15)
        results['benchmarks']['scalability'] = scalability_results
        
        # 2. Fault Tolerance Testing
        print("\n🔥 Phase 2: Fault Tolerance Testing")
        fault_results = await self.benchmarker.benchmark_fault_tolerance(
            total_nodes=10, failure_rate=0.3
        )
        results['benchmarks']['fault_tolerance'] = fault_results
        
        # 3. Network Topology Analysis
        print("\n🌐 Phase 3: Network Topology Analysis")
        topology_results = await self.benchmarker.benchmark_network_topology()
        results['benchmarks']['network_topology'] = topology_results
        
        # 4. Model Architecture Comparison
        print("\n🧠 Phase 4: Model Architecture Testing")
        architecture_results = await self.benchmarker.benchmark_model_architectures()
        results['benchmarks']['model_architectures'] = architecture_results
        
        # 5. Load Testing
        print("\n⚡ Phase 5: Load Testing")
        load_results = await self.load_tester.run_stress_test(
            concurrent_nodes=20, duration=300
        )
        results['benchmarks']['load_testing'] = load_results
        
        # 6. Memory Profiling
        print("\n🧮 Phase 6: Memory Profiling")
        memory_results = await self.memory_profiler.profile_memory_usage(
            configurations=['small', 'medium', 'large']
        )
        results['benchmarks']['memory_profiling'] = memory_results
        
        # 7. Performance Comparison
        print("\n📊 Phase 7: Performance Comparison")
        comparison_results = await self.comparator.compare_configurations([
            'small', 'medium', 'large', 'distributed'
        ])
        results['benchmarks']['performance_comparison'] = comparison_results
        
        results['end_time'] = time.time()
        results['total_duration'] = results['end_time'] - start_time
        
        # Generate comprehensive report
        await self._generate_comprehensive_report(results)
        
        print(f"\n✅ Benchmark suite completed in {results['total_duration']:.2f} seconds")
        return results
        
    async def run_quick_benchmark(self) -> Dict[str, Any]:
        """Run a quick benchmark for development testing"""
        print("⚡ Running Quick Benchmark Suite")
        print("=" * 40)
        
        start_time = time.time()
        results = {
            'start_time': start_time,
            'benchmarks': {},
            'system_info': self._get_system_info()
        }
        
        # Quick scalability test (up to 5 nodes)
        print("\n📈 Quick Scalability Test (1-5 nodes)")
        scalability_results = await self.benchmarker.benchmark_scalability(max_nodes=5)
        results['benchmarks']['scalability'] = scalability_results
        
        # Quick memory test
        print("\n🧮 Quick Memory Test")
        memory_results = await self.memory_profiler.profile_memory_usage(
            configurations=['small']
        )
        results['benchmarks']['memory'] = memory_results
        
        results['end_time'] = time.time()
        results['total_duration'] = results['end_time'] - start_time
        
        print(f"\n✅ Quick benchmark completed in {results['total_duration']:.2f} seconds")
        return results
        
    async def run_custom_benchmark(self, config_file: str) -> Dict[str, Any]:
        """Run benchmark from custom configuration file"""
        print(f"🔧 Running Custom Benchmark: {config_file}")
        
        with open(config_file, 'r') as f:
            config = json.load(f)
            
        # Execute benchmarks based on config
        results = {'custom_config': config, 'benchmarks': {}}
        
        if config.get('scalability', {}).get('enabled', False):
            max_nodes = config['scalability'].get('max_nodes', 10)
            results['benchmarks']['scalability'] = await self.benchmarker.benchmark_scalability(max_nodes)
            
        if config.get('fault_tolerance', {}).get('enabled', False):
            failure_rate = config['fault_tolerance'].get('failure_rate', 0.3)
            results['benchmarks']['fault_tolerance'] = await self.benchmarker.benchmark_fault_tolerance(
                failure_rate=failure_rate
            )
            
        return results
        
    def _get_system_info(self) -> Dict[str, Any]:
        """Get system information for benchmark context"""
        import platform
        import psutil
        
        try:
            import torch
            cuda_available = torch.cuda.is_available()
            cuda_devices = torch.cuda.device_count() if cuda_available else 0
        except ImportError:
            cuda_available = False
            cuda_devices = 0
            
        return {
            'platform': platform.platform(),
            'python_version': platform.python_version(),
            'cpu_count': psutil.cpu_count(),
            'memory_gb': psutil.virtual_memory().total / (1024**3),
            'cuda_available': cuda_available,
            'cuda_devices': cuda_devices,
            'timestamp': time.time()
        }
        
    async def _generate_comprehensive_report(self, results: Dict[str, Any]) -> None:
        """Generate a comprehensive benchmark report"""
        report_file = self.output_dir / f"comprehensive_report_{int(time.time())}.md"
        
        report_content = f"""# ncrsh-Swarm Comprehensive Benchmark Report

## System Information
- **Platform:** {results['system_info']['platform']}
- **Python:** {results['system_info']['python_version']}
- **CPU Cores:** {results['system_info']['cpu_count']}
- **Memory:** {results['system_info']['memory_gb']:.1f} GB
- **CUDA Available:** {results['system_info']['cuda_available']}
- **CUDA Devices:** {results['system_info']['cuda_devices']}
- **Test Duration:** {results['total_duration']:.2f} seconds

## Executive Summary

{self._generate_executive_summary(results)}

## Detailed Results

### 📈 Scalability Analysis
{self._format_scalability_results(results['benchmarks'].get('scalability', {}))}

### 🔥 Fault Tolerance Assessment
{self._format_fault_tolerance_results(results['benchmarks'].get('fault_tolerance', {}))}

### 🌐 Network Topology Performance
{self._format_topology_results(results['benchmarks'].get('network_topology', {}))}

### 🧠 Model Architecture Comparison
{self._format_architecture_results(results['benchmarks'].get('model_architectures', {}))}

### ⚡ Load Testing Results
{self._format_load_testing_results(results['benchmarks'].get('load_testing', {}))}

### 🧮 Memory Profiling
{self._format_memory_results(results['benchmarks'].get('memory_profiling', {}))}

### 📊 Performance Comparison
{self._format_comparison_results(results['benchmarks'].get('performance_comparison', {}))}

## Recommendations

{self._generate_recommendations(results)}

## Performance Optimization Tips

{self._generate_optimization_tips(results)}

---
*Report generated by ncrsh-Swarm Benchmark Runner*
*Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(results['start_time']))}*
"""
        
        with open(report_file, 'w') as f:
            f.write(report_content)
            
        print(f"📄 Comprehensive report saved: {report_file}")
        
    def _generate_executive_summary(self, results: Dict[str, Any]) -> str:
        """Generate executive summary of benchmark results"""
        benchmarks = results.get('benchmarks', {})
        
        summary_points = []
        
        # Scalability summary
        if 'scalability' in benchmarks:
            scalability = benchmarks['scalability']
            analysis = scalability.get('analysis', {})
            max_nodes = analysis.get('max_nodes_tested', 0)
            efficiency = analysis.get('avg_scaling_efficiency', 0)
            summary_points.append(f"- **Scalability:** Tested up to {max_nodes} nodes with {efficiency:.1%} avg efficiency")
            
        # Fault tolerance summary
        if 'fault_tolerance' in benchmarks:
            fault = benchmarks['fault_tolerance']
            analysis = fault.get('analysis', {})
            retention = analysis.get('performance_retention', 0)
            summary_points.append(f"- **Fault Tolerance:** {retention:.1%} performance retention under failures")
            
        # Memory summary
        if 'memory_profiling' in benchmarks:
            summary_points.append("- **Memory Usage:** Profiled across multiple configurations")
            
        return "\n".join(summary_points) if summary_points else "No benchmark data available."
        
    def _format_scalability_results(self, results: Dict[str, Any]) -> str:
        """Format scalability results section"""
        if not results:
            return "No scalability data available."
            
        analysis = results.get('analysis', {})
        
        return f"""
**Key Metrics:**
- Maximum nodes tested: {analysis.get('max_nodes_tested', 'N/A')}
- Peak throughput: {analysis.get('peak_throughput', 0):.2f} samples/sec
- Average scaling efficiency: {analysis.get('avg_scaling_efficiency', 0):.1%}
- Recommended node count: {analysis.get('recommended_node_count', 'N/A')}

**Analysis:**
The swarm demonstrates {'excellent' if analysis.get('avg_scaling_efficiency', 0) > 0.8 else 'good' if analysis.get('avg_scaling_efficiency', 0) > 0.6 else 'moderate'} scaling characteristics.
"""
        
    def _format_fault_tolerance_results(self, results: Dict[str, Any]) -> str:
        """Format fault tolerance results section"""
        if not results:
            return "No fault tolerance data available."
            
        analysis = results.get('analysis', {})
        
        return f"""
**Key Metrics:**
- Performance retention: {analysis.get('performance_retention', 0):.1%}
- Failure rate tested: {analysis.get('failure_rate', 0):.1%}
- Recovery time: {analysis.get('recovery_time', 0):.1f} seconds

**Analysis:**
The swarm shows {'excellent' if analysis.get('performance_retention', 0) > 0.8 else 'good' if analysis.get('performance_retention', 0) > 0.6 else 'moderate'} fault tolerance.
"""
        
    def _format_topology_results(self, results: Dict[str, Any]) -> str:
        """Format topology results section"""
        if not results:
            return "No topology data available."
            
        analysis = results.get('analysis', {})
        
        return f"""
**Best Topology:** {analysis.get('best_topology', 'Unknown')}

**Topology Rankings:**
{chr(10).join(f"{i+1}. {topo}" for i, topo in enumerate(analysis.get('topology_rankings', [])))}

**Connectivity Scores:**
{chr(10).join(f"- {k}: {v:.1%}" for k, v in analysis.get('connectivity_scores', {}).items())}
"""
        
    def _format_architecture_results(self, results: Dict[str, Any]) -> str:
        """Format architecture results section"""
        if not results:
            return "No architecture data available."
            
        analysis = results.get('analysis', {})
        
        return f"""
**Most Efficient Architecture:** {analysis.get('most_efficient', 'Unknown')}

**Efficiency Rankings:**
{chr(10).join(f"{i+1}. {arch}" for i, arch in enumerate(analysis.get('architecture_rankings', [])))}

**Efficiency Scores:**
{chr(10).join(f"- {k}: {v:.3f}" for k, v in analysis.get('efficiency_scores', {}).items())}
"""
        
    def _format_load_testing_results(self, results: Dict[str, Any]) -> str:
        """Format load testing results section"""
        if not results:
            return "No load testing data available."
            
        return f"""
**Load Testing Results:**
- Maximum concurrent nodes: {results.get('max_nodes', 'N/A')}
- Test duration: {results.get('duration', 0):.1f} seconds
- Success rate: {results.get('success_rate', 0):.1%}
- Average response time: {results.get('avg_response_time', 0):.3f} seconds
"""
        
    def _format_memory_results(self, results: Dict[str, Any]) -> str:
        """Format memory results section"""
        if not results:
            return "No memory profiling data available."
            
        return f"""
**Memory Usage Analysis:**
- Peak memory usage: {results.get('peak_memory', 0)} MB
- Average memory per node: {results.get('avg_memory_per_node', 0)} MB
- Memory efficiency score: {results.get('efficiency_score', 0):.2f}
"""
        
    def _format_comparison_results(self, results: Dict[str, Any]) -> str:
        """Format comparison results section"""
        if not results:
            return "No comparison data available."
            
        return f"""
**Performance Comparison:**
- Best configuration: {results.get('best_config', 'Unknown')}
- Performance spread: {results.get('performance_spread', 0):.1%}
- Recommended for production: {results.get('production_recommendation', 'Unknown')}
"""
        
    def _generate_recommendations(self, results: Dict[str, Any]) -> str:
        """Generate optimization recommendations"""
        recommendations = []
        
        # Based on scalability results
        benchmarks = results.get('benchmarks', {})
        if 'scalability' in benchmarks:
            scalability = benchmarks['scalability']
            analysis = scalability.get('analysis', {})
            recommended_nodes = analysis.get('recommended_node_count', 0)
            if recommended_nodes > 0:
                recommendations.append(f"- **Optimal Node Count:** Use {recommended_nodes} nodes for best performance")
                
        # Based on fault tolerance
        if 'fault_tolerance' in benchmarks:
            fault = benchmarks['fault_tolerance']
            analysis = fault.get('analysis', {})
            retention = analysis.get('performance_retention', 0)
            if retention < 0.7:
                recommendations.append("- **Fault Tolerance:** Consider increasing replication factor")
                
        # Based on topology
        if 'network_topology' in benchmarks:
            topology = benchmarks['network_topology']
            analysis = topology.get('analysis', {})
            best_topology = analysis.get('best_topology', '')
            if best_topology:
                recommendations.append(f"- **Network Topology:** Use {best_topology} topology for optimal connectivity")
                
        return "\n".join(recommendations) if recommendations else "No specific recommendations at this time."
        
    def _generate_optimization_tips(self, results: Dict[str, Any]) -> str:
        """Generate performance optimization tips"""
        tips = [
            "- **Memory Optimization:** Enable gradient checkpointing for large models",
            "- **Network Optimization:** Use compression for gradient sharing",
            "- **Training Optimization:** Adjust sync frequency based on network latency",
            "- **Resource Management:** Monitor memory usage and scale nodes accordingly",
            "- **Security:** Enable cryptographic verification in production environments"
        ]
        
        return "\n".join(tips)


async def main():
    """Main entry point for benchmark runner"""
    parser = argparse.ArgumentParser(description="ncrsh-Swarm Benchmark Runner")
    parser.add_argument('--mode', choices=['full', 'quick', 'custom'], default='quick',
                        help='Benchmark mode to run')
    parser.add_argument('--config', help='Custom benchmark configuration file')
    parser.add_argument('--output', default='./benchmark_results',
                        help='Output directory for results')
    
    args = parser.parse_args()
    
    runner = BenchmarkRunner(args.output)
    
    try:
        if args.mode == 'full':
            results = await runner.run_full_benchmark_suite()
        elif args.mode == 'quick':
            results = await runner.run_quick_benchmark()
        elif args.mode == 'custom':
            if not args.config:
                print("❌ Custom mode requires --config argument")
                sys.exit(1)
            results = await runner.run_custom_benchmark(args.config)
            
        print(f"\n🎉 Benchmark completed successfully!")
        print(f"📄 Results saved to: {args.output}")
        
    except KeyboardInterrupt:
        print("\n⏹️  Benchmark interrupted by user")
    except Exception as e:
        print(f"\n❌ Benchmark failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())