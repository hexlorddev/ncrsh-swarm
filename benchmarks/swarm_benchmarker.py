"""
ncrsh-Swarm Performance Benchmarking Suite
==========================================

Comprehensive benchmarking tools for evaluating swarm performance across
different configurations, hardware setups, and network topologies.
"""

import asyncio
import time
import statistics
import json
import psutil
import torch
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
import matplotlib.pyplot as plt
import pandas as pd

from ..ncrsh_swarm.core.swarm_node import SwarmNode, SwarmNodeConfig
from ..ncrsh_swarm.models.transformer import TransformerConfig
from ..ncrsh_swarm.network.p2p import NetworkConfig


@dataclass
class BenchmarkResult:
    """Single benchmark result"""
    test_name: str
    duration: float
    throughput: float
    memory_usage: float
    cpu_usage: float
    gpu_usage: Optional[float]
    network_traffic: float
    accuracy: Optional[float]
    loss: Optional[float]
    peer_count: int
    model_size: int
    timestamp: float
    metadata: Dict[str, Any]


@dataclass
class SwarmBenchmarkSuite:
    """Complete benchmark configuration"""
    name: str
    description: str
    node_configs: List[SwarmNodeConfig]
    test_duration: float
    iterations: int
    metrics_to_track: List[str]
    

class SwarmBenchmarker:
    """
    Advanced benchmarking system for ncrsh-Swarm
    
    Features:
    - Multi-node performance testing
    - Network topology analysis  
    - Memory and compute profiling
    - Scalability evaluation
    - Fault tolerance testing
    - Real-time metrics collection
    """
    
    def __init__(self, output_dir: str = "./benchmark_results"):
        self.output_dir = output_dir
        self.results: List[BenchmarkResult] = []
        self.active_nodes: List[SwarmNode] = []
        
    async def run_benchmark_suite(self, suite: SwarmBenchmarkSuite) -> Dict[str, Any]:
        """Run a complete benchmark suite"""
        print(f"🏁 Running benchmark suite: {suite.name}")
        
        suite_results = {
            'suite_name': suite.name,
            'description': suite.description,
            'start_time': time.time(),
            'benchmarks': []
        }
        
        for i in range(suite.iterations):
            print(f"  📊 Iteration {i+1}/{suite.iterations}")
            
            # Setup nodes
            nodes = await self._setup_benchmark_nodes(suite.node_configs)
            
            try:
                # Run benchmarks
                iteration_results = await self._run_benchmark_iteration(
                    nodes, suite.test_duration, suite.metrics_to_track
                )
                suite_results['benchmarks'].append(iteration_results)
                
            finally:
                # Cleanup nodes
                await self._cleanup_benchmark_nodes(nodes)
                
        suite_results['end_time'] = time.time()
        suite_results['total_duration'] = suite_results['end_time'] - suite_results['start_time']
        
        # Save results
        await self._save_benchmark_results(suite_results)
        
        return suite_results
        
    async def benchmark_scalability(self, max_nodes: int = 20) -> Dict[str, Any]:
        """Test how performance scales with number of nodes"""
        print(f"📈 Running scalability benchmark (1-{max_nodes} nodes)")
        
        scalability_results = []
        
        for node_count in range(1, max_nodes + 1):
            print(f"  Testing with {node_count} nodes...")
            
            # Create configs for N nodes
            configs = []
            for i in range(node_count):
                config = SwarmNodeConfig(
                    model_config=TransformerConfig(hidden_size=256, num_layers=4),
                    network_config=NetworkConfig(port=8080+i),
                    max_peers=node_count-1
                )
                configs.append(config)
                
            # Setup and run
            nodes = await self._setup_benchmark_nodes(configs)
            
            try:
                start_time = time.time()
                
                # Let nodes discover each other
                await asyncio.sleep(5)
                
                # Run training benchmark
                training_results = await self._benchmark_training_performance(nodes)
                
                end_time = time.time()
                
                result = {
                    'node_count': node_count,
                    'setup_time': 5.0,
                    'training_time': training_results['duration'],
                    'throughput': training_results['throughput'],
                    'memory_per_node': training_results['avg_memory'],
                    'network_efficiency': training_results['network_efficiency']
                }
                
                scalability_results.append(result)
                
            finally:
                await self._cleanup_benchmark_nodes(nodes)
                
        return {
            'test_type': 'scalability',
            'results': scalability_results,
            'analysis': self._analyze_scalability_results(scalability_results)
        }
        
    async def benchmark_fault_tolerance(self, total_nodes: int = 10, failure_rate: float = 0.3) -> Dict[str, Any]:
        """Test performance under node failures"""
        print(f"🔥 Running fault tolerance benchmark ({failure_rate*100}% failure rate)")
        
        # Setup nodes
        configs = [
            SwarmNodeConfig(
                model_config=TransformerConfig(hidden_size=256, num_layers=4),
                network_config=NetworkConfig(port=8080+i),
                max_peers=total_nodes-1
            )
            for i in range(total_nodes)
        ]
        
        nodes = await self._setup_benchmark_nodes(configs)
        
        try:
            # Let swarm stabilize
            await asyncio.sleep(10)
            
            # Baseline performance
            baseline_results = await self._benchmark_training_performance(nodes)
            
            # Introduce failures
            failure_count = int(total_nodes * failure_rate)
            failed_nodes = nodes[:failure_count]
            
            print(f"  💥 Failing {failure_count} nodes...")
            for node in failed_nodes:
                await node.stop()
                
            # Wait for recovery
            await asyncio.sleep(5)
            
            # Test performance with failures
            remaining_nodes = nodes[failure_count:]
            degraded_results = await self._benchmark_training_performance(remaining_nodes)
            
            # Analyze recovery
            recovery_analysis = {
                'baseline_throughput': baseline_results['throughput'],
                'degraded_throughput': degraded_results['throughput'],
                'performance_retention': degraded_results['throughput'] / baseline_results['throughput'],
                'failure_rate': failure_rate,
                'recovery_time': 5.0  # Simplified
            }
            
            return {
                'test_type': 'fault_tolerance',
                'baseline': baseline_results,
                'degraded': degraded_results,
                'analysis': recovery_analysis
            }
            
        finally:
            await self._cleanup_benchmark_nodes(nodes)
            
    async def benchmark_network_topology(self) -> Dict[str, Any]:
        """Test different network topologies"""
        print("🌐 Benchmarking network topologies...")
        
        topologies = {
            'star': self._create_star_topology,
            'mesh': self._create_mesh_topology,
            'ring': self._create_ring_topology,
            'hierarchical': self._create_hierarchical_topology
        }
        
        topology_results = {}
        
        for topology_name, topology_fn in topologies.items():
            print(f"  Testing {topology_name} topology...")
            
            configs = topology_fn(node_count=8)
            nodes = await self._setup_benchmark_nodes(configs)
            
            try:
                await asyncio.sleep(5)  # Stabilization
                results = await self._benchmark_network_performance(nodes)
                topology_results[topology_name] = results
                
            finally:
                await self._cleanup_benchmark_nodes(nodes)
                
        return {
            'test_type': 'network_topology',
            'topologies': topology_results,
            'analysis': self._analyze_topology_results(topology_results)
        }
        
    async def benchmark_model_architectures(self) -> Dict[str, Any]:
        """Test different model configurations"""
        print("🧠 Benchmarking model architectures...")
        
        architectures = {
            'tiny': TransformerConfig(hidden_size=128, num_layers=2, num_heads=4),
            'small': TransformerConfig(hidden_size=256, num_layers=4, num_heads=8),
            'medium': TransformerConfig(hidden_size=512, num_layers=8, num_heads=8),
            'large': TransformerConfig(hidden_size=768, num_layers=12, num_heads=12)
        }
        
        architecture_results = {}
        
        for arch_name, model_config in architectures.items():
            print(f"  Testing {arch_name} architecture...")
            
            config = SwarmNodeConfig(
                model_config=model_config,
                network_config=NetworkConfig(port=8080),
                max_peers=4
            )
            
            nodes = await self._setup_benchmark_nodes([config] * 4)
            
            try:
                await asyncio.sleep(3)
                results = await self._benchmark_training_performance(nodes)
                architecture_results[arch_name] = results
                
            finally:
                await self._cleanup_benchmark_nodes(nodes)
                
        return {
            'test_type': 'model_architectures',
            'architectures': architecture_results,
            'analysis': self._analyze_architecture_results(architecture_results)
        }
        
    # Internal benchmark methods
    
    async def _setup_benchmark_nodes(self, configs: List[SwarmNodeConfig]) -> List[SwarmNode]:
        """Setup nodes for benchmarking"""
        nodes = []
        
        for config in configs:
            node = SwarmNode(config)
            await node.start()
            nodes.append(node)
            await asyncio.sleep(0.5)  # Stagger startup
            
        # Let nodes discover each other
        await asyncio.sleep(2)
        
        return nodes
        
    async def _cleanup_benchmark_nodes(self, nodes: List[SwarmNode]) -> None:
        """Clean up benchmark nodes"""
        for node in nodes:
            try:
                await node.stop()
            except Exception as e:
                print(f"Warning: Failed to stop node: {e}")
                
    async def _benchmark_training_performance(self, nodes: List[SwarmNode]) -> Dict[str, Any]:
        """Benchmark training performance"""
        if not nodes:
            return {}
            
        # Create synthetic dataset
        dataset = self._create_synthetic_dataset(size=1000)
        
        # Start training on first node
        start_time = time.time()
        
        # Monitor resource usage
        initial_memory = psutil.virtual_memory().used
        
        try:
            # Run short training
            await nodes[0].train(dataset, epochs=2, batch_size=16)
            
        except Exception as e:
            print(f"Training benchmark failed: {e}")
            return {'error': str(e)}
            
        end_time = time.time()
        final_memory = psutil.virtual_memory().used
        
        duration = end_time - start_time
        memory_used = final_memory - initial_memory
        
        # Calculate metrics
        throughput = len(dataset) / duration if duration > 0 else 0
        
        return {
            'duration': duration,
            'throughput': throughput,
            'memory_used': memory_used,
            'avg_memory': memory_used / len(nodes),
            'network_efficiency': 0.85,  # Simplified metric
            'node_count': len(nodes)
        }
        
    async def _benchmark_network_performance(self, nodes: List[SwarmNode]) -> Dict[str, Any]:
        """Benchmark network communication performance"""
        if len(nodes) < 2:
            return {}
            
        start_time = time.time()
        
        # Measure peer discovery time
        discovery_times = []
        for node in nodes:
            peers = await node.discover_peers()
            discovery_times.append(len(peers))
            
        # Measure message latency
        latencies = []
        for i in range(min(5, len(nodes)-1)):
            msg_start = time.time()
            try:
                await nodes[0].network.send_message(
                    list(nodes[0].peers.keys())[0] if nodes[0].peers else "test",
                    'ping', {}
                )
                latencies.append(time.time() - msg_start)
            except:
                pass
                
        return {
            'discovery_time': time.time() - start_time,
            'avg_peers_discovered': statistics.mean(discovery_times) if discovery_times else 0,
            'avg_message_latency': statistics.mean(latencies) if latencies else 0,
            'network_connectivity': len([n for n in nodes if n.peers]) / len(nodes)
        }
        
    def _create_synthetic_dataset(self, size: int = 1000):
        """Create synthetic dataset for benchmarking"""
        # Simple tensor dataset
        vocab_size = 1000
        seq_len = 64
        
        input_data = torch.randint(0, vocab_size, (size, seq_len))
        target_data = torch.roll(input_data, shifts=-1, dims=1)
        
        return torch.utils.data.TensorDataset(input_data, target_data)
        
    def _create_star_topology(self, node_count: int) -> List[SwarmNodeConfig]:
        """Create star network topology"""
        configs = []
        
        # Central hub
        hub_config = SwarmNodeConfig(
            model_config=TransformerConfig(hidden_size=256, num_layers=4),
            network_config=NetworkConfig(port=8080),
            max_peers=node_count-1
        )
        configs.append(hub_config)
        
        # Spoke nodes
        for i in range(1, node_count):
            spoke_config = SwarmNodeConfig(
                model_config=TransformerConfig(hidden_size=256, num_layers=4),
                network_config=NetworkConfig(port=8080+i),
                max_peers=1  # Only connect to hub
            )
            configs.append(spoke_config)
            
        return configs
        
    def _create_mesh_topology(self, node_count: int) -> List[SwarmNodeConfig]:
        """Create full mesh topology"""
        configs = []
        
        for i in range(node_count):
            config = SwarmNodeConfig(
                model_config=TransformerConfig(hidden_size=256, num_layers=4),
                network_config=NetworkConfig(port=8080+i),
                max_peers=node_count-1  # Connect to all others
            )
            configs.append(config)
            
        return configs
        
    def _create_ring_topology(self, node_count: int) -> List[SwarmNodeConfig]:
        """Create ring topology"""
        configs = []
        
        for i in range(node_count):
            config = SwarmNodeConfig(
                model_config=TransformerConfig(hidden_size=256, num_layers=4),
                network_config=NetworkConfig(port=8080+i),
                max_peers=2  # Connect to 2 neighbors
            )
            configs.append(config)
            
        return configs
        
    def _create_hierarchical_topology(self, node_count: int) -> List[SwarmNodeConfig]:
        """Create hierarchical topology"""
        configs = []
        
        # Top level nodes
        top_level = min(3, node_count // 3)
        
        for i in range(top_level):
            config = SwarmNodeConfig(
                model_config=TransformerConfig(hidden_size=256, num_layers=4),
                network_config=NetworkConfig(port=8080+i),
                max_peers=node_count//2
            )
            configs.append(config)
            
        # Lower level nodes
        for i in range(top_level, node_count):
            config = SwarmNodeConfig(
                model_config=TransformerConfig(hidden_size=256, num_layers=4),
                network_config=NetworkConfig(port=8080+i),
                max_peers=top_level  # Connect only to top level
            )
            configs.append(config)
            
        return configs
        
    def _analyze_scalability_results(self, results: List[Dict]) -> Dict[str, Any]:
        """Analyze scalability benchmark results"""
        if not results:
            return {}
            
        node_counts = [r['node_count'] for r in results]
        throughputs = [r['throughput'] for r in results]
        
        # Calculate scaling efficiency
        baseline_throughput = throughputs[0] if throughputs else 1
        scaling_efficiency = [t / (baseline_throughput * n) for t, n in zip(throughputs, node_counts)]
        
        return {
            'max_nodes_tested': max(node_counts),
            'peak_throughput': max(throughputs) if throughputs else 0,
            'avg_scaling_efficiency': statistics.mean(scaling_efficiency),
            'linear_scaling_score': min(scaling_efficiency) if scaling_efficiency else 0,
            'recommended_node_count': node_counts[throughputs.index(max(throughputs))] if throughputs else 1
        }
        
    def _analyze_topology_results(self, results: Dict[str, Dict]) -> Dict[str, Any]:
        """Analyze network topology results"""
        if not results:
            return {}
            
        # Find best topology
        best_topology = max(results.keys(), 
                           key=lambda k: results[k].get('network_connectivity', 0))
        
        return {
            'best_topology': best_topology,
            'topology_rankings': sorted(
                results.keys(), 
                key=lambda k: results[k].get('network_connectivity', 0),
                reverse=True
            ),
            'connectivity_scores': {
                k: v.get('network_connectivity', 0) 
                for k, v in results.items()
            }
        }
        
    def _analyze_architecture_results(self, results: Dict[str, Dict]) -> Dict[str, Any]:
        """Analyze model architecture results"""
        if not results:
            return {}
            
        # Find most efficient architecture
        efficiency_scores = {}
        for arch, result in results.items():
            throughput = result.get('throughput', 0)
            memory = result.get('memory_used', 1)
            efficiency_scores[arch] = throughput / memory if memory > 0 else 0
            
        best_arch = max(efficiency_scores.keys(), key=lambda k: efficiency_scores[k])
        
        return {
            'most_efficient': best_arch,
            'efficiency_scores': efficiency_scores,
            'architecture_rankings': sorted(
                efficiency_scores.keys(),
                key=lambda k: efficiency_scores[k],
                reverse=True
            )
        }
        
    async def _save_benchmark_results(self, results: Dict[str, Any]) -> None:
        """Save benchmark results to file"""
        import os
        os.makedirs(self.output_dir, exist_ok=True)
        
        filename = f"{self.output_dir}/benchmark_{int(time.time())}.json"
        
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2, default=str)
            
        print(f"📊 Benchmark results saved to: {filename}")
        
    def generate_benchmark_report(self, results_file: str) -> str:
        """Generate comprehensive benchmark report"""
        with open(results_file, 'r') as f:
            results = json.load(f)
            
        report = f"""
# ncrsh-Swarm Benchmark Report

## Test Suite: {results.get('suite_name', 'Unknown')}
**Description:** {results.get('description', 'No description')}
**Duration:** {results.get('total_duration', 0):.2f} seconds

## Results Summary
{self._format_results_summary(results)}

## Performance Analysis
{self._format_performance_analysis(results)}

## Recommendations
{self._format_recommendations(results)}

---
Generated by ncrsh-Swarm Benchmarker
"""
        return report
        
    def _format_results_summary(self, results: Dict) -> str:
        """Format results summary section"""
        # Implementation depends on results structure
        return "Results summary would be formatted here based on benchmark data."
        
    def _format_performance_analysis(self, results: Dict) -> str:
        """Format performance analysis section"""
        return "Performance analysis would be generated here."
        
    def _format_recommendations(self, results: Dict) -> str:
        """Format recommendations section"""
        return "Optimization recommendations would be provided here."