"""
Load Testing for ncrsh-Swarm
============================

Stress testing and load analysis for swarm networks under high traffic
and concurrent node scenarios.
"""

import asyncio
import time
import statistics
import random
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import json

from ..ncrsh_swarm.core.swarm_node import SwarmNode, SwarmNodeConfig
from ..ncrsh_swarm.models.transformer import TransformerConfig
from ..ncrsh_swarm.network.p2p import NetworkConfig


@dataclass
class LoadTestResult:
    """Result from a load test"""
    test_name: str
    duration: float
    total_requests: int
    successful_requests: int
    failed_requests: int
    avg_response_time: float
    max_response_time: float
    min_response_time: float
    throughput: float
    error_rate: float
    memory_peak: float
    cpu_peak: float
    network_peak: float


class SwarmLoadTester:
    """
    Advanced load testing for ncrsh-Swarm networks
    
    Features:
    - Concurrent node stress testing
    - Message flooding scenarios
    - Resource exhaustion testing
    - Gradual load ramping
    - Breaking point detection
    - Recovery time measurement
    """
    
    def __init__(self, output_dir: str = "./load_test_results"):
        self.output_dir = output_dir
        self.results: List[LoadTestResult] = []
        
    async def run_stress_test(self, concurrent_nodes: int = 50, duration: float = 300) -> Dict[str, Any]:
        """Run comprehensive stress test"""
        print(f"⚡ Running stress test: {concurrent_nodes} nodes for {duration}s")
        
        stress_results = {
            'test_type': 'stress_test',
            'parameters': {
                'concurrent_nodes': concurrent_nodes,
                'duration': duration
            },
            'phases': {},
            'summary': {}
        }
        
        # Phase 1: Gradual ramp-up
        print("  📈 Phase 1: Gradual ramp-up")
        rampup_results = await self._run_rampup_test(concurrent_nodes, duration * 0.3)
        stress_results['phases']['rampup'] = rampup_results
        
        # Phase 2: Sustained load
        print("  ⚖️  Phase 2: Sustained load")
        sustained_results = await self._run_sustained_load(concurrent_nodes, duration * 0.4)
        stress_results['phases']['sustained'] = sustained_results
        
        # Phase 3: Peak load burst
        print("  💥 Phase 3: Peak load burst")
        burst_results = await self._run_burst_test(concurrent_nodes * 1.5, duration * 0.2)
        stress_results['phases']['burst'] = burst_results
        
        # Phase 4: Graceful degradation
        print("  📉 Phase 4: Graceful degradation")
        degradation_results = await self._run_degradation_test(concurrent_nodes, duration * 0.1)
        stress_results['phases']['degradation'] = degradation_results
        
        # Analyze overall results
        stress_results['summary'] = self._analyze_stress_results(stress_results['phases'])
        
        return stress_results
        
    async def run_breaking_point_test(self, max_nodes: int = 100) -> Dict[str, Any]:
        """Find the breaking point of the swarm"""
        print(f"🔥 Finding breaking point (up to {max_nodes} nodes)")
        
        breaking_point_results = {
            'test_type': 'breaking_point',
            'max_nodes_tested': max_nodes,
            'breaking_point': None,
            'performance_curve': [],
            'recovery_analysis': {}
        }
        
        current_nodes = 5
        step_size = 5
        
        while current_nodes <= max_nodes:
            print(f"  Testing {current_nodes} nodes...")
            
            # Test current load level
            test_result = await self._test_load_level(current_nodes, duration=60)
            breaking_point_results['performance_curve'].append({
                'node_count': current_nodes,
                'success_rate': test_result['success_rate'],
                'avg_response_time': test_result['avg_response_time'],
                'throughput': test_result['throughput']
            })
            
            # Check if we've hit the breaking point
            if test_result['success_rate'] < 0.8 or test_result['avg_response_time'] > 5.0:
                breaking_point_results['breaking_point'] = current_nodes
                print(f"  💥 Breaking point found at {current_nodes} nodes")
                
                # Test recovery
                recovery_result = await self._test_recovery(current_nodes // 2)
                breaking_point_results['recovery_analysis'] = recovery_result
                break
                
            current_nodes += step_size
            
        return breaking_point_results
        
    async def run_message_flood_test(self, node_count: int = 10, messages_per_second: int = 1000) -> Dict[str, Any]:
        """Test swarm under message flooding"""
        print(f"🌊 Message flood test: {messages_per_second} msgs/sec across {node_count} nodes")
        
        flood_results = {
            'test_type': 'message_flood',
            'parameters': {
                'node_count': node_count,
                'target_mps': messages_per_second
            },
            'actual_performance': {},
            'bottlenecks': [],
            'recommendations': []
        }
        
        # Setup nodes
        nodes = await self._setup_load_test_nodes(node_count)
        
        try:
            start_time = time.time()
            
            # Generate message flood
            message_tasks = []
            for i in range(messages_per_second):
                delay = i / messages_per_second  # Spread over 1 second
                task = asyncio.create_task(
                    self._send_delayed_message(nodes, delay)
                )
                message_tasks.append(task)
                
            # Wait for all messages
            results = await asyncio.gather(*message_tasks, return_exceptions=True)
            
            end_time = time.time()
            
            # Analyze results
            successful = sum(1 for r in results if not isinstance(r, Exception))
            failed = len(results) - successful
            
            flood_results['actual_performance'] = {
                'duration': end_time - start_time,
                'messages_sent': len(results),
                'successful': successful,
                'failed': failed,
                'success_rate': successful / len(results) if results else 0,
                'actual_mps': successful / (end_time - start_time) if end_time > start_time else 0
            }
            
            # Identify bottlenecks
            if flood_results['actual_performance']['success_rate'] < 0.9:
                flood_results['bottlenecks'].append('Message delivery failure rate too high')
                
            if flood_results['actual_performance']['actual_mps'] < messages_per_second * 0.8:
                flood_results['bottlenecks'].append('Throughput below target')
                
        finally:
            await self._cleanup_load_test_nodes(nodes)
            
        return flood_results
        
    async def run_resource_exhaustion_test(self, node_count: int = 5) -> Dict[str, Any]:
        """Test behavior under resource exhaustion"""
        print(f"🔋 Resource exhaustion test with {node_count} nodes")
        
        exhaustion_results = {
            'test_type': 'resource_exhaustion',
            'memory_exhaustion': {},
            'cpu_exhaustion': {},
            'network_exhaustion': {},
            'recovery_times': {}
        }
        
        nodes = await self._setup_load_test_nodes(node_count)
        
        try:
            # Test memory exhaustion
            print("  🧠 Testing memory exhaustion...")
            memory_result = await self._test_memory_exhaustion(nodes)
            exhaustion_results['memory_exhaustion'] = memory_result
            
            # Test CPU exhaustion  
            print("  ⚙️  Testing CPU exhaustion...")
            cpu_result = await self._test_cpu_exhaustion(nodes)
            exhaustion_results['cpu_exhaustion'] = cpu_result
            
            # Test network exhaustion
            print("  🌐 Testing network exhaustion...")
            network_result = await self._test_network_exhaustion(nodes)
            exhaustion_results['network_exhaustion'] = network_result
            
        finally:
            await self._cleanup_load_test_nodes(nodes)
            
        return exhaustion_results
        
    # Internal test methods
    
    async def _run_rampup_test(self, target_nodes: int, duration: float) -> Dict[str, Any]:
        """Gradually increase load to target"""
        rampup_data = []
        
        for i in range(1, target_nodes + 1, max(1, target_nodes // 10)):
            test_result = await self._test_load_level(i, duration / 10)
            rampup_data.append({
                'node_count': i,
                'performance': test_result
            })
            
        return {
            'phase': 'rampup',
            'duration': duration,
            'data_points': rampup_data,
            'final_node_count': target_nodes
        }
        
    async def _run_sustained_load(self, node_count: int, duration: float) -> Dict[str, Any]:
        """Run sustained load at target level"""
        start_time = time.time()
        
        # Sample performance every 30 seconds
        sample_interval = 30.0
        samples = []
        
        while time.time() - start_time < duration:
            sample_result = await self._test_load_level(node_count, sample_interval)
            samples.append({
                'timestamp': time.time() - start_time,
                'performance': sample_result
            })
            
        return {
            'phase': 'sustained',
            'duration': duration,
            'node_count': node_count,
            'samples': samples,
            'stability_score': self._calculate_stability_score(samples)
        }
        
    async def _run_burst_test(self, burst_nodes: int, duration: float) -> Dict[str, Any]:
        """Run burst load test"""
        # Quick burst of high load
        burst_result = await self._test_load_level(int(burst_nodes), duration)
        
        return {
            'phase': 'burst',
            'duration': duration,
            'burst_node_count': int(burst_nodes),
            'performance': burst_result,
            'burst_impact': burst_result.get('success_rate', 0) < 0.9
        }
        
    async def _run_degradation_test(self, initial_nodes: int, duration: float) -> Dict[str, Any]:
        """Test graceful degradation"""
        # Simulate gradual node failures
        degradation_data = []
        
        for failure_rate in [0.1, 0.2, 0.3, 0.4, 0.5]:
            remaining_nodes = int(initial_nodes * (1 - failure_rate))
            if remaining_nodes < 1:
                break
                
            test_result = await self._test_load_level(remaining_nodes, duration / 5)
            degradation_data.append({
                'failure_rate': failure_rate,
                'remaining_nodes': remaining_nodes,
                'performance': test_result
            })
            
        return {
            'phase': 'degradation',
            'duration': duration,
            'initial_nodes': initial_nodes,
            'degradation_curve': degradation_data
        }
        
    async def _test_load_level(self, node_count: int, duration: float) -> Dict[str, Any]:
        """Test performance at specific load level"""
        nodes = await self._setup_load_test_nodes(node_count)
        
        try:
            start_time = time.time()
            
            # Generate load
            tasks = []
            for _ in range(node_count * 10):  # 10 operations per node
                task = asyncio.create_task(self._simulate_operation(nodes))
                tasks.append(task)
                
            # Execute with timeout
            try:
                results = await asyncio.wait_for(
                    asyncio.gather(*tasks, return_exceptions=True),
                    timeout=duration
                )
            except asyncio.TimeoutError:
                results = [Exception("Timeout")] * len(tasks)
                
            end_time = time.time()
            
            # Analyze results
            successful = sum(1 for r in results if not isinstance(r, Exception))
            failed = len(results) - successful
            
            response_times = [
                r for r in results 
                if isinstance(r, (int, float)) and r > 0
            ]
            
            return {
                'node_count': node_count,
                'duration': end_time - start_time,
                'total_operations': len(results),
                'successful_operations': successful,
                'failed_operations': failed,
                'success_rate': successful / len(results) if results else 0,
                'avg_response_time': statistics.mean(response_times) if response_times else 0,
                'throughput': successful / (end_time - start_time) if end_time > start_time else 0
            }
            
        finally:
            await self._cleanup_load_test_nodes(nodes)
            
    async def _setup_load_test_nodes(self, count: int) -> List[SwarmNode]:
        """Setup nodes for load testing"""
        nodes = []
        
        for i in range(count):
            config = SwarmNodeConfig(
                model_config=TransformerConfig(
                    hidden_size=128,  # Small for load testing
                    num_layers=2,
                    num_heads=4
                ),
                network_config=NetworkConfig(port=9000+i),
                max_peers=min(count-1, 10)
            )
            
            node = SwarmNode(config)
            
            try:
                await asyncio.wait_for(node.start(), timeout=10.0)
                nodes.append(node)
                await asyncio.sleep(0.1)  # Stagger startup
            except asyncio.TimeoutError:
                print(f"Warning: Node {i} startup timeout")
                break
            except Exception as e:
                print(f"Warning: Node {i} startup failed: {e}")
                break
                
        return nodes
        
    async def _cleanup_load_test_nodes(self, nodes: List[SwarmNode]) -> None:
        """Cleanup load test nodes"""
        cleanup_tasks = []
        for node in nodes:
            task = asyncio.create_task(self._safe_stop_node(node))
            cleanup_tasks.append(task)
            
        await asyncio.gather(*cleanup_tasks, return_exceptions=True)
        
    async def _safe_stop_node(self, node: SwarmNode) -> None:
        """Safely stop a node with timeout"""
        try:
            await asyncio.wait_for(node.stop(), timeout=5.0)
        except Exception as e:
            print(f"Warning: Failed to stop node: {e}")
            
    async def _simulate_operation(self, nodes: List[SwarmNode]) -> float:
        """Simulate a swarm operation and return response time"""
        if not nodes:
            raise Exception("No nodes available")
            
        start_time = time.time()
        
        try:
            # Random operation simulation
            operation_type = random.choice(['discover', 'message', 'status'])
            
            if operation_type == 'discover':
                await nodes[0].discover_peers()
            elif operation_type == 'message' and len(nodes) > 1:
                # Simulate message between nodes
                await asyncio.sleep(random.uniform(0.001, 0.01))
            elif operation_type == 'status':
                await nodes[0].get_swarm_status()
                
            return time.time() - start_time
            
        except Exception as e:
            raise Exception(f"Operation failed: {e}")
            
    async def _send_delayed_message(self, nodes: List[SwarmNode], delay: float) -> bool:
        """Send a message with specified delay"""
        await asyncio.sleep(delay)
        
        try:
            if len(nodes) >= 2:
                # Simulate message between random nodes
                sender = random.choice(nodes)
                if sender.peers:
                    peer_id = random.choice(list(sender.peers.keys()))
                    await sender.network.send_message(peer_id, 'test_message', {'data': 'load_test'})
                    return True
            return False
        except Exception:
            return False
            
    async def _test_memory_exhaustion(self, nodes: List[SwarmNode]) -> Dict[str, Any]:
        """Test memory exhaustion scenarios"""
        # Simulate memory-intensive operations
        large_data = []
        
        try:
            # Gradually increase memory usage
            for i in range(100):
                large_data.append([random.random() for _ in range(10000)])
                await asyncio.sleep(0.1)
                
            return {
                'memory_allocated': len(large_data) * 10000 * 8,  # bytes
                'nodes_responsive': len([n for n in nodes if n.is_running]),
                'memory_exhaustion_reached': True
            }
            
        except MemoryError:
            return {
                'memory_allocated': len(large_data) * 10000 * 8,
                'nodes_responsive': len([n for n in nodes if n.is_running]),
                'memory_exhaustion_reached': True
            }
        finally:
            del large_data
            
    async def _test_cpu_exhaustion(self, nodes: List[SwarmNode]) -> Dict[str, Any]:
        """Test CPU exhaustion scenarios"""
        # CPU-intensive computation
        start_time = time.time()
        
        # Run CPU-intensive task for 10 seconds
        end_time = start_time + 10
        computation_count = 0
        
        while time.time() < end_time:
            # Simulate CPU-intensive work
            _ = sum(i**2 for i in range(1000))
            computation_count += 1
            
            if computation_count % 100 == 0:
                await asyncio.sleep(0.001)  # Brief yield
                
        return {
            'duration': time.time() - start_time,
            'computations_completed': computation_count,
            'nodes_responsive': len([n for n in nodes if n.is_running]),
            'cpu_exhaustion_impact': computation_count < 1000
        }
        
    async def _test_network_exhaustion(self, nodes: List[SwarmNode]) -> Dict[str, Any]:
        """Test network exhaustion scenarios"""
        if len(nodes) < 2:
            return {'error': 'Need at least 2 nodes for network test'}
            
        # Generate network flood
        message_tasks = []
        
        for _ in range(1000):  # Send 1000 messages rapidly
            task = asyncio.create_task(
                self._send_delayed_message(nodes, random.uniform(0, 1))
            )
            message_tasks.append(task)
            
        start_time = time.time()
        results = await asyncio.gather(*message_tasks, return_exceptions=True)
        end_time = time.time()
        
        successful = sum(1 for r in results if r is True)
        
        return {
            'duration': end_time - start_time,
            'messages_attempted': len(results),
            'messages_successful': successful,
            'network_success_rate': successful / len(results) if results else 0,
            'nodes_responsive': len([n for n in nodes if n.is_running])
        }
        
    async def _test_recovery(self, reduced_nodes: int) -> Dict[str, Any]:
        """Test recovery after breaking point"""
        print(f"  🔄 Testing recovery with {reduced_nodes} nodes...")
        
        recovery_start = time.time()
        
        # Test at reduced load
        recovery_result = await self._test_load_level(reduced_nodes, duration=30)
        
        recovery_time = time.time() - recovery_start
        
        return {
            'recovery_time': recovery_time,
            'reduced_node_count': reduced_nodes,
            'recovery_performance': recovery_result,
            'recovery_successful': recovery_result.get('success_rate', 0) > 0.9
        }
        
    def _analyze_stress_results(self, phases: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze overall stress test results"""
        summary = {
            'overall_success': True,
            'critical_issues': [],
            'performance_degradation': [],
            'recommendations': []
        }
        
        # Check each phase for issues
        for phase_name, phase_data in phases.items():
            if phase_name == 'burst':
                if phase_data.get('burst_impact', False):
                    summary['critical_issues'].append(f"Performance degraded during burst phase")
                    
        # Add recommendations based on findings
        if summary['critical_issues']:
            summary['recommendations'].append("Consider increasing resource allocation")
            summary['recommendations'].append("Implement better load balancing")
            
        return summary
        
    def _calculate_stability_score(self, samples: List[Dict]) -> float:
        """Calculate stability score from performance samples"""
        if not samples:
            return 0.0
            
        success_rates = [s['performance'].get('success_rate', 0) for s in samples]
        
        if not success_rates:
            return 0.0
            
        # Stability is based on consistency of success rates
        avg_success = statistics.mean(success_rates)
        std_success = statistics.stdev(success_rates) if len(success_rates) > 1 else 0
        
        # Higher score for high average and low standard deviation
        stability_score = avg_success * (1 - std_success)
        
        return max(0.0, min(1.0, stability_score))


# Example usage
async def main():
    """Example load testing execution"""
    load_tester = SwarmLoadTester()
    
    print("🚀 Starting Load Testing Suite")
    
    # Quick stress test
    stress_results = await load_tester.run_stress_test(concurrent_nodes=10, duration=60)
    print(f"Stress test completed: {len(stress_results['phases'])} phases")
    
    # Breaking point test
    breaking_results = await load_tester.run_breaking_point_test(max_nodes=20)
    breaking_point = breaking_results.get('breaking_point')
    if breaking_point:
        print(f"Breaking point found at {breaking_point} nodes")
    else:
        print("No breaking point found within test limits")


if __name__ == "__main__":
    asyncio.run(main())