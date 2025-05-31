"""
ncrsh-Swarm Utility Tools Collection
===================================

Command-line tools and utilities for managing, monitoring, and optimizing
ncrsh-Swarm deployments across distributed environments.
"""

import asyncio
import argparse
import json
import time
import sys
import subprocess
from typing import Dict, List, Any, Optional
from pathlib import Path
import psutil
import socket
import platform

from ..ncrsh_swarm.core.swarm_node import SwarmNode, SwarmNodeConfig
from ..ncrsh_swarm.core.config import ConfigManager
from ..benchmarks.swarm_benchmarker import SwarmBenchmarker
from ..datasets.dataset_manager import DatasetManager


class SwarmHealthChecker:
    """Health monitoring and diagnostics for swarm networks"""
    
    def __init__(self):
        self.config_manager = ConfigManager()
        
    async def check_system_health(self) -> Dict[str, Any]:
        """Comprehensive system health check"""
        health_report = {
            'timestamp': time.time(),
            'system_info': self._get_system_info(),
            'network_health': await self._check_network_health(),
            'resource_health': self._check_resource_health(),
            'dependency_health': self._check_dependencies(),
            'swarm_health': await self._check_swarm_health(),
            'overall_status': 'unknown'
        }
        
        # Determine overall health
        issues = []
        if not health_report['network_health']['healthy']:
            issues.append('network')
        if not health_report['resource_health']['healthy']:
            issues.append('resources')
        if not health_report['dependency_health']['healthy']:
            issues.append('dependencies')
            
        health_report['overall_status'] = 'healthy' if not issues else 'unhealthy'
        health_report['issues'] = issues
        
        return health_report
        
    def _get_system_info(self) -> Dict[str, Any]:
        """Get comprehensive system information"""
        return {
            'platform': platform.platform(),
            'python_version': platform.python_version(),
            'cpu_count': psutil.cpu_count(),
            'memory_total_gb': psutil.virtual_memory().total / (1024**3),
            'disk_free_gb': psutil.disk_usage('/').free / (1024**3),
            'hostname': socket.gethostname(),
            'ip_address': self._get_local_ip()
        }
        
    async def _check_network_health(self) -> Dict[str, Any]:
        """Check network connectivity and ports"""
        health = {
            'healthy': True,
            'issues': [],
            'port_availability': {},
            'dns_resolution': True,
            'internet_connectivity': True
        }
        
        # Check common ports
        ports_to_check = [8080, 8081, 8082]  # Default swarm ports
        
        for port in ports_to_check:
            is_available = self._check_port_available(port)
            health['port_availability'][port] = is_available
            
            if not is_available:
                health['issues'].append(f"Port {port} is not available")
                health['healthy'] = False
                
        # Check DNS resolution
        try:
            socket.gethostbyname('google.com')
        except socket.gaierror:
            health['dns_resolution'] = False
            health['issues'].append("DNS resolution failed")
            health['healthy'] = False
            
        # Check internet connectivity
        try:
            import urllib.request
            urllib.request.urlopen('http://google.com', timeout=5)
        except Exception:
            health['internet_connectivity'] = False
            health['issues'].append("No internet connectivity")
            
        return health
        
    def _check_resource_health(self) -> Dict[str, Any]:
        """Check system resource availability"""
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        cpu_percent = psutil.cpu_percent(interval=1)
        
        health = {
            'healthy': True,
            'issues': [],
            'memory_usage_percent': memory.percent,
            'disk_usage_percent': disk.used / disk.total * 100,
            'cpu_usage_percent': cpu_percent
        }
        
        # Check thresholds
        if memory.percent > 90:
            health['issues'].append(f"High memory usage: {memory.percent:.1f}%")
            health['healthy'] = False
            
        if health['disk_usage_percent'] > 95:
            health['issues'].append(f"High disk usage: {health['disk_usage_percent']:.1f}%")
            health['healthy'] = False
            
        if cpu_percent > 95:
            health['issues'].append(f"High CPU usage: {cpu_percent:.1f}%")
            health['healthy'] = False
            
        return health
        
    def _check_dependencies(self) -> Dict[str, Any]:
        """Check required dependencies"""
        health = {
            'healthy': True,
            'issues': [],
            'dependencies': {}
        }
        
        required_packages = [
            'torch', 'numpy', 'aiohttp', 'websockets', 'cryptography', 'msgpack'
        ]
        
        for package in required_packages:
            try:
                __import__(package)
                health['dependencies'][package] = 'available'
            except ImportError:
                health['dependencies'][package] = 'missing'
                health['issues'].append(f"Missing package: {package}")
                health['healthy'] = False
                
        return health
        
    async def _check_swarm_health(self) -> Dict[str, Any]:
        """Check swarm-specific health"""
        health = {
            'healthy': True,
            'issues': [],
            'config_valid': True,
            'models_loadable': True
        }
        
        try:
            # Test configuration loading
            config = self.config_manager.get_default_config()
            warnings = self.config_manager.validate_config(config)
            
            if warnings:
                health['issues'].extend(warnings)
                health['config_valid'] = False
                
        except Exception as e:
            health['config_valid'] = False
            health['issues'].append(f"Config validation failed: {e}")
            health['healthy'] = False
            
        return health
        
    def _check_port_available(self, port: int) -> bool:
        """Check if a port is available"""
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            sock.bind(('localhost', port))
            sock.listen(1)
            return True
        except OSError:
            return False
        finally:
            sock.close()
            
    def _get_local_ip(self) -> str:
        """Get local IP address"""
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.connect(("8.8.8.8", 80))
            ip = sock.getsockname()[0]
            sock.close()
            return ip
        except Exception:
            return "127.0.0.1"


class SwarmDeploymentTool:
    """Tools for deploying and managing swarm instances"""
    
    def __init__(self):
        self.config_manager = ConfigManager()
        
    async def deploy_local_swarm(self, num_nodes: int = 3, config_preset: str = 'small') -> List[str]:
        """Deploy a local swarm for testing"""
        print(f"🚀 Deploying local swarm with {num_nodes} nodes...")
        
        # Get base configuration
        base_config = self.config_manager.get_preset_config(config_preset)
        
        node_processes = []
        node_ids = []
        
        for i in range(num_nodes):
            # Create unique configuration for each node
            node_config = SwarmNodeConfig(
                model_config=base_config.model,
                network_config=base_config.network,
                max_peers=num_nodes - 1
            )
            
            # Adjust port for each node
            node_config.network_config.port = 8080 + i
            node_config.network_config.discovery_port = 8090 + i
            
            # Create node
            node = SwarmNode(node_config)
            
            try:
                await node.start()
                
                status = await node.get_swarm_status()
                node_id = status['node_id']
                node_ids.append(node_id)
                
                print(f"  ✅ Node {i+1} started: {node_id[:8]} on port {8080 + i}")
                
                # Store reference (in real deployment, would manage processes)
                node_processes.append(node)
                
            except Exception as e:
                print(f"  ❌ Failed to start node {i+1}: {e}")
                
        print(f"🎉 Local swarm deployed with {len(node_ids)} nodes")
        
        # Let nodes discover each other
        await asyncio.sleep(5)
        
        # Check connectivity
        for i, node in enumerate(node_processes):
            peers = await node.discover_peers()
            print(f"  Node {i+1} discovered {len(peers)} peers")
            
        return node_ids
        
    async def generate_docker_compose(self, num_nodes: int = 5, output_file: str = "docker-compose.yml"):
        """Generate Docker Compose file for swarm deployment"""
        compose_config = {
            'version': '3.8',
            'services': {},
            'networks': {
                'swarm-network': {
                    'driver': 'bridge'
                }
            }
        }
        
        for i in range(num_nodes):
            service_name = f"swarm-node-{i+1}"
            
            compose_config['services'][service_name] = {
                'build': '.',
                'environment': [
                    f'NCRSH_PORT={8080 + i}',
                    f'NCRSH_DISCOVERY_PORT={8090 + i}',
                    f'NCRSH_MAX_PEERS={num_nodes - 1}',
                    f'NODE_ID=node-{i+1}'
                ],
                'ports': [
                    f'{8080 + i}:{8080 + i}',
                    f'{8090 + i}:{8090 + i}'
                ],
                'networks': ['swarm-network'],
                'volumes': [
                    './data:/app/data',
                    './logs:/app/logs'
                ],
                'restart': 'unless-stopped'
            }
            
        # Save to file
        import yaml
        with open(output_file, 'w') as f:
            yaml.dump(compose_config, f, default_flow_style=False)
            
        print(f"📄 Docker Compose file generated: {output_file}")
        
    def generate_kubernetes_manifests(self, num_nodes: int = 5, output_dir: str = "./k8s"):
        """Generate Kubernetes manifests for swarm deployment"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Namespace
        namespace_manifest = {
            'apiVersion': 'v1',
            'kind': 'Namespace',
            'metadata': {
                'name': 'ncrsh-swarm'
            }
        }
        
        # ConfigMap
        configmap_manifest = {
            'apiVersion': 'v1',
            'kind': 'ConfigMap',
            'metadata': {
                'name': 'swarm-config',
                'namespace': 'ncrsh-swarm'
            },
            'data': {
                'config.yaml': self._generate_k8s_config()
            }
        }
        
        # Service
        service_manifest = {
            'apiVersion': 'v1',
            'kind': 'Service',
            'metadata': {
                'name': 'swarm-service',
                'namespace': 'ncrsh-swarm'
            },
            'spec': {
                'type': 'ClusterIP',
                'ports': [
                    {'port': 8080, 'targetPort': 8080, 'name': 'swarm-port'},
                    {'port': 8081, 'targetPort': 8081, 'name': 'discovery-port'}
                ],
                'selector': {'app': 'ncrsh-swarm'}
            }
        }
        
        # Deployment
        deployment_manifest = {
            'apiVersion': 'apps/v1',
            'kind': 'Deployment',
            'metadata': {
                'name': 'swarm-deployment',
                'namespace': 'ncrsh-swarm'
            },
            'spec': {
                'replicas': num_nodes,
                'selector': {
                    'matchLabels': {'app': 'ncrsh-swarm'}
                },
                'template': {
                    'metadata': {
                        'labels': {'app': 'ncrsh-swarm'}
                    },
                    'spec': {
                        'containers': [{
                            'name': 'swarm-node',
                            'image': 'ncrsh-swarm:latest',
                            'ports': [
                                {'containerPort': 8080},
                                {'containerPort': 8081}
                            ],
                            'env': [
                                {'name': 'NCRSH_PORT', 'value': '8080'},
                                {'name': 'NCRSH_DISCOVERY_PORT', 'value': '8081'},
                                {'name': 'NCRSH_MAX_PEERS', 'value': str(num_nodes - 1)}
                            ],
                            'volumeMounts': [{
                                'name': 'config-volume',
                                'mountPath': '/app/config'
                            }],
                            'resources': {
                                'requests': {
                                    'memory': '512Mi',
                                    'cpu': '250m'
                                },
                                'limits': {
                                    'memory': '2Gi',
                                    'cpu': '1000m'
                                }
                            }
                        }],
                        'volumes': [{
                            'name': 'config-volume',
                            'configMap': {'name': 'swarm-config'}
                        }]
                    }
                }
            }
        }
        
        # Save manifests
        manifests = [
            ('namespace.yaml', namespace_manifest),
            ('configmap.yaml', configmap_manifest),
            ('service.yaml', service_manifest),
            ('deployment.yaml', deployment_manifest)
        ]
        
        import yaml
        for filename, manifest in manifests:
            file_path = output_path / filename
            with open(file_path, 'w') as f:
                yaml.dump(manifest, f, default_flow_style=False)
                
        print(f"📁 Kubernetes manifests generated in: {output_dir}")
        
    def _generate_k8s_config(self) -> str:
        """Generate Kubernetes configuration"""
        config = self.config_manager.get_preset_config('distributed')
        import yaml
        return yaml.dump(config.to_dict(), default_flow_style=False)


class SwarmDiagnosticTool:
    """Advanced diagnostics and troubleshooting tools"""
    
    async def run_diagnostics(self) -> Dict[str, Any]:
        """Run comprehensive diagnostics"""
        print("🔍 Running ncrsh-Swarm diagnostics...")
        
        diagnostics = {
            'timestamp': time.time(),
            'system_check': {},
            'network_test': {},
            'performance_test': {},
            'recommendations': []
        }
        
        # System check
        health_checker = SwarmHealthChecker()
        diagnostics['system_check'] = await health_checker.check_system_health()
        
        # Network connectivity test
        diagnostics['network_test'] = await self._test_network_connectivity()
        
        # Basic performance test
        diagnostics['performance_test'] = await self._run_performance_test()
        
        # Generate recommendations
        diagnostics['recommendations'] = self._generate_recommendations(diagnostics)
        
        return diagnostics
        
    async def _test_network_connectivity(self) -> Dict[str, Any]:
        """Test network connectivity between nodes"""
        test_results = {
            'local_connectivity': True,
            'peer_discovery': True,
            'message_passing': True,
            'latency_ms': 0.0,
            'throughput_mbps': 0.0
        }
        
        try:
            # Create test nodes
            config = SwarmNodeConfig()
            config.network_config.port = 9000
            config.network_config.discovery_port = 9001
            
            node1 = SwarmNode(config)
            await node1.start()
            
            config2 = SwarmNodeConfig()
            config2.network_config.port = 9002
            config2.network_config.discovery_port = 9003
            
            node2 = SwarmNode(config2)
            await node2.start()
            
            # Test peer discovery
            await asyncio.sleep(2)
            peers = await node1.discover_peers()
            test_results['peer_discovery'] = len(peers) > 0
            
            # Test message latency
            if peers:
                start_time = time.time()
                try:
                    await node1.network.send_message(peers[0], 'ping', {})
                    latency = (time.time() - start_time) * 1000
                    test_results['latency_ms'] = latency
                except Exception:
                    test_results['message_passing'] = False
                    
            # Cleanup
            await node1.stop()
            await node2.stop()
            
        except Exception as e:
            test_results['local_connectivity'] = False
            test_results['error'] = str(e)
            
        return test_results
        
    async def _run_performance_test(self) -> Dict[str, Any]:
        """Run basic performance test"""
        print("  📊 Running performance test...")
        
        try:
            # Create benchmarker
            benchmarker = SwarmBenchmarker()
            
            # Run quick scalability test
            results = await benchmarker.benchmark_scalability(max_nodes=3)
            
            return {
                'test_completed': True,
                'max_nodes_tested': results.get('analysis', {}).get('max_nodes_tested', 0),
                'scaling_efficiency': results.get('analysis', {}).get('avg_scaling_efficiency', 0),
                'peak_throughput': results.get('analysis', {}).get('peak_throughput', 0)
            }
            
        except Exception as e:
            return {
                'test_completed': False,
                'error': str(e)
            }
            
    def _generate_recommendations(self, diagnostics: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on diagnostics"""
        recommendations = []
        
        system_check = diagnostics.get('system_check', {})
        network_test = diagnostics.get('network_test', {})
        performance_test = diagnostics.get('performance_test', {})
        
        # System recommendations
        if not system_check.get('overall_status') == 'healthy':
            recommendations.append("Address system health issues before deploying swarm")
            
        resource_health = system_check.get('resource_health', {})
        if resource_health.get('memory_usage_percent', 0) > 80:
            recommendations.append("Consider increasing available memory")
            
        # Network recommendations
        if not network_test.get('peer_discovery', True):
            recommendations.append("Check firewall settings for peer discovery")
            
        latency = network_test.get('latency_ms', 0)
        if latency > 100:
            recommendations.append(f"High network latency ({latency:.1f}ms) - consider network optimization")
            
        # Performance recommendations
        if performance_test.get('test_completed'):
            efficiency = performance_test.get('scaling_efficiency', 0)
            if efficiency < 0.5:
                recommendations.append("Poor scaling efficiency - check resource allocation")
                
        if not recommendations:
            recommendations.append("System appears healthy - ready for swarm deployment")
            
        return recommendations


# CLI entry points for tools

async def health_command(args):
    """Health check command"""
    checker = SwarmHealthChecker()
    health = await checker.check_system_health()
    
    print(f"🏥 System Health Check")
    print(f"Overall Status: {'✅ Healthy' if health['overall_status'] == 'healthy' else '❌ Unhealthy'}")
    
    if health['issues']:
        print(f"\n⚠️  Issues Found:")
        for issue in health['issues']:
            print(f"  - {issue}")
            
    print(f"\n📊 Resource Usage:")
    resource_health = health['resource_health']
    print(f"  Memory: {resource_health['memory_usage_percent']:.1f}%")
    print(f"  CPU: {resource_health['cpu_usage_percent']:.1f}%")
    print(f"  Disk: {resource_health['disk_usage_percent']:.1f}%")


async def deploy_command(args):
    """Deployment command"""
    deployer = SwarmDeploymentTool()
    
    if args.local:
        node_ids = await deployer.deploy_local_swarm(args.nodes, args.preset)
        print(f"Local swarm deployed with nodes: {[nid[:8] for nid in node_ids]}")
        
    if args.docker:
        await deployer.generate_docker_compose(args.nodes, args.output)
        
    if args.kubernetes:
        deployer.generate_kubernetes_manifests(args.nodes, args.output)


async def diagnose_command(args):
    """Diagnostics command"""
    diagnostic_tool = SwarmDiagnosticTool()
    results = await diagnostic_tool.run_diagnostics()
    
    print(f"🔍 Diagnostic Results")
    print(f"System Health: {'✅' if results['system_check']['overall_status'] == 'healthy' else '❌'}")
    print(f"Network Test: {'✅' if results['network_test']['peer_discovery'] else '❌'}")
    print(f"Performance Test: {'✅' if results['performance_test']['test_completed'] else '❌'}")
    
    print(f"\n💡 Recommendations:")
    for rec in results['recommendations']:
        print(f"  - {rec}")


async def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(description="ncrsh-Swarm Tools")
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Health check command
    health_parser = subparsers.add_parser('health', help='Check system health')
    
    # Deployment commands
    deploy_parser = subparsers.add_parser('deploy', help='Deploy swarm instances')
    deploy_parser.add_argument('--local', action='store_true', help='Deploy local swarm')
    deploy_parser.add_argument('--docker', action='store_true', help='Generate Docker Compose')
    deploy_parser.add_argument('--kubernetes', action='store_true', help='Generate Kubernetes manifests')
    deploy_parser.add_argument('--nodes', type=int, default=3, help='Number of nodes')
    deploy_parser.add_argument('--preset', default='small', help='Configuration preset')
    deploy_parser.add_argument('--output', default='./deployment', help='Output directory')
    
    # Diagnostics command
    diagnose_parser = subparsers.add_parser('diagnose', help='Run system diagnostics')
    
    args = parser.parse_args()
    
    if args.command == 'health':
        await health_command(args)
    elif args.command == 'deploy':
        await deploy_command(args)
    elif args.command == 'diagnose':
        await diagnose_command(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    asyncio.run(main())