"""
ncrsh-Swarm Real-Time Dashboard
==============================

Web-based dashboard for monitoring and managing swarm networks with
real-time metrics, visualization, and control interfaces.
"""

import asyncio
import json
import time
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import socket

from aiohttp import web, WSMsgType
import aiohttp_cors
import jinja2
import aiofiles
from pathlib import Path


class SwarmDashboard:
    """
    Real-time web dashboard for ncrsh-Swarm monitoring
    
    Features:
    - Live network topology visualization
    - Real-time performance metrics
    - Node health monitoring
    - Training progress tracking
    - Interactive swarm control
    - WebSocket-based updates
    """
    
    def __init__(self, port: int = 8082, swarm_manager=None):
        self.port = port
        self.swarm_manager = swarm_manager
        self.app = web.Application()
        self.websockets: List[web.WebSocketResponse] = []
        
        # Dashboard state
        self.metrics_history: Dict[str, List] = {
            'throughput': [],
            'memory_usage': [],
            'network_activity': [],
            'node_count': [],
            'error_rate': []
        }
        
        self.node_registry: Dict[str, Dict] = {}
        self.alert_queue: List[Dict] = []
        
        # Setup routes and middleware
        self._setup_routes()
        self._setup_cors()
        self._setup_templates()
        
        # Start background tasks
        self.monitoring_task = None
        
    def _setup_routes(self):
        """Setup HTTP routes"""
        # Static file serving
        self.app.router.add_static('/static/', path='dashboard/static', name='static')
        
        # Main dashboard routes
        self.app.router.add_get('/', self.index_handler)
        self.app.router.add_get('/dashboard', self.dashboard_handler)
        self.app.router.add_get('/nodes', self.nodes_handler)
        self.app.router.add_get('/metrics', self.metrics_handler)
        self.app.router.add_get('/topology', self.topology_handler)
        self.app.router.add_get('/logs', self.logs_handler)
        
        # API endpoints
        self.app.router.add_get('/api/status', self.api_status)
        self.app.router.add_get('/api/nodes', self.api_nodes)
        self.app.router.add_get('/api/metrics', self.api_metrics)
        self.app.router.add_get('/api/topology', self.api_topology)
        self.app.router.add_post('/api/nodes/{node_id}/stop', self.api_stop_node)
        self.app.router.add_post('/api/nodes/{node_id}/start', self.api_start_node)
        self.app.router.add_post('/api/swarm/scale', self.api_scale_swarm)
        
        # WebSocket endpoint
        self.app.router.add_get('/ws', self.websocket_handler)
        
    def _setup_cors(self):
        """Setup CORS for API access"""
        cors = aiohttp_cors.setup(self.app, defaults={
            "*": aiohttp_cors.ResourceOptions(
                allow_credentials=True,
                expose_headers="*",
                allow_headers="*",
                allow_methods="*"
            )
        })
        
        # Add CORS to all routes
        for route in list(self.app.router.routes()):
            cors.add(route)
            
    def _setup_templates(self):
        """Setup Jinja2 templates"""
        template_dir = Path(__file__).parent / 'templates'
        self.jinja_env = jinja2.Environment(
            loader=jinja2.FileSystemLoader(str(template_dir)),
            autoescape=jinja2.select_autoescape(['html', 'xml'])
        )
        
    async def start(self):
        """Start the dashboard server"""
        self.monitoring_task = asyncio.create_task(self._monitoring_loop())
        
        runner = web.AppRunner(self.app)
        await runner.setup()
        
        site = web.TCPSite(runner, '0.0.0.0', self.port)
        await site.start()
        
        print(f"🎛️  Dashboard started at http://localhost:{self.port}")
        return runner
        
    async def stop(self):
        """Stop the dashboard server"""
        if self.monitoring_task:
            self.monitoring_task.cancel()
            
        # Close all websockets
        for ws in self.websockets:
            await ws.close()
            
    # Route handlers
    
    async def index_handler(self, request):
        """Main dashboard page"""
        template = self.jinja_env.get_template('index.html')
        
        # Get current swarm status
        swarm_status = await self._get_swarm_status()
        
        content = template.render(
            title="ncrsh-Swarm Dashboard",
            swarm_status=swarm_status,
            node_count=len(self.node_registry),
            active_alerts=len([a for a in self.alert_queue if a.get('active', True)])
        )
        
        return web.Response(text=content, content_type='text/html')
        
    async def dashboard_handler(self, request):
        """Real-time dashboard view"""
        template = self.jinja_env.get_template('dashboard.html')
        
        content = template.render(
            title="Real-Time Dashboard",
            websocket_url=f"ws://localhost:{self.port}/ws"
        )
        
        return web.Response(text=content, content_type='text/html')
        
    async def nodes_handler(self, request):
        """Node management page"""
        template = self.jinja_env.get_template('nodes.html')
        
        nodes_data = []
        for node_id, node_info in self.node_registry.items():
            nodes_data.append({
                'id': node_id,
                'status': node_info.get('status', 'unknown'),
                'address': node_info.get('address', 'N/A'),
                'peers': node_info.get('peer_count', 0),
                'uptime': node_info.get('uptime', 0),
                'memory_usage': node_info.get('memory_usage', 0),
                'last_seen': node_info.get('last_seen', 'Never')
            })
            
        content = template.render(
            title="Node Management",
            nodes=nodes_data
        )
        
        return web.Response(text=content, content_type='text/html')
        
    async def metrics_handler(self, request):
        """Metrics and analytics page"""
        template = self.jinja_env.get_template('metrics.html')
        
        # Calculate summary statistics
        metrics_summary = {
            'avg_throughput': self._calculate_average('throughput'),
            'peak_memory': self._calculate_peak('memory_usage'),
            'total_network_traffic': self._calculate_total('network_activity'),
            'uptime_percentage': self._calculate_uptime_percentage()
        }
        
        content = template.render(
            title="Metrics & Analytics",
            metrics_summary=metrics_summary
        )
        
        return web.Response(text=content, content_type='text/html')
        
    async def topology_handler(self, request):
        """Network topology visualization"""
        template = self.jinja_env.get_template('topology.html')
        
        # Generate topology data
        topology_data = await self._generate_topology_data()
        
        content = template.render(
            title="Network Topology",
            topology_data=json.dumps(topology_data)
        )
        
        return web.Response(text=content, content_type='text/html')
        
    async def logs_handler(self, request):
        """Log viewer page"""
        template = self.jinja_env.get_template('logs.html')
        
        # Get recent logs
        logs = await self._get_recent_logs(limit=100)
        
        content = template.render(
            title="System Logs",
            logs=logs
        )
        
        return web.Response(text=content, content_type='text/html')
        
    # API endpoints
    
    async def api_status(self, request):
        """Get overall swarm status"""
        status = await self._get_swarm_status()
        return web.json_response(status)
        
    async def api_nodes(self, request):
        """Get node information"""
        return web.json_response({
            'nodes': self.node_registry,
            'count': len(self.node_registry)
        })
        
    async def api_metrics(self, request):
        """Get metrics data"""
        # Get time range from query parameters
        hours = int(request.query.get('hours', 1))
        
        # Filter metrics by time range
        cutoff_time = time.time() - (hours * 3600)
        
        filtered_metrics = {}
        for metric_name, values in self.metrics_history.items():
            filtered_values = [
                v for v in values 
                if v.get('timestamp', 0) > cutoff_time
            ]
            filtered_metrics[metric_name] = filtered_values
            
        return web.json_response({
            'metrics': filtered_metrics,
            'time_range_hours': hours,
            'data_points': sum(len(v) for v in filtered_metrics.values())
        })
        
    async def api_topology(self, request):
        """Get network topology data"""
        topology = await self._generate_topology_data()
        return web.json_response(topology)
        
    async def api_stop_node(self, request):
        """Stop a specific node"""
        node_id = request.match_info['node_id']
        
        if self.swarm_manager:
            success = await self.swarm_manager.stop_node(node_id)
            return web.json_response({
                'success': success,
                'message': f"Node {node_id} stop command sent"
            })
        else:
            return web.json_response({
                'success': False,
                'message': "Swarm manager not available"
            }, status=503)
            
    async def api_start_node(self, request):
        """Start a specific node"""
        node_id = request.match_info['node_id']
        
        if self.swarm_manager:
            success = await self.swarm_manager.start_node(node_id)
            return web.json_response({
                'success': success,
                'message': f"Node {node_id} start command sent"
            })
        else:
            return web.json_response({
                'success': False,
                'message': "Swarm manager not available"
            }, status=503)
            
    async def api_scale_swarm(self, request):
        """Scale the swarm up or down"""
        data = await request.json()
        target_nodes = data.get('target_nodes', 0)
        
        if self.swarm_manager:
            result = await self.swarm_manager.scale_to(target_nodes)
            return web.json_response(result)
        else:
            return web.json_response({
                'success': False,
                'message': "Swarm manager not available"
            }, status=503)
            
    # WebSocket handler
    
    async def websocket_handler(self, request):
        """Handle WebSocket connections for real-time updates"""
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        
        self.websockets.append(ws)
        
        try:
            # Send initial data
            await ws.send_str(json.dumps({
                'type': 'initial',
                'data': {
                    'nodes': self.node_registry,
                    'metrics': self.metrics_history,
                    'alerts': self.alert_queue
                }
            }))
            
            # Handle incoming messages
            async for msg in ws:
                if msg.type == WSMsgType.TEXT:
                    try:
                        data = json.loads(msg.data)
                        await self._handle_websocket_message(ws, data)
                    except json.JSONDecodeError:
                        await ws.send_str(json.dumps({
                            'type': 'error',
                            'message': 'Invalid JSON'
                        }))
                elif msg.type == WSMsgType.ERROR:
                    break
                    
        except Exception as e:
            logging.error(f"WebSocket error: {e}")
        finally:
            if ws in self.websockets:
                self.websockets.remove(ws)
                
        return ws
        
    async def _handle_websocket_message(self, ws: web.WebSocketResponse, data: Dict):
        """Handle incoming WebSocket messages"""
        msg_type = data.get('type')
        
        if msg_type == 'ping':
            await ws.send_str(json.dumps({'type': 'pong'}))
        elif msg_type == 'subscribe':
            # Subscribe to specific metrics
            metrics = data.get('metrics', [])
            await ws.send_str(json.dumps({
                'type': 'subscription_confirmed',
                'metrics': metrics
            }))
        elif msg_type == 'request_update':
            # Send current state
            await ws.send_str(json.dumps({
                'type': 'update',
                'data': {
                    'nodes': self.node_registry,
                    'timestamp': time.time()
                }
            }))
            
    # Background monitoring
    
    async def _monitoring_loop(self):
        """Background monitoring loop"""
        while True:
            try:
                # Collect metrics
                await self._collect_metrics()
                
                # Check for alerts
                await self._check_alerts()
                
                # Broadcast updates to connected clients
                await self._broadcast_updates()
                
                await asyncio.sleep(5)  # Update every 5 seconds
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logging.error(f"Monitoring loop error: {e}")
                await asyncio.sleep(10)
                
    async def _collect_metrics(self):
        """Collect current metrics"""
        timestamp = time.time()
        
        # Simulate metric collection (would integrate with actual swarm)
        if self.swarm_manager:
            # Get real metrics from swarm manager
            swarm_metrics = await self.swarm_manager.get_metrics()
            
            self.metrics_history['throughput'].append({
                'timestamp': timestamp,
                'value': swarm_metrics.get('throughput', 0)
            })
            
            self.metrics_history['memory_usage'].append({
                'timestamp': timestamp,
                'value': swarm_metrics.get('memory_usage', 0)
            })
            
            self.metrics_history['network_activity'].append({
                'timestamp': timestamp,
                'value': swarm_metrics.get('network_activity', 0)
            })
            
            self.metrics_history['node_count'].append({
                'timestamp': timestamp,
                'value': swarm_metrics.get('node_count', 0)
            })
            
            self.metrics_history['error_rate'].append({
                'timestamp': timestamp,
                'value': swarm_metrics.get('error_rate', 0)
            })
            
        else:
            # Simulate metrics for demo
            import random
            
            self.metrics_history['throughput'].append({
                'timestamp': timestamp,
                'value': random.uniform(50, 200)
            })
            
            self.metrics_history['memory_usage'].append({
                'timestamp': timestamp,
                'value': random.uniform(200, 800)
            })
            
        # Trim old data (keep last hour)
        cutoff = timestamp - 3600
        for metric_name in self.metrics_history:
            self.metrics_history[metric_name] = [
                m for m in self.metrics_history[metric_name]
                if m['timestamp'] > cutoff
            ]
            
    async def _check_alerts(self):
        """Check for alert conditions"""
        timestamp = time.time()
        
        # Check memory usage
        recent_memory = self.metrics_history.get('memory_usage', [])
        if recent_memory:
            latest_memory = recent_memory[-1]['value']
            if latest_memory > 700:  # MB
                alert = {
                    'id': f"memory_alert_{int(timestamp)}",
                    'type': 'warning',
                    'title': 'High Memory Usage',
                    'message': f'Memory usage is {latest_memory:.0f}MB',
                    'timestamp': timestamp,
                    'active': True
                }
                
                # Avoid duplicate alerts
                if not any(a.get('type') == 'warning' and 'Memory' in a.get('title', '') 
                          for a in self.alert_queue if a.get('active')):
                    self.alert_queue.append(alert)
                    
        # Check node connectivity
        if len(self.node_registry) == 0:
            alert = {
                'id': f"connectivity_alert_{int(timestamp)}",
                'type': 'error',
                'title': 'No Nodes Connected',
                'message': 'No swarm nodes are currently connected',
                'timestamp': timestamp,
                'active': True
            }
            
            if not any(a.get('type') == 'error' and 'Nodes' in a.get('title', '')
                      for a in self.alert_queue if a.get('active')):
                self.alert_queue.append(alert)
                
        # Clean up old alerts
        self.alert_queue = [
            alert for alert in self.alert_queue
            if timestamp - alert.get('timestamp', 0) < 3600  # Keep for 1 hour
        ]
        
    async def _broadcast_updates(self):
        """Broadcast updates to connected WebSocket clients"""
        if not self.websockets:
            return
            
        update_data = {
            'type': 'metrics_update',
            'data': {
                'metrics': {
                    k: v[-10:] for k, v in self.metrics_history.items()  # Last 10 points
                },
                'nodes': len(self.node_registry),
                'alerts': [a for a in self.alert_queue if a.get('active')],
                'timestamp': time.time()
            }
        }
        
        # Send to all connected clients
        disconnected = []
        for ws in self.websockets:
            try:
                await ws.send_str(json.dumps(update_data))
            except Exception as e:
                logging.error(f"Failed to send WebSocket update: {e}")
                disconnected.append(ws)
                
        # Remove disconnected clients
        for ws in disconnected:
            if ws in self.websockets:
                self.websockets.remove(ws)
                
    # Helper methods
    
    async def _get_swarm_status(self):
        """Get overall swarm status"""
        if self.swarm_manager:
            return await self.swarm_manager.get_status()
        else:
            return {
                'status': 'demo_mode',
                'nodes': len(self.node_registry),
                'uptime': time.time() - getattr(self, 'start_time', time.time()),
                'version': '0.1.0'
            }
            
    async def _generate_topology_data(self):
        """Generate network topology visualization data"""
        nodes = []
        links = []
        
        # Add nodes
        for node_id, node_info in self.node_registry.items():
            nodes.append({
                'id': node_id,
                'name': node_id[:8],
                'status': node_info.get('status', 'unknown'),
                'peers': node_info.get('peer_count', 0),
                'address': node_info.get('address', ''),
                'group': 1 if node_info.get('status') == 'active' else 2
            })
            
        # Add connections (simplified)
        for i, node1 in enumerate(nodes):
            for j, node2 in enumerate(nodes):
                if i < j and node1['peers'] > 0 and node2['peers'] > 0:
                    links.append({
                        'source': node1['id'],
                        'target': node2['id'],
                        'strength': min(node1['peers'], node2['peers'])
                    })
                    
        return {
            'nodes': nodes,
            'links': links
        }
        
    async def _get_recent_logs(self, limit: int = 100):
        """Get recent log entries"""
        # This would integrate with actual logging system
        logs = []
        
        for i in range(limit):
            logs.append({
                'timestamp': time.time() - (i * 10),
                'level': 'INFO',
                'component': 'swarm_node',
                'message': f'Sample log message {i}'
            })
            
        return logs[:limit]
        
    def _calculate_average(self, metric_name: str) -> float:
        """Calculate average value for a metric"""
        values = self.metrics_history.get(metric_name, [])
        if not values:
            return 0.0
            
        return sum(v['value'] for v in values) / len(values)
        
    def _calculate_peak(self, metric_name: str) -> float:
        """Calculate peak value for a metric"""
        values = self.metrics_history.get(metric_name, [])
        if not values:
            return 0.0
            
        return max(v['value'] for v in values)
        
    def _calculate_total(self, metric_name: str) -> float:
        """Calculate total value for a metric"""
        values = self.metrics_history.get(metric_name, [])
        return sum(v['value'] for v in values)
        
    def _calculate_uptime_percentage(self) -> float:
        """Calculate overall uptime percentage"""
        # Simplified calculation
        return 99.5  # Would calculate from actual downtime data
        
    # Node registration methods
    
    def register_node(self, node_id: str, node_info: Dict):
        """Register a node with the dashboard"""
        self.node_registry[node_id] = {
            **node_info,
            'registered_at': time.time(),
            'last_seen': time.time()
        }
        
    def unregister_node(self, node_id: str):
        """Unregister a node from the dashboard"""
        if node_id in self.node_registry:
            del self.node_registry[node_id]
            
    def update_node_status(self, node_id: str, status_update: Dict):
        """Update node status"""
        if node_id in self.node_registry:
            self.node_registry[node_id].update(status_update)
            self.node_registry[node_id]['last_seen'] = time.time()


# CLI interface for dashboard
async def main():
    """Start dashboard as standalone application"""
    import argparse
    
    parser = argparse.ArgumentParser(description="ncrsh-Swarm Dashboard")
    parser.add_argument('--port', type=int, default=8082, help='Dashboard port')
    parser.add_argument('--host', default='0.0.0.0', help='Host to bind to')
    
    args = parser.parse_args()
    
    dashboard = SwarmDashboard(port=args.port)
    dashboard.start_time = time.time()  # Track start time for demo
    
    try:
        runner = await dashboard.start()
        print(f"🎛️  Dashboard running at http://{args.host}:{args.port}")
        print("Press Ctrl+C to stop")
        
        # Keep running
        while True:
            await asyncio.sleep(1)
            
    except KeyboardInterrupt:
        print("\n🛑 Stopping dashboard...")
        await dashboard.stop()
        await runner.cleanup()
        
    except Exception as e:
        print(f"❌ Dashboard error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())