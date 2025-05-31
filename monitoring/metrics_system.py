"""
ncrsh-Swarm Monitoring System
============================

Advanced monitoring, metrics collection, and alerting system for
distributed swarm networks with real-time analytics and visualization.
"""

import asyncio
import time
import json
import logging
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, asdict
from pathlib import Path
import psutil
import sqlite3
from collections import deque, defaultdict
import statistics
import threading
from datetime import datetime, timedelta


@dataclass
class MetricPoint:
    """Single metric data point"""
    timestamp: float
    node_id: str
    metric_name: str
    value: float
    tags: Dict[str, str] = None
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = {}


@dataclass
class Alert:
    """System alert definition"""
    id: str
    name: str
    description: str
    severity: str  # 'critical', 'warning', 'info'
    condition: str  # e.g., 'cpu_usage > 90'
    threshold: float
    metric_name: str
    node_id: Optional[str] = None
    active: bool = True
    triggered_at: Optional[float] = None
    resolved_at: Optional[float] = None


class MetricsCollector:
    """
    Collects metrics from swarm nodes and system resources
    
    Features:
    - Real-time metric collection
    - Multi-node aggregation
    - Historical data storage
    - Performance analytics
    """
    
    def __init__(self, storage_path: str = "./metrics.db"):
        self.storage_path = storage_path
        self.metrics_buffer: deque = deque(maxlen=10000)
        self.node_metrics: Dict[str, Dict[str, deque]] = defaultdict(lambda: defaultdict(lambda: deque(maxlen=1000)))
        
        # Initialize database
        self._init_database()
        
        # Background tasks
        self.collection_task: Optional[asyncio.Task] = None
        self.is_collecting = False
        
    def _init_database(self):
        """Initialize SQLite database for metrics storage"""
        conn = sqlite3.connect(self.storage_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL,
                node_id TEXT,
                metric_name TEXT,
                value REAL,
                tags TEXT
            )
        ''')
        
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_timestamp ON metrics(timestamp)
        ''')
        
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_node_metric ON metrics(node_id, metric_name)
        ''')
        
        conn.commit()
        conn.close()
        
    async def start_collection(self, interval: float = 5.0):
        """Start automatic metrics collection"""
        if self.is_collecting:
            return
            
        self.is_collecting = True
        self.collection_task = asyncio.create_task(self._collection_loop(interval))
        
    async def stop_collection(self):
        """Stop metrics collection"""
        self.is_collecting = False
        
        if self.collection_task:
            self.collection_task.cancel()
            try:
                await self.collection_task
            except asyncio.CancelledError:
                pass
                
    async def _collection_loop(self, interval: float):
        """Background metrics collection loop"""
        while self.is_collecting:
            try:
                # Collect system metrics
                await self._collect_system_metrics()
                
                # Persist metrics to database
                await self._persist_metrics()
                
                await asyncio.sleep(interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logging.error(f"Metrics collection error: {e}")
                await asyncio.sleep(1)
                
    async def _collect_system_metrics(self):
        """Collect system-level metrics"""
        timestamp = time.time()
        
        # CPU metrics
        cpu_percent = psutil.cpu_percent(interval=None)
        self.record_metric('system_cpu_usage', cpu_percent, timestamp=timestamp, node_id='system')
        
        # Memory metrics
        memory = psutil.virtual_memory()
        self.record_metric('system_memory_usage', memory.percent, timestamp=timestamp, node_id='system')
        self.record_metric('system_memory_available', memory.available / (1024**3), timestamp=timestamp, node_id='system')
        
        # Disk metrics
        disk = psutil.disk_usage('/')
        disk_percent = (disk.used / disk.total) * 100
        self.record_metric('system_disk_usage', disk_percent, timestamp=timestamp, node_id='system')
        self.record_metric('system_disk_free', disk.free / (1024**3), timestamp=timestamp, node_id='system')
        
        # Network metrics
        network = psutil.net_io_counters()
        self.record_metric('system_network_bytes_sent', network.bytes_sent, timestamp=timestamp, node_id='system')
        self.record_metric('system_network_bytes_recv', network.bytes_recv, timestamp=timestamp, node_id='system')
        
    def record_metric(self, metric_name: str, value: float, node_id: str = 'default', 
                     timestamp: Optional[float] = None, tags: Optional[Dict[str, str]] = None):
        """Record a metric value"""
        if timestamp is None:
            timestamp = time.time()
            
        metric_point = MetricPoint(
            timestamp=timestamp,
            node_id=node_id,
            metric_name=metric_name,
            value=value,
            tags=tags or {}
        )
        
        # Add to buffer
        self.metrics_buffer.append(metric_point)
        
        # Add to node-specific metrics
        self.node_metrics[node_id][metric_name].append((timestamp, value))
        
    async def _persist_metrics(self):
        """Persist buffered metrics to database"""
        if not self.metrics_buffer:
            return
            
        # Extract metrics to persist
        metrics_to_persist = []
        while self.metrics_buffer and len(metrics_to_persist) < 1000:
            metrics_to_persist.append(self.metrics_buffer.popleft())
            
        if not metrics_to_persist:
            return
            
        # Insert into database
        def insert_metrics():
            conn = sqlite3.connect(self.storage_path)
            cursor = conn.cursor()
            
            data = [
                (m.timestamp, m.node_id, m.metric_name, m.value, json.dumps(m.tags))
                for m in metrics_to_persist
            ]
            
            cursor.executemany(
                'INSERT INTO metrics (timestamp, node_id, metric_name, value, tags) VALUES (?, ?, ?, ?, ?)',
                data
            )
            
            conn.commit()
            conn.close()
            
        # Run in thread to avoid blocking
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, insert_metrics)
        
    def get_metric_history(self, metric_name: str, node_id: str = None, 
                          hours: int = 1) -> List[MetricPoint]:
        """Get historical metric data"""
        cutoff_time = time.time() - (hours * 3600)
        
        conn = sqlite3.connect(self.storage_path)
        cursor = conn.cursor()
        
        if node_id:
            cursor.execute('''
                SELECT timestamp, node_id, metric_name, value, tags
                FROM metrics
                WHERE metric_name = ? AND node_id = ? AND timestamp > ?
                ORDER BY timestamp
            ''', (metric_name, node_id, cutoff_time))
        else:
            cursor.execute('''
                SELECT timestamp, node_id, metric_name, value, tags
                FROM metrics
                WHERE metric_name = ? AND timestamp > ?
                ORDER BY timestamp
            ''', (metric_name, cutoff_time))
            
        rows = cursor.fetchall()
        conn.close()
        
        return [
            MetricPoint(
                timestamp=row[0],
                node_id=row[1],
                metric_name=row[2],
                value=row[3],
                tags=json.loads(row[4]) if row[4] else {}
            )
            for row in rows
        ]
        
    def get_latest_metrics(self, node_id: str = None) -> Dict[str, float]:
        """Get latest metric values"""
        if node_id and node_id in self.node_metrics:
            return {
                metric_name: history[-1][1] if history else 0.0
                for metric_name, history in self.node_metrics[node_id].items()
            }
        else:
            # Return system metrics
            return {
                metric_name: history[-1][1] if history else 0.0
                for metric_name, history in self.node_metrics['system'].items()
            }
            
    def get_metric_statistics(self, metric_name: str, node_id: str = None, 
                            hours: int = 1) -> Dict[str, float]:
        """Get statistical summary of a metric"""
        history = self.get_metric_history(metric_name, node_id, hours)
        
        if not history:
            return {}
            
        values = [point.value for point in history]
        
        return {
            'min': min(values),
            'max': max(values),
            'mean': statistics.mean(values),
            'median': statistics.median(values),
            'stdev': statistics.stdev(values) if len(values) > 1 else 0.0,
            'count': len(values)
        }


class AlertManager:
    """
    Alert management system for monitoring swarm health
    
    Features:
    - Configurable alert rules
    - Multi-severity alerting
    - Alert aggregation and deduplication
    - Notification channels
    """
    
    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics_collector = metrics_collector
        self.alerts: Dict[str, Alert] = {}
        self.alert_history: List[Alert] = []
        self.notification_handlers: List[Callable] = []
        
        # Background alerting
        self.alerting_task: Optional[asyncio.Task] = None
        self.is_alerting = False
        
    def add_alert_rule(self, alert: Alert):
        """Add an alert rule"""
        self.alerts[alert.id] = alert
        
    def remove_alert_rule(self, alert_id: str):
        """Remove an alert rule"""
        if alert_id in self.alerts:
            del self.alerts[alert_id]
            
    def add_notification_handler(self, handler: Callable):
        """Add a notification handler function"""
        self.notification_handlers.append(handler)
        
    async def start_alerting(self, check_interval: float = 30.0):
        """Start alert monitoring"""
        if self.is_alerting:
            return
            
        self.is_alerting = True
        self.alerting_task = asyncio.create_task(self._alerting_loop(check_interval))
        
    async def stop_alerting(self):
        """Stop alert monitoring"""
        self.is_alerting = False
        
        if self.alerting_task:
            self.alerting_task.cancel()
            try:
                await self.alerting_task
            except asyncio.CancelledError:
                pass
                
    async def _alerting_loop(self, interval: float):
        """Background alert checking loop"""
        while self.is_alerting:
            try:
                await self._check_alerts()
                await asyncio.sleep(interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logging.error(f"Alert checking error: {e}")
                await asyncio.sleep(5)
                
    async def _check_alerts(self):
        """Check all alert conditions"""
        for alert in self.alerts.values():
            if not alert.active:
                continue
                
            try:
                # Get latest metric value
                latest_metrics = self.metrics_collector.get_latest_metrics(alert.node_id)
                current_value = latest_metrics.get(alert.metric_name)
                
                if current_value is None:
                    continue
                    
                # Evaluate condition
                triggered = self._evaluate_condition(alert.condition, current_value, alert.threshold)
                
                # Handle alert state changes
                if triggered and alert.triggered_at is None:
                    # New alert triggered
                    await self._trigger_alert(alert, current_value)
                elif not triggered and alert.triggered_at is not None:
                    # Alert resolved
                    await self._resolve_alert(alert, current_value)
                    
            except Exception as e:
                logging.error(f"Error checking alert {alert.id}: {e}")
                
    def _evaluate_condition(self, condition: str, value: float, threshold: float) -> bool:
        """Evaluate alert condition"""
        # Simple condition evaluation
        if '>' in condition:
            return value > threshold
        elif '<' in condition:
            return value < threshold
        elif '>=' in condition:
            return value >= threshold
        elif '<=' in condition:
            return value <= threshold
        elif '==' in condition:
            return abs(value - threshold) < 1e-6
        else:
            return False
            
    async def _trigger_alert(self, alert: Alert, current_value: float):
        """Trigger an alert"""
        alert.triggered_at = time.time()
        
        # Add to history
        alert_copy = Alert(**asdict(alert))
        self.alert_history.append(alert_copy)
        
        # Send notifications
        await self._send_notification(alert, 'triggered', current_value)
        
        logging.warning(f"ALERT TRIGGERED: {alert.name} - {alert.description} (value: {current_value})")
        
    async def _resolve_alert(self, alert: Alert, current_value: float):
        """Resolve an alert"""
        alert.resolved_at = time.time()
        
        # Send notifications
        await self._send_notification(alert, 'resolved', current_value)
        
        logging.info(f"ALERT RESOLVED: {alert.name} (value: {current_value})")
        
        # Reset for next trigger
        alert.triggered_at = None
        alert.resolved_at = None
        
    async def _send_notification(self, alert: Alert, action: str, value: float):
        """Send alert notifications"""
        notification_data = {
            'alert': asdict(alert),
            'action': action,
            'current_value': value,
            'timestamp': time.time()
        }
        
        for handler in self.notification_handlers:
            try:
                await handler(notification_data)
            except Exception as e:
                logging.error(f"Notification handler error: {e}")
                
    def get_active_alerts(self) -> List[Alert]:
        """Get currently active alerts"""
        return [alert for alert in self.alerts.values() if alert.triggered_at is not None]
        
    def get_alert_history(self, hours: int = 24) -> List[Alert]:
        """Get alert history"""
        cutoff_time = time.time() - (hours * 3600)
        
        return [
            alert for alert in self.alert_history
            if alert.triggered_at and alert.triggered_at > cutoff_time
        ]


class PerformanceAnalyzer:
    """
    Performance analysis and optimization recommendations
    
    Features:
    - Performance trend analysis
    - Bottleneck detection
    - Resource optimization recommendations
    - Capacity planning
    """
    
    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics_collector = metrics_collector
        
    async def analyze_performance(self, hours: int = 24) -> Dict[str, Any]:
        """Comprehensive performance analysis"""
        analysis = {
            'timestamp': time.time(),
            'time_period_hours': hours,
            'system_performance': {},
            'node_performance': {},
            'bottlenecks': [],
            'recommendations': [],
            'trends': {}
        }
        
        # Analyze system performance
        analysis['system_performance'] = await self._analyze_system_performance(hours)
        
        # Detect bottlenecks
        analysis['bottlenecks'] = await self._detect_bottlenecks(hours)
        
        # Generate recommendations
        analysis['recommendations'] = await self._generate_recommendations(analysis)
        
        # Analyze trends
        analysis['trends'] = await self._analyze_trends(hours)
        
        return analysis
        
    async def _analyze_system_performance(self, hours: int) -> Dict[str, Any]:
        """Analyze overall system performance"""
        performance = {}
        
        # CPU analysis
        cpu_stats = self.metrics_collector.get_metric_statistics('system_cpu_usage', 'system', hours)
        performance['cpu'] = {
            'average_usage': cpu_stats.get('mean', 0),
            'peak_usage': cpu_stats.get('max', 0),
            'utilization_score': min(100, cpu_stats.get('mean', 0) / 80 * 100)  # 80% is high
        }
        
        # Memory analysis
        memory_stats = self.metrics_collector.get_metric_statistics('system_memory_usage', 'system', hours)
        performance['memory'] = {
            'average_usage': memory_stats.get('mean', 0),
            'peak_usage': memory_stats.get('max', 0),
            'utilization_score': min(100, memory_stats.get('mean', 0) / 85 * 100)  # 85% is high
        }
        
        # Disk analysis
        disk_stats = self.metrics_collector.get_metric_statistics('system_disk_usage', 'system', hours)
        performance['disk'] = {
            'average_usage': disk_stats.get('mean', 0),
            'peak_usage': disk_stats.get('max', 0),
            'utilization_score': min(100, disk_stats.get('mean', 0) / 90 * 100)  # 90% is high
        }
        
        return performance
        
    async def _detect_bottlenecks(self, hours: int) -> List[Dict[str, Any]]:
        """Detect performance bottlenecks"""
        bottlenecks = []
        
        # CPU bottleneck
        cpu_stats = self.metrics_collector.get_metric_statistics('system_cpu_usage', 'system', hours)
        if cpu_stats.get('mean', 0) > 80:
            bottlenecks.append({
                'type': 'cpu',
                'severity': 'high' if cpu_stats['mean'] > 95 else 'medium',
                'description': f"High CPU usage: {cpu_stats['mean']:.1f}% average",
                'metric_value': cpu_stats['mean']
            })
            
        # Memory bottleneck
        memory_stats = self.metrics_collector.get_metric_statistics('system_memory_usage', 'system', hours)
        if memory_stats.get('mean', 0) > 85:
            bottlenecks.append({
                'type': 'memory',
                'severity': 'high' if memory_stats['mean'] > 95 else 'medium',
                'description': f"High memory usage: {memory_stats['mean']:.1f}% average",
                'metric_value': memory_stats['mean']
            })
            
        # Disk bottleneck
        disk_stats = self.metrics_collector.get_metric_statistics('system_disk_usage', 'system', hours)
        if disk_stats.get('mean', 0) > 90:
            bottlenecks.append({
                'type': 'disk',
                'severity': 'high' if disk_stats['mean'] > 98 else 'medium',
                'description': f"High disk usage: {disk_stats['mean']:.1f}% average",
                'metric_value': disk_stats['mean']
            })
            
        return bottlenecks
        
    async def _generate_recommendations(self, analysis: Dict[str, Any]) -> List[str]:
        """Generate optimization recommendations"""
        recommendations = []
        
        # CPU recommendations
        cpu_performance = analysis['system_performance'].get('cpu', {})
        if cpu_performance.get('average_usage', 0) > 80:
            recommendations.append("Consider scaling out to more nodes to reduce CPU load")
            recommendations.append("Enable gradient checkpointing to reduce computation")
            
        # Memory recommendations
        memory_performance = analysis['system_performance'].get('memory', {})
        if memory_performance.get('average_usage', 0) > 85:
            recommendations.append("Increase available memory or use memory-efficient training")
            recommendations.append("Enable mixed precision training to reduce memory usage")
            
        # Bottleneck-specific recommendations
        for bottleneck in analysis.get('bottlenecks', []):
            if bottleneck['type'] == 'cpu' and bottleneck['severity'] == 'high':
                recommendations.append("Critical: Immediate CPU scaling required")
            elif bottleneck['type'] == 'memory' and bottleneck['severity'] == 'high':
                recommendations.append("Critical: Memory upgrade or optimization needed")
            elif bottleneck['type'] == 'disk' and bottleneck['severity'] == 'high':
                recommendations.append("Critical: Disk cleanup or expansion required")
                
        if not recommendations:
            recommendations.append("System performance is optimal")
            
        return recommendations
        
    async def _analyze_trends(self, hours: int) -> Dict[str, Any]:
        """Analyze performance trends"""
        trends = {}
        
        # Get recent data for trend analysis
        for metric_name in ['system_cpu_usage', 'system_memory_usage', 'system_disk_usage']:
            history = self.metrics_collector.get_metric_history(metric_name, 'system', hours)
            
            if len(history) < 2:
                continue
                
            # Calculate trend (simple linear regression)
            values = [point.value for point in history]
            timestamps = [point.timestamp for point in history]
            
            # Normalize timestamps
            start_time = timestamps[0]
            normalized_times = [(t - start_time) for t in timestamps]
            
            if len(values) > 1:
                # Simple trend calculation
                recent_avg = statistics.mean(values[-len(values)//4:])  # Last quarter
                early_avg = statistics.mean(values[:len(values)//4])    # First quarter
                
                trend_direction = 'increasing' if recent_avg > early_avg else 'decreasing'
                trend_magnitude = abs(recent_avg - early_avg)
                
                trends[metric_name] = {
                    'direction': trend_direction,
                    'magnitude': trend_magnitude,
                    'recent_average': recent_avg,
                    'early_average': early_avg
                }
                
        return trends


# Default alert configurations
def create_default_alerts() -> List[Alert]:
    """Create default alert configurations"""
    return [
        Alert(
            id='high_cpu_usage',
            name='High CPU Usage',
            description='CPU usage is above 90%',
            severity='warning',
            condition='cpu_usage > 90',
            threshold=90.0,
            metric_name='system_cpu_usage'
        ),
        Alert(
            id='critical_cpu_usage',
            name='Critical CPU Usage',
            description='CPU usage is above 95%',
            severity='critical',
            condition='cpu_usage > 95',
            threshold=95.0,
            metric_name='system_cpu_usage'
        ),
        Alert(
            id='high_memory_usage',
            name='High Memory Usage',
            description='Memory usage is above 85%',
            severity='warning',
            condition='memory_usage > 85',
            threshold=85.0,
            metric_name='system_memory_usage'
        ),
        Alert(
            id='critical_memory_usage',
            name='Critical Memory Usage',
            description='Memory usage is above 95%',
            severity='critical',
            condition='memory_usage > 95',
            threshold=95.0,
            metric_name='system_memory_usage'
        ),
        Alert(
            id='high_disk_usage',
            name='High Disk Usage',
            description='Disk usage is above 90%',
            severity='warning',
            condition='disk_usage > 90',
            threshold=90.0,
            metric_name='system_disk_usage'
        )
    ]


# Example usage and testing
async def main():
    """Example monitoring system usage"""
    print("📊 ncrsh-Swarm Monitoring System")
    
    # Create metrics collector
    collector = MetricsCollector()
    
    # Create alert manager
    alert_manager = AlertManager(collector)
    
    # Add default alerts
    for alert in create_default_alerts():
        alert_manager.add_alert_rule(alert)
        
    # Add notification handler
    async def log_notification(notification_data):
        alert_info = notification_data['alert']
        action = notification_data['action']
        value = notification_data['current_value']
        
        print(f"🚨 Alert {action}: {alert_info['name']} (value: {value:.1f})")
        
    alert_manager.add_notification_handler(log_notification)
    
    # Start monitoring
    await collector.start_collection(interval=2.0)
    await alert_manager.start_alerting(check_interval=5.0)
    
    print("🔍 Monitoring started - collecting metrics...")
    
    try:
        # Run for a short time
        await asyncio.sleep(30)
        
        # Get some metrics
        print("\n📈 Latest Metrics:")
        latest = collector.get_latest_metrics('system')
        for metric, value in latest.items():
            print(f"  {metric}: {value:.1f}")
            
        # Performance analysis
        analyzer = PerformanceAnalyzer(collector)
        analysis = await analyzer.analyze_performance(hours=1)
        
        print(f"\n🔍 Performance Analysis:")
        print(f"  CPU Score: {analysis['system_performance']['cpu']['utilization_score']:.1f}/100")
        print(f"  Memory Score: {analysis['system_performance']['memory']['utilization_score']:.1f}/100")
        print(f"  Recommendations: {len(analysis['recommendations'])}")
        
        for rec in analysis['recommendations'][:3]:
            print(f"    - {rec}")
            
    finally:
        await collector.stop_collection()
        await alert_manager.stop_alerting()
        
    print("✅ Monitoring demo completed")


if __name__ == "__main__":
    asyncio.run(main())