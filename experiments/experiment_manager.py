"""
ncrsh-Swarm Experiment Framework
==============================

Advanced experiment management system for distributed neural network research
with automated tracking, hyperparameter optimization, and result analysis.
"""

import asyncio
import json
import time
import uuid
import logging
from typing import Dict, List, Any, Optional, Callable, Union
from pathlib import Path
from dataclasses import dataclass, asdict, field
import pickle
import numpy as np
import matplotlib.pyplot as plt
from enum import Enum
import yaml

from ..ncrsh_swarm.core.swarm_node import SwarmNode, SwarmNodeConfig
from ..ncrsh_swarm.models.transformer import TransformerConfig
from ..benchmarks.swarm_benchmarker import SwarmBenchmarker


class ExperimentStatus(Enum):
    """Experiment execution status"""
    PENDING = "pending"
    RUNNING = "running" 
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class ExperimentConfig:
    """Configuration for a swarm experiment"""
    name: str
    description: str
    
    # Model configuration
    model_config: Dict[str, Any] = field(default_factory=dict)
    
    # Training configuration
    training_config: Dict[str, Any] = field(default_factory=dict)
    
    # Swarm configuration
    swarm_config: Dict[str, Any] = field(default_factory=dict)
    
    # Hyperparameters to optimize
    hyperparameters: Dict[str, Any] = field(default_factory=dict)
    
    # Metrics to track
    metrics: List[str] = field(default_factory=lambda: ['loss', 'accuracy', 'throughput'])
    
    # Resource constraints
    max_nodes: int = 10
    max_runtime_hours: float = 24.0
    max_memory_gb: float = 8.0
    
    # Experiment metadata
    tags: List[str] = field(default_factory=list)
    author: str = "ncrsh-swarm"
    priority: int = 1  # 1=high, 5=low


@dataclass
class ExperimentResult:
    """Results from a completed experiment"""
    experiment_id: str
    config: ExperimentConfig
    status: ExperimentStatus
    
    # Performance metrics
    final_metrics: Dict[str, float] = field(default_factory=dict)
    metric_history: Dict[str, List[float]] = field(default_factory=dict)
    
    # Resource usage
    resource_usage: Dict[str, Any] = field(default_factory=dict)
    
    # Timing information
    start_time: float = 0.0
    end_time: float = 0.0
    duration_seconds: float = 0.0
    
    # Additional data
    logs: List[str] = field(default_factory=list)
    artifacts: Dict[str, str] = field(default_factory=dict)  # filename -> path
    error_message: Optional[str] = None


class ExperimentTracker:
    """
    Track and manage experiment execution
    
    Features:
    - Real-time metric logging
    - Resource monitoring
    - Distributed experiment coordination
    - Result visualization and analysis
    """
    
    def __init__(self, experiment_id: str, output_dir: str = "./experiments"):
        self.experiment_id = experiment_id
        self.output_dir = Path(output_dir) / experiment_id
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.metrics: Dict[str, List[Tuple[float, float]]] = {}  # metric -> [(timestamp, value)]
        self.logs: List[Tuple[float, str, str]] = []  # (timestamp, level, message)
        self.artifacts: Dict[str, str] = {}
        
        # Setup logging
        self.logger = logging.getLogger(f"Experiment-{experiment_id[:8]}")
        self._setup_file_logging()
        
    def _setup_file_logging(self):
        """Setup file-based logging for the experiment"""
        log_file = self.output_dir / "experiment.log"
        
        handler = logging.FileHandler(log_file)
        handler.setLevel(logging.INFO)
        
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)
        
    def log_metric(self, name: str, value: float, step: Optional[int] = None):
        """Log a metric value"""
        timestamp = time.time()
        
        if name not in self.metrics:
            self.metrics[name] = []
            
        self.metrics[name].append((timestamp, value))
        
        # Also log to file
        self.logger.info(f"METRIC {name}: {value} (step: {step})")
        
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """Log multiple metrics at once"""
        for name, value in metrics.items():
            self.log_metric(name, value, step)
            
    def log_message(self, message: str, level: str = "INFO"):
        """Log a text message"""
        timestamp = time.time()
        self.logs.append((timestamp, level, message))
        
        # Log to file
        log_func = getattr(self.logger, level.lower(), self.logger.info)
        log_func(message)
        
    def save_artifact(self, name: str, data: Any, format: str = "pickle"):
        """Save an experiment artifact"""
        if format == "pickle":
            file_path = self.output_dir / f"{name}.pkl"
            with open(file_path, 'wb') as f:
                pickle.dump(data, f)
        elif format == "json":
            file_path = self.output_dir / f"{name}.json"
            with open(file_path, 'w') as f:
                json.dump(data, f, indent=2, default=str)
        elif format == "numpy":
            file_path = self.output_dir / f"{name}.npy"
            np.save(file_path, data)
        else:
            raise ValueError(f"Unsupported format: {format}")
            
        self.artifacts[name] = str(file_path)
        self.log_message(f"Saved artifact: {name} ({format})")
        
    def save_plot(self, name: str, figure=None):
        """Save a matplotlib plot"""
        if figure is None:
            figure = plt.gcf()
            
        file_path = self.output_dir / f"{name}.png"
        figure.savefig(file_path, dpi=300, bbox_inches='tight')
        
        self.artifacts[name] = str(file_path)
        self.log_message(f"Saved plot: {name}")
        
    def get_metric_history(self, name: str) -> List[Tuple[float, float]]:
        """Get the history of a metric"""
        return self.metrics.get(name, [])
        
    def get_latest_metric(self, name: str) -> Optional[float]:
        """Get the latest value of a metric"""
        history = self.get_metric_history(name)
        return history[-1][1] if history else None
        
    def plot_metrics(self, metric_names: Optional[List[str]] = None):
        """Plot metric histories"""
        if metric_names is None:
            metric_names = list(self.metrics.keys())
            
        fig, axes = plt.subplots(len(metric_names), 1, figsize=(10, 3*len(metric_names)))
        if len(metric_names) == 1:
            axes = [axes]
            
        for i, metric_name in enumerate(metric_names):
            history = self.get_metric_history(metric_name)
            
            if history:
                timestamps, values = zip(*history)
                # Convert to relative time (minutes from start)
                start_time = timestamps[0]
                relative_times = [(t - start_time) / 60 for t in timestamps]
                
                axes[i].plot(relative_times, values, 'b-', linewidth=2)
                axes[i].set_title(f'{metric_name} over time')
                axes[i].set_xlabel('Time (minutes)')
                axes[i].set_ylabel(metric_name)
                axes[i].grid(True, alpha=0.3)
                
        plt.tight_layout()
        self.save_plot(f"metrics_plot")
        
    def export_results(self) -> Dict[str, Any]:
        """Export all experiment data"""
        return {
            'experiment_id': self.experiment_id,
            'metrics': self.metrics,
            'logs': self.logs,
            'artifacts': self.artifacts,
            'output_dir': str(self.output_dir)
        }


class ExperimentManager:
    """
    Central experiment management system
    
    Features:
    - Experiment queue management
    - Hyperparameter optimization
    - Distributed experiment execution
    - Result analysis and comparison
    """
    
    def __init__(self, experiments_dir: str = "./experiments"):
        self.experiments_dir = Path(experiments_dir)
        self.experiments_dir.mkdir(parents=True, exist_ok=True)
        
        self.experiments: Dict[str, ExperimentResult] = {}
        self.running_experiments: Dict[str, ExperimentTracker] = {}
        
        # Load existing experiments
        self._load_experiments()
        
    def _load_experiments(self):
        """Load existing experiments from disk"""
        for exp_dir in self.experiments_dir.iterdir():
            if exp_dir.is_dir():
                result_file = exp_dir / "result.json"
                if result_file.exists():
                    try:
                        with open(result_file, 'r') as f:
                            result_data = json.load(f)
                        
                        # Reconstruct ExperimentResult
                        config_data = result_data.pop('config')
                        config = ExperimentConfig(**config_data)
                        
                        result = ExperimentResult(
                            config=config,
                            **result_data
                        )
                        result.status = ExperimentStatus(result.status)
                        
                        self.experiments[result.experiment_id] = result
                        
                    except Exception as e:
                        print(f"Failed to load experiment {exp_dir.name}: {e}")
                        
    def create_experiment(self, config: ExperimentConfig) -> str:
        """Create a new experiment"""
        experiment_id = str(uuid.uuid4())
        
        # Create experiment result
        result = ExperimentResult(
            experiment_id=experiment_id,
            config=config,
            status=ExperimentStatus.PENDING
        )
        
        self.experiments[experiment_id] = result
        
        # Save to disk
        self._save_experiment(result)
        
        return experiment_id
        
    async def run_experiment(self, experiment_id: str) -> ExperimentResult:
        """Run a single experiment"""
        if experiment_id not in self.experiments:
            raise ValueError(f"Experiment {experiment_id} not found")
            
        result = self.experiments[experiment_id]
        
        if result.status != ExperimentStatus.PENDING:
            raise ValueError(f"Experiment {experiment_id} is not pending")
            
        # Create tracker
        tracker = ExperimentTracker(experiment_id, str(self.experiments_dir))
        self.running_experiments[experiment_id] = tracker
        
        # Update status
        result.status = ExperimentStatus.RUNNING
        result.start_time = time.time()
        
        try:
            tracker.log_message("Starting experiment execution")
            
            # Execute the experiment
            await self._execute_experiment(result, tracker)
            
            # Mark as completed
            result.status = ExperimentStatus.COMPLETED
            tracker.log_message("Experiment completed successfully")
            
        except Exception as e:
            result.status = ExperimentStatus.FAILED
            result.error_message = str(e)
            tracker.log_message(f"Experiment failed: {e}", "ERROR")
            
        finally:
            result.end_time = time.time()
            result.duration_seconds = result.end_time - result.start_time
            
            # Collect final results
            result.final_metrics = {
                name: tracker.get_latest_metric(name) or 0.0
                for name in result.config.metrics
            }
            result.metric_history = tracker.metrics
            result.logs = [f"{level}: {msg}" for _, level, msg in tracker.logs]
            result.artifacts = tracker.artifacts
            
            # Generate plots
            tracker.plot_metrics()
            
            # Save final result
            self._save_experiment(result)
            
            # Cleanup
            if experiment_id in self.running_experiments:
                del self.running_experiments[experiment_id]
                
        return result
        
    async def _execute_experiment(self, result: ExperimentResult, tracker: ExperimentTracker):
        """Execute the actual experiment"""
        config = result.config
        
        # Create swarm configuration
        swarm_config = SwarmNodeConfig(
            model_config=TransformerConfig(**config.model_config),
            max_peers=config.max_nodes,
            **config.swarm_config
        )
        
        # Create swarm node
        node = SwarmNode(swarm_config)
        
        try:
            tracker.log_message(f"Starting swarm node with {config.max_nodes} max peers")
            await node.start()
            
            # Wait for peer discovery
            await asyncio.sleep(5)
            
            # Log initial status
            status = await node.get_swarm_status()
            tracker.log_metrics({
                'peer_count': status['peer_count'],
                'model_params': status['model_params']
            })
            
            # Simulate training (in real implementation, would use actual training)
            training_epochs = config.training_config.get('epochs', 10)
            
            for epoch in range(training_epochs):
                # Simulate training metrics
                loss = 2.0 * np.exp(-epoch * 0.1) + np.random.normal(0, 0.05)
                accuracy = (1 - np.exp(-epoch * 0.15)) * 0.95 + np.random.normal(0, 0.02)
                throughput = 100 + np.random.normal(0, 10)
                
                tracker.log_metrics({
                    'loss': max(0.1, loss),
                    'accuracy': min(0.99, max(0.0, accuracy)),
                    'throughput': max(50, throughput)
                }, step=epoch)
                
                tracker.log_message(f"Epoch {epoch+1}/{training_epochs} completed")
                
                # Simulate training time
                await asyncio.sleep(1)
                
                # Check for early stopping
                if tracker.get_latest_metric('loss') < 0.1:
                    tracker.log_message("Early stopping - loss threshold reached")
                    break
                    
            # Save model checkpoint
            model_state = node.model.state_dict()
            tracker.save_artifact("final_model", model_state)
            
        finally:
            await node.stop()
            
    def _save_experiment(self, result: ExperimentResult):
        """Save experiment result to disk"""
        exp_dir = self.experiments_dir / result.experiment_id
        exp_dir.mkdir(exist_ok=True)
        
        result_file = exp_dir / "result.json"
        
        # Convert to dict for JSON serialization
        result_dict = asdict(result)
        result_dict['status'] = result.status.value
        
        with open(result_file, 'w') as f:
            json.dump(result_dict, f, indent=2, default=str)
            
    async def run_hyperparameter_sweep(
        self, 
        base_config: ExperimentConfig,
        param_grid: Dict[str, List[Any]],
        max_concurrent: int = 3
    ) -> List[str]:
        """Run hyperparameter sweep"""
        import itertools
        
        # Generate all parameter combinations
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        combinations = list(itertools.product(*param_values))
        
        print(f"🔬 Running hyperparameter sweep: {len(combinations)} experiments")
        
        experiment_ids = []
        
        # Create experiments for each combination
        for i, combo in enumerate(combinations):
            # Create modified config
            sweep_config = ExperimentConfig(
                name=f"{base_config.name}_sweep_{i}",
                description=f"Hyperparameter sweep {i+1}/{len(combinations)}",
                model_config=base_config.model_config.copy(),
                training_config=base_config.training_config.copy(),
                swarm_config=base_config.swarm_config.copy(),
                metrics=base_config.metrics.copy(),
                tags=base_config.tags + ['hyperparameter_sweep']
            )
            
            # Apply hyperparameters
            for param_name, param_value in zip(param_names, combo):
                if param_name.startswith('model.'):
                    sweep_config.model_config[param_name[6:]] = param_value
                elif param_name.startswith('training.'):
                    sweep_config.training_config[param_name[9:]] = param_value
                else:
                    sweep_config.hyperparameters[param_name] = param_value
                    
            experiment_id = self.create_experiment(sweep_config)
            experiment_ids.append(experiment_id)
            
        # Run experiments with concurrency limit
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def run_with_semaphore(exp_id):
            async with semaphore:
                return await self.run_experiment(exp_id)
                
        # Execute all experiments
        tasks = [run_with_semaphore(exp_id) for exp_id in experiment_ids]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Analyze results
        self._analyze_sweep_results(experiment_ids, param_names)
        
        return experiment_ids
        
    def _analyze_sweep_results(self, experiment_ids: List[str], param_names: List[str]):
        """Analyze hyperparameter sweep results"""
        results_data = []
        
        for exp_id in experiment_ids:
            if exp_id in self.experiments:
                result = self.experiments[exp_id]
                
                if result.status == ExperimentStatus.COMPLETED:
                    data = {
                        'experiment_id': exp_id,
                        'final_loss': result.final_metrics.get('loss', float('inf')),
                        'final_accuracy': result.final_metrics.get('accuracy', 0.0),
                        'duration': result.duration_seconds
                    }
                    
                    # Add hyperparameters
                    for param_name, param_value in result.config.hyperparameters.items():
                        data[param_name] = param_value
                        
                    results_data.append(data)
                    
        # Find best experiment
        if results_data:
            best_result = min(results_data, key=lambda x: x['final_loss'])
            print(f"🏆 Best experiment: {best_result['experiment_id']}")
            print(f"   Final loss: {best_result['final_loss']:.4f}")
            print(f"   Final accuracy: {best_result['final_accuracy']:.4f}")
            
            # Save sweep analysis
            analysis_file = self.experiments_dir / "sweep_analysis.json"
            with open(analysis_file, 'w') as f:
                json.dump({
                    'best_result': best_result,
                    'all_results': results_data,
                    'param_names': param_names
                }, f, indent=2)
                
    def list_experiments(self, status: Optional[ExperimentStatus] = None) -> List[ExperimentResult]:
        """List experiments, optionally filtered by status"""
        experiments = list(self.experiments.values())
        
        if status:
            experiments = [exp for exp in experiments if exp.status == status]
            
        return sorted(experiments, key=lambda x: x.start_time, reverse=True)
        
    def get_experiment(self, experiment_id: str) -> Optional[ExperimentResult]:
        """Get a specific experiment"""
        return self.experiments.get(experiment_id)
        
    def compare_experiments(self, experiment_ids: List[str]) -> Dict[str, Any]:
        """Compare multiple experiments"""
        comparison_data = {
            'experiments': {},
            'metrics_comparison': {},
            'summary': {}
        }
        
        for exp_id in experiment_ids:
            if exp_id in self.experiments:
                result = self.experiments[exp_id]
                comparison_data['experiments'][exp_id] = {
                    'name': result.config.name,
                    'status': result.status.value,
                    'final_metrics': result.final_metrics,
                    'duration': result.duration_seconds
                }
                
        # Analyze metrics across experiments
        all_metric_names = set()
        for exp_data in comparison_data['experiments'].values():
            all_metric_names.update(exp_data['final_metrics'].keys())
            
        for metric_name in all_metric_names:
            metric_values = []
            
            for exp_id, exp_data in comparison_data['experiments'].items():
                value = exp_data['final_metrics'].get(metric_name)
                if value is not None:
                    metric_values.append(value)
                    
            if metric_values:
                comparison_data['metrics_comparison'][metric_name] = {
                    'min': min(metric_values),
                    'max': max(metric_values),
                    'mean': sum(metric_values) / len(metric_values),
                    'std': np.std(metric_values)
                }
                
        return comparison_data


# CLI interface
async def main():
    """Example experiment execution"""
    print("🔬 ncrsh-Swarm Experiment Framework")
    
    # Create experiment manager
    manager = ExperimentManager()
    
    # Create base experiment configuration
    config = ExperimentConfig(
        name="test_transformer_training",
        description="Test transformer training with swarm",
        model_config={
            'hidden_size': 256,
            'num_layers': 4,
            'num_heads': 8
        },
        training_config={
            'epochs': 5,
            'batch_size': 16,
            'learning_rate': 0.001
        },
        swarm_config={
            'max_peers': 3
        },
        tags=['test', 'transformer', 'distributed']
    )
    
    # Run single experiment
    print("Running single experiment...")
    experiment_id = manager.create_experiment(config)
    result = await manager.run_experiment(experiment_id)
    
    print(f"Experiment completed with status: {result.status.value}")
    print(f"Final metrics: {result.final_metrics}")
    
    # Run hyperparameter sweep
    print("\nRunning hyperparameter sweep...")
    param_grid = {
        'model.hidden_size': [128, 256, 512],
        'training.learning_rate': [0.0001, 0.001, 0.01]
    }
    
    sweep_ids = await manager.run_hyperparameter_sweep(
        config, param_grid, max_concurrent=2
    )
    
    print(f"Completed sweep with {len(sweep_ids)} experiments")
    
    # Compare results
    comparison = manager.compare_experiments(sweep_ids[:3])
    print(f"Comparison metrics: {comparison['metrics_comparison']}")


if __name__ == "__main__":
    asyncio.run(main())