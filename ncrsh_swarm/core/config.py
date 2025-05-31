"""
Configuration management for SwarmNodes
"""

import os
import json
import yaml
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict
from pathlib import Path

from .swarm_node import SwarmNodeConfig
from ..models.transformer import TransformerConfig
from ..network.p2p import NetworkConfig


@dataclass
class SwarmConfig:
    """
    Complete configuration for a Swarm deployment
    """
    # Node configuration
    node: SwarmNodeConfig
    
    # Model configuration  
    model: TransformerConfig
    
    # Network configuration
    network: NetworkConfig
    
    # Training configuration
    training: Dict[str, Any]
    
    # Deployment configuration
    deployment: Dict[str, Any]
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'SwarmConfig':
        """Create SwarmConfig from dictionary"""
        return cls(
            node=SwarmNodeConfig(**config_dict.get('node', {})),
            model=TransformerConfig(**config_dict.get('model', {})),
            network=NetworkConfig(**config_dict.get('network', {})),
            training=config_dict.get('training', {}),
            deployment=config_dict.get('deployment', {})
        )
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert SwarmConfig to dictionary"""
        return {
            'node': asdict(self.node),
            'model': asdict(self.model),
            'network': asdict(self.network),
            'training': self.training,
            'deployment': self.deployment
        }
        
    @classmethod
    def load_from_file(cls, config_path: str) -> 'SwarmConfig':
        """Load configuration from file"""
        config_path = Path(config_path)
        
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
            
        with open(config_path, 'r') as f:
            if config_path.suffix.lower() in ['.yaml', '.yml']:
                config_dict = yaml.safe_load(f)
            elif config_path.suffix.lower() == '.json':
                config_dict = json.load(f)
            else:
                raise ValueError(f"Unsupported config file format: {config_path.suffix}")
                
        return cls.from_dict(config_dict)
        
    def save_to_file(self, config_path: str) -> None:
        """Save configuration to file"""
        config_path = Path(config_path)
        config_dict = self.to_dict()
        
        # Create directory if it doesn't exist
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(config_path, 'w') as f:
            if config_path.suffix.lower() in ['.yaml', '.yml']:
                yaml.dump(config_dict, f, default_flow_style=False, indent=2)
            elif config_path.suffix.lower() == '.json':
                json.dump(config_dict, f, indent=2)
            else:
                raise ValueError(f"Unsupported config file format: {config_path.suffix}")


class ConfigManager:
    """
    Manages configuration templates and presets
    """
    
    def __init__(self, config_dir: Optional[str] = None):
        self.config_dir = Path(config_dir or os.path.expanduser("~/.ncrsh-swarm"))
        self.config_dir.mkdir(parents=True, exist_ok=True)
        
    def get_default_config(self) -> SwarmConfig:
        """Get default configuration"""
        return SwarmConfig(
            node=SwarmNodeConfig(
                max_peers=10,
                sync_interval=30.0,
                clone_threshold=5,
                enable_auto_clone=True
            ),
            model=TransformerConfig(
                vocab_size=50257,
                max_seq_len=1024,
                hidden_size=768,
                num_layers=12,
                num_heads=12,
                dropout=0.1
            ),
            network=NetworkConfig(
                port=8080,
                discovery_port=8081,
                max_connections=100,
                timeout=30.0,
                enable_upnp=True
            ),
            training={
                'epochs': 100,
                'batch_size': 32,
                'learning_rate': 0.001,
                'sync_frequency': 5,
                'enable_mixed_precision': True,
                'gradient_clipping': 1.0
            },
            deployment={
                'auto_start': True,
                'log_level': 'INFO',
                'metrics_enabled': True,
                'dashboard_port': 8082,
                'data_dir': str(self.config_dir / 'data')
            }
        )
        
    def get_preset_config(self, preset_name: str) -> SwarmConfig:
        """Get a preset configuration"""
        presets = {
            'small': self._get_small_preset(),
            'medium': self._get_medium_preset(),
            'large': self._get_large_preset(),
            'distributed': self._get_distributed_preset(),
            'local': self._get_local_preset()
        }
        
        if preset_name not in presets:
            available = ', '.join(presets.keys())
            raise ValueError(f"Unknown preset: {preset_name}. Available: {available}")
            
        return presets[preset_name]
        
    def list_presets(self) -> List[str]:
        """List available preset configurations"""
        return ['small', 'medium', 'large', 'distributed', 'local']
        
    def save_preset(self, preset_name: str, config: SwarmConfig) -> None:
        """Save a custom preset"""
        preset_path = self.config_dir / 'presets' / f"{preset_name}.yaml"
        config.save_to_file(str(preset_path))
        
    def load_preset(self, preset_name: str) -> SwarmConfig:
        """Load a custom preset"""
        preset_path = self.config_dir / 'presets' / f"{preset_name}.yaml"
        if not preset_path.exists():
            raise FileNotFoundError(f"Preset not found: {preset_name}")
        return SwarmConfig.load_from_file(str(preset_path))
        
    def _get_small_preset(self) -> SwarmConfig:
        """Small model preset for testing"""
        config = self.get_default_config()
        
        # Small model
        config.model.hidden_size = 256
        config.model.num_layers = 6
        config.model.num_heads = 8
        config.model.max_seq_len = 512
        
        # Conservative training
        config.training['batch_size'] = 16
        config.training['learning_rate'] = 0.0005
        
        # Limited networking
        config.node.max_peers = 3
        config.network.max_connections = 10
        
        return config
        
    def _get_medium_preset(self) -> SwarmConfig:
        """Medium model preset for development"""
        config = self.get_default_config()
        
        # Medium model (default is already medium)
        config.training['batch_size'] = 32
        config.training['learning_rate'] = 0.001
        
        return config
        
    def _get_large_preset(self) -> SwarmConfig:
        """Large model preset for production"""
        config = self.get_default_config()
        
        # Large model
        config.model.hidden_size = 1024
        config.model.num_layers = 24
        config.model.num_heads = 16
        config.model.max_seq_len = 2048
        
        # Aggressive training
        config.training['batch_size'] = 64
        config.training['learning_rate'] = 0.0002
        config.training['gradient_clipping'] = 0.5
        
        # More networking
        config.node.max_peers = 20
        config.network.max_connections = 200
        
        return config
        
    def _get_distributed_preset(self) -> SwarmConfig:
        """Distributed deployment preset"""
        config = self.get_default_config()
        
        # Optimized for distribution
        config.node.max_peers = 50
        config.node.sync_interval = 15.0
        config.node.clone_threshold = 3
        config.node.enable_auto_clone = True
        
        # Fast networking
        config.network.max_connections = 500
        config.network.timeout = 10.0
        
        # Frequent syncing
        config.training['sync_frequency'] = 3
        config.training['batch_size'] = 128
        
        return config
        
    def _get_local_preset(self) -> SwarmConfig:
        """Local development preset"""
        config = self.get_default_config()
        
        # Single node setup
        config.node.max_peers = 1
        config.node.enable_auto_clone = False
        
        # Local networking
        config.network.port = 8080
        config.network.discovery_port = 8081
        config.network.bind_address = "127.0.0.1"
        
        # Simple training
        config.training['sync_frequency'] = 10
        config.training['batch_size'] = 8
        
        return config
        
    def validate_config(self, config: SwarmConfig) -> List[str]:
        """Validate configuration and return any warnings"""
        warnings = []
        
        # Model validation
        if config.model.hidden_size % config.model.num_heads != 0:
            warnings.append("hidden_size must be divisible by num_heads")
            
        if config.model.max_seq_len > 4096:
            warnings.append("max_seq_len > 4096 may cause memory issues")
            
        # Network validation
        if config.network.port == config.network.discovery_port:
            warnings.append("port and discovery_port should be different")
            
        if config.network.max_connections < config.node.max_peers:
            warnings.append("max_connections should be >= max_peers")
            
        # Training validation
        if config.training.get('batch_size', 32) > 256:
            warnings.append("Large batch_size may cause memory issues")
            
        if config.training.get('learning_rate', 0.001) > 0.01:
            warnings.append("High learning_rate may cause training instability")
            
        # Node validation
        if config.node.sync_interval < 5.0:
            warnings.append("Very low sync_interval may cause network congestion")
            
        return warnings
        
    def get_environment_config(self) -> Dict[str, Any]:
        """Get configuration from environment variables"""
        env_config = {}
        
        # Network settings
        if 'NCRSH_PORT' in os.environ:
            env_config.setdefault('network', {})['port'] = int(os.environ['NCRSH_PORT'])
            
        if 'NCRSH_DISCOVERY_PORT' in os.environ:
            env_config.setdefault('network', {})['discovery_port'] = int(os.environ['NCRSH_DISCOVERY_PORT'])
            
        if 'NCRSH_BIND_ADDRESS' in os.environ:
            env_config.setdefault('network', {})['bind_address'] = os.environ['NCRSH_BIND_ADDRESS']
            
        # Node settings
        if 'NCRSH_MAX_PEERS' in os.environ:
            env_config.setdefault('node', {})['max_peers'] = int(os.environ['NCRSH_MAX_PEERS'])
            
        if 'NCRSH_AUTO_CLONE' in os.environ:
            env_config.setdefault('node', {})['enable_auto_clone'] = os.environ['NCRSH_AUTO_CLONE'].lower() == 'true'
            
        # Training settings
        if 'NCRSH_BATCH_SIZE' in os.environ:
            env_config.setdefault('training', {})['batch_size'] = int(os.environ['NCRSH_BATCH_SIZE'])
            
        if 'NCRSH_LEARNING_RATE' in os.environ:
            env_config.setdefault('training', {})['learning_rate'] = float(os.environ['NCRSH_LEARNING_RATE'])
            
        # Bootstrap nodes
        if 'NCRSH_BOOTSTRAP_NODES' in os.environ:
            bootstrap_nodes = [
                node.strip() for node in os.environ['NCRSH_BOOTSTRAP_NODES'].split(',')
                if node.strip()
            ]
            env_config.setdefault('network', {})['bootstrap_nodes'] = bootstrap_nodes
            
        return env_config
        
    def merge_configs(self, base_config: SwarmConfig, override_config: Dict[str, Any]) -> SwarmConfig:
        """Merge override configuration into base configuration"""
        # Convert base config to dict
        merged_dict = base_config.to_dict()
        
        # Deep merge override config
        def deep_merge(base_dict, override_dict):
            for key, value in override_dict.items():
                if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
                    deep_merge(base_dict[key], value)
                else:
                    base_dict[key] = value
                    
        deep_merge(merged_dict, override_config)
        
        # Convert back to SwarmConfig
        return SwarmConfig.from_dict(merged_dict)