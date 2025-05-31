"""
Command-line interface for ncrsh-Swarm
"""

import asyncio
import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

from .core.swarm_node import SwarmNode, SwarmNodeConfig
from .core.config import ConfigManager, SwarmConfig
from .models.transformer import TransformerConfig
from .network.p2p import NetworkConfig


def setup_logging(level: str = "INFO"):
    """Setup logging configuration"""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('ncrsh-swarm.log')
        ]
    )


async def start_node(args):
    """Start a swarm node"""
    print("🧠 Starting ncrsh-Swarm Node...")
    
    # Load configuration
    config_manager = ConfigManager()
    
    if args.config:
        config = SwarmConfig.load_from_file(args.config)
    elif args.preset:
        config = config_manager.get_preset_config(args.preset)
    else:
        config = config_manager.get_default_config()
    
    # Override with command line arguments
    if args.port:
        config.network.port = args.port
    if args.max_peers:
        config.node.max_peers = args.max_peers
    if args.bootstrap:
        config.network.bootstrap_nodes = args.bootstrap.split(',')
        
    # Validate configuration
    warnings = config_manager.validate_config(config)
    if warnings:
        print("⚠️  Configuration warnings:")
        for warning in warnings:
            print(f"   - {warning}")
        print()
    
    # Create and start node
    node = SwarmNode(config.node)
    
    # Setup callbacks for better user experience
    node.on_peer_joined = lambda peer_id, info: print(f"✅ Peer joined: {peer_id[:8]}")
    node.on_peer_left = lambda peer_id, info: print(f"❌ Peer left: {peer_id[:8]}")
    node.on_model_updated = lambda state: print("🔄 Model updated via consensus")
    
    try:
        await node.start()
        
        # Display node information
        status = await node.get_swarm_status()
        print(f"🌐 Node ID: {status['node_id'][:8]}")
        print(f"🔌 Address: {status['network_address']}")
        print(f"👥 Max Peers: {config.node.max_peers}")
        print(f"🧠 Model Parameters: {status['model_params']:,}")
        print()
        print("Node is running! Press Ctrl+C to stop.")
        
        # Keep running
        while True:
            await asyncio.sleep(10)
            
            # Print periodic status
            status = await node.get_swarm_status()
            peer_count = status['peer_count']
            training = "🔥 Training" if status['training_active'] else "💤 Idle"
            
            print(f"📊 Status: {training} | Peers: {peer_count} | Clones: {status['clone_requests']}")
            
    except KeyboardInterrupt:
        print("\n🛑 Stopping node...")
        await node.stop()
        print("✅ Node stopped successfully")


async def train_node(args):
    """Train a swarm node"""
    print("🎓 Starting distributed training...")
    
    # This would connect to a running node and initiate training
    # For now, just show what would happen
    print(f"Would train for {args.epochs} epochs with dataset: {args.dataset}")
    print("Training functionality requires integration with actual datasets")


def create_config(args):
    """Create a configuration file"""
    config_manager = ConfigManager()
    
    if args.preset:
        config = config_manager.get_preset_config(args.preset)
    else:
        config = config_manager.get_default_config()
    
    # Save configuration
    config.save_to_file(args.output)
    print(f"✅ Configuration saved to: {args.output}")
    
    # Show preview
    if args.show:
        print("\n📄 Configuration preview:")
        import yaml
        print(yaml.dump(config.to_dict(), default_flow_style=False, indent=2))


def list_presets(args):
    """List available configuration presets"""
    config_manager = ConfigManager()
    presets = config_manager.list_presets()
    
    print("📋 Available configuration presets:")
    for preset in presets:
        print(f"   - {preset}")
    
    if args.describe:
        print(f"\n📝 Description of '{args.describe}' preset:")
        config = config_manager.get_preset_config(args.describe)
        print(f"   Model: {config.model.num_layers} layers, {config.model.hidden_size} hidden size")
        print(f"   Network: Max {config.node.max_peers} peers, port {config.network.port}")
        print(f"   Training: Batch size {config.training.get('batch_size', 32)}")


def show_status(args):
    """Show status of running nodes"""
    print("📊 Swarm Status:")
    print("This feature requires connecting to running nodes")
    print("Implementation would query nodes for their current status")


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="ncrsh-Swarm: Neural Network Framework That Self-Clones Across Systems",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  ncrsh-swarm start                          # Start with default config
  ncrsh-swarm start --preset small          # Start with small model preset
  ncrsh-swarm start --port 9090 --max-peers 5  # Override settings
  ncrsh-swarm config --preset large -o large.yaml  # Create config file
  ncrsh-swarm train --dataset ./data --epochs 50    # Train on dataset
        """
    )
    
    parser.add_argument('--log-level', default='INFO', 
                      choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                      help='Set logging level')
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Start command
    start_parser = subparsers.add_parser('start', help='Start a swarm node')
    start_parser.add_argument('--config', '-c', help='Configuration file path')
    start_parser.add_argument('--preset', '-p', help='Configuration preset name')
    start_parser.add_argument('--port', type=int, help='Network port')
    start_parser.add_argument('--max-peers', type=int, help='Maximum number of peers')
    start_parser.add_argument('--bootstrap', help='Comma-separated bootstrap node addresses')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train the swarm')
    train_parser.add_argument('--dataset', required=True, help='Path to training dataset')
    train_parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    train_parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    
    # Config command
    config_parser = subparsers.add_parser('config', help='Create configuration file')
    config_parser.add_argument('--preset', '-p', help='Base preset to use')
    config_parser.add_argument('--output', '-o', default='swarm-config.yaml', 
                             help='Output configuration file')
    config_parser.add_argument('--show', action='store_true', 
                             help='Show configuration preview')
    
    # Presets command
    presets_parser = subparsers.add_parser('presets', help='List configuration presets')
    presets_parser.add_argument('--describe', help='Describe a specific preset')
    
    # Status command
    status_parser = subparsers.add_parser('status', help='Show swarm status')
    status_parser.add_argument('--node', help='Specific node to query')
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    
    # Handle commands
    if args.command == 'start':
        asyncio.run(start_node(args))
    elif args.command == 'train':
        asyncio.run(train_node(args))
    elif args.command == 'config':
        create_config(args)
    elif args.command == 'presets':
        list_presets(args)
    elif args.command == 'status':
        show_status(args)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()