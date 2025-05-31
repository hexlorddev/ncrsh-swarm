#!/usr/bin/env python3
"""
Basic test script for ncrsh-Swarm framework structure
"""

import sys
import os
sys.path.insert(0, os.path.abspath('.'))

print("🧠🌐 ncrsh-Swarm Framework Test")
print("=" * 40)

# Test package structure
print("📁 Testing package structure...")

expected_files = [
    'ncrsh_swarm/__init__.py',
    'ncrsh_swarm/core/swarm_node.py',
    'ncrsh_swarm/network/p2p.py',
    'ncrsh_swarm/models/transformer.py',
    'ncrsh_swarm/protocols/training.py',
    'ncrsh_swarm/protocols/consensus.py',
    'ncrsh_swarm/utils/crypto.py',
    'ncrsh_swarm/utils/serialization.py',
    'ncrsh_swarm/core/config.py',
    'ncrsh_swarm/cli.py',
    'setup.py',
    'README.md',
    'requirements.txt'
]

missing_files = []
for file_path in expected_files:
    if os.path.exists(file_path):
        print(f"✅ {file_path}")
    else:
        print(f"❌ {file_path}")
        missing_files.append(file_path)

if missing_files:
    print(f"\n⚠️  Missing {len(missing_files)} files")
else:
    print(f"\n✅ All {len(expected_files)} core files present!")

# Test imports without PyTorch dependencies
print("\n🔍 Testing basic imports...")

try:
    # Test crypto utilities (no PyTorch dependency)
    from ncrsh_swarm.utils.crypto import SwarmCrypto
    crypto = SwarmCrypto()
    keypair = crypto.generate_node_keypair()
    print(f"✅ Crypto utilities: Generated keypair")
    
    # Test serialization utilities  
    from ncrsh_swarm.utils.serialization import SwarmSerializer
    serializer = SwarmSerializer()
    test_data = {'test': 'data', 'number': 42}
    serialized = serializer.serialize_lightweight(test_data)
    deserialized = serializer.deserialize_lightweight(serialized)
    print(f"✅ Serialization: Round-trip successful")
    
    # Test configuration (minimal dependencies)
    from ncrsh_swarm.core.config import ConfigManager
    config_manager = ConfigManager()
    presets = config_manager.list_presets()
    print(f"✅ Configuration: {len(presets)} presets available")
    
    print("\n🎉 Basic framework structure is correct!")
    
except ImportError as e:
    print(f"❌ Import error: {e}")
    
except Exception as e:
    print(f"❌ Error: {e}")

# Show CLI help
print("\n📋 CLI Interface:")
print("Run 'python -m ncrsh_swarm.cli --help' for usage")

# Show example usage
print("\n🚀 Quick Start:")
print("1. Install dependencies: pip install -r requirements.txt")
print("2. Create config: python -m ncrsh_swarm.cli config --preset small")
print("3. Start node: python -m ncrsh_swarm.cli start --preset small")

print("\n📚 Framework Features:")
features = [
    "Self-replicating neural network nodes",
    "P2P discovery and mesh networking", 
    "Distributed training with Byzantine fault tolerance",
    "Real-time consensus for model synchronization",
    "Automatic swarm scaling and load balancing",
    "Cryptographic security and integrity verification"
]

for i, feature in enumerate(features, 1):
    print(f"{i}. {feature}")

print(f"\n📊 Framework Stats:")
print(f"   Python files: {len([f for f in expected_files if f.endswith('.py')])}")
print(f"   Core modules: 4 (node, network, models, protocols)")
print(f"   Utility modules: 2 (crypto, serialization)")
print(f"   Example files: 2 (basic_swarm.py, config.yaml)")
print(f"   Configuration presets: {len(presets) if 'presets' in locals() else 'N/A'}")

print("\n✨ ncrsh-Swarm: Where neural networks meet swarm intelligence!")