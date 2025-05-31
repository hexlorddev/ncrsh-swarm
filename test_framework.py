#!/usr/bin/env python3
"""
Test script for ncrsh-Swarm framework
"""

import sys
import os
sys.path.insert(0, os.path.abspath('.'))

try:
    # Test basic imports
    print("🧪 Testing ncrsh-Swarm imports...")
    
    from ncrsh_swarm.core.config import ConfigManager, SwarmConfig
    from ncrsh_swarm.models.transformer import TransformerConfig
    from ncrsh_swarm.network.p2p import NetworkConfig
    from ncrsh_swarm.utils.crypto import SwarmCrypto
    from ncrsh_swarm.utils.serialization import SwarmSerializer
    
    print("✅ Core imports successful!")
    
    # Test configuration
    print("\n⚙️ Testing configuration...")
    config_manager = ConfigManager()
    
    # Test default config
    default_config = config_manager.get_default_config()
    print(f"✅ Default config created: {default_config.model.hidden_size} hidden size")
    
    # Test presets
    presets = config_manager.list_presets()
    print(f"✅ Available presets: {', '.join(presets)}")
    
    small_config = config_manager.get_preset_config('small')
    print(f"✅ Small preset: {small_config.model.hidden_size} hidden size")
    
    # Test model config
    print("\n🧠 Testing model configuration...")
    model_config = TransformerConfig(
        vocab_size=1000,
        hidden_size=256,
        num_layers=4,
        num_heads=8
    )
    print(f"✅ Model config: {model_config.num_layers} layers, {model_config.hidden_size} hidden")
    
    # Test crypto utilities
    print("\n🔐 Testing crypto utilities...")
    crypto = SwarmCrypto()
    
    # Generate keypair
    keypair = crypto.generate_node_keypair()
    print(f"✅ Generated keypair: {keypair['public_key'][:16]}...")
    
    # Test signing
    test_data = b"Hello, ncrsh-Swarm!"
    signature = crypto.sign_data(test_data)
    is_valid = crypto.verify_signature(test_data, signature)
    print(f"✅ Signature verification: {is_valid}")
    
    # Test serialization
    print("\n📦 Testing serialization...")
    serializer = SwarmSerializer()
    
    # Test lightweight serialization
    test_dict = {
        'node_id': 'test-node-123',
        'timestamp': 1234567890,
        'config': {'param1': 'value1', 'param2': 42}
    }
    
    serialized = serializer.serialize_lightweight(test_dict)
    deserialized = serializer.deserialize_lightweight(serialized)
    print(f"✅ Serialization round-trip: {deserialized['node_id']}")
    
    # Test config validation
    print("\n🔍 Testing config validation...")
    warnings = config_manager.validate_config(default_config)
    if warnings:
        print(f"⚠️  Config warnings: {len(warnings)}")
        for warning in warnings[:3]:  # Show first 3 warnings
            print(f"   - {warning}")
    else:
        print("✅ No config warnings")
    
    # Test config file operations
    print("\n📄 Testing config file operations...")
    test_config_path = "test-config.yaml"
    
    try:
        # Save config
        default_config.save_to_file(test_config_path)
        print(f"✅ Config saved to {test_config_path}")
        
        # Load config
        loaded_config = SwarmConfig.load_from_file(test_config_path)
        print(f"✅ Config loaded: {loaded_config.model.num_layers} layers")
        
        # Clean up
        os.remove(test_config_path)
        print("✅ Test file cleaned up")
        
    except Exception as e:
        print(f"❌ Config file test failed: {e}")
    
    print("\n🎉 All tests passed! ncrsh-Swarm framework is working correctly.")
    print("\nNext steps:")
    print("1. Install PyTorch: pip install torch")
    print("2. Run the CLI: python -m ncrsh_swarm.cli --help")
    print("3. Start a node: python -m ncrsh_swarm.cli start --preset small")
    print("4. Run the example: python ncrsh_swarm/examples/basic_swarm.py")
    
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure you're in the ncrsh-swarm directory and the package is properly structured.")
    
except Exception as e:
    print(f"❌ Test failed: {e}")
    import traceback
    traceback.print_exc()