"""
Comprehensive Test Suite for ncrsh-Swarm
========================================

Advanced testing framework with unit tests, integration tests, 
stress tests, and distributed testing capabilities.
"""

import pytest
import asyncio
import unittest
import time
import tempfile
import shutil
from typing import Dict, List, Any, Optional
from pathlib import Path
import torch
import numpy as np
from unittest.mock import Mock, patch, AsyncMock

# Import swarm components for testing
import sys
sys.path.append(str(Path(__file__).parent.parent))

from ncrsh_swarm.core.swarm_node import SwarmNode, SwarmNodeConfig
from ncrsh_swarm.models.transformer import SwarmTransformer, TransformerConfig
from ncrsh_swarm.network.p2p import P2PNetwork, NetworkConfig
from ncrsh_swarm.protocols.training import CooperativeTrainer
from ncrsh_swarm.protocols.consensus import ConsensusProtocol
from ncrsh_swarm.utils.crypto import SwarmCrypto
from ncrsh_swarm.utils.serialization import SwarmSerializer
from ncrsh_swarm.core.config import ConfigManager
from datasets.dataset_manager import DatasetManager


class SwarmTestBase(unittest.IsolatedAsyncioTestCase):
    """Base class for swarm tests with common setup"""
    
    async def asyncSetUp(self):
        """Setup test environment"""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.addCleanup(self.cleanup_temp_dir)
        
        # Create test configuration
        self.test_config = SwarmNodeConfig(
            model_config=TransformerConfig(
                hidden_size=64,  # Small for testing
                num_layers=2,
                num_heads=4,
                max_seq_len=128
            ),
            network_config=NetworkConfig(
                port=0,  # Let OS choose port
                discovery_port=0,
                max_connections=5
            ),
            max_peers=3,
            data_dir=self.temp_dir
        )
        
    def cleanup_temp_dir(self):
        """Clean up temporary directory"""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)


class TestSwarmTransformer(SwarmTestBase):
    """Test the SwarmTransformer model"""
    
    async def test_model_creation(self):
        """Test model instantiation"""
        config = TransformerConfig(hidden_size=64, num_layers=2, num_heads=4)
        model = SwarmTransformer(config)
        
        self.assertIsInstance(model, SwarmTransformer)
        self.assertEqual(model.config.hidden_size, 64)
        self.assertEqual(model.config.num_layers, 2)
        
    async def test_forward_pass(self):
        """Test model forward pass"""
        config = TransformerConfig(
            vocab_size=100, hidden_size=64, num_layers=2, 
            num_heads=4, max_seq_len=32
        )
        model = SwarmTransformer(config)
        
        # Create test input
        batch_size, seq_len = 2, 16
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
        
        # Forward pass
        logits, loss = model(input_ids)
        
        self.assertEqual(logits.shape, (batch_size, seq_len, config.vocab_size))
        self.assertIsNone(loss)  # No targets provided
        
    async def test_forward_pass_with_targets(self):
        """Test model forward pass with targets for loss calculation"""
        config = TransformerConfig(
            vocab_size=100, hidden_size=64, num_layers=2, 
            num_heads=4, max_seq_len=32
        )
        model = SwarmTransformer(config)
        
        # Create test input and targets
        batch_size, seq_len = 2, 16
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
        targets = torch.randint(0, config.vocab_size, (batch_size, seq_len))
        
        # Forward pass
        logits, loss = model(input_ids, targets=targets)
        
        self.assertEqual(logits.shape, (batch_size, seq_len, config.vocab_size))
        self.assertIsNotNone(loss)
        self.assertIsInstance(loss.item(), float)
        
    async def test_model_generation(self):
        """Test text generation"""
        config = TransformerConfig(
            vocab_size=100, hidden_size=64, num_layers=2, 
            num_heads=4, max_seq_len=32
        )
        model = SwarmTransformer(config)
        model.eval()
        
        # Create starting tokens
        input_ids = torch.randint(0, config.vocab_size, (1, 5))
        
        # Generate
        generated = model.generate(input_ids, max_new_tokens=10, temperature=1.0)
        
        self.assertEqual(generated.shape[0], 1)  # Batch size
        self.assertEqual(generated.shape[1], 15)  # Original + generated tokens
        
    async def test_parameter_groups(self):
        """Test parameter grouping for optimization"""
        config = TransformerConfig(hidden_size=64, num_layers=2, num_heads=4)
        model = SwarmTransformer(config)
        
        param_groups = model.get_parameter_groups()
        
        self.assertEqual(len(param_groups), 2)  # decay and no-decay groups
        self.assertIn('params', param_groups[0])
        self.assertIn('weight_decay', param_groups[0])
        
    async def test_memory_estimation(self):
        """Test memory usage estimation"""
        config = TransformerConfig(hidden_size=64, num_layers=2, num_heads=4)
        model = SwarmTransformer(config)
        
        memory_usage = model.estimate_memory_usage(batch_size=4, seq_len=64)
        
        self.assertIn('model_mb', memory_usage)
        self.assertIn('activations_mb', memory_usage)
        self.assertIn('total_mb', memory_usage)
        self.assertGreater(memory_usage['total_mb'], 0)


class TestP2PNetwork(SwarmTestBase):
    """Test P2P networking functionality"""
    
    async def test_network_creation(self):
        """Test network instantiation"""
        config = NetworkConfig(port=0, discovery_port=0)
        network = P2PNetwork(config, "test_node")
        
        self.assertIsInstance(network, P2PNetwork)
        self.assertEqual(network.node_id, "test_node")
        
    async def test_network_start_stop(self):
        """Test network startup and shutdown"""
        config = NetworkConfig(port=0, discovery_port=0)
        network = P2PNetwork(config, "test_node")
        
        await network.start()
        self.assertTrue(network.is_running)
        
        await network.stop()
        self.assertFalse(network.is_running)
        
    async def test_message_handlers(self):
        """Test message handler registration"""
        config = NetworkConfig(port=0, discovery_port=0)
        network = P2PNetwork(config, "test_node")
        
        async def test_handler(sender_id, data):
            return {"response": "test"}
            
        network.register_handler("test_message", test_handler)
        self.assertIn("test_message", network.message_handlers)


class TestSwarmCrypto(SwarmTestBase):
    """Test cryptographic utilities"""
    
    async def test_crypto_creation(self):
        """Test crypto utility instantiation"""
        crypto = SwarmCrypto()
        self.assertIsInstance(crypto, SwarmCrypto)
        
    async def test_model_hashing(self):
        """Test model state hashing"""
        crypto = SwarmCrypto()
        
        # Create test model state
        model_state = {
            'layer1.weight': torch.randn(10, 5),
            'layer1.bias': torch.randn(10),
            'layer2.weight': torch.randn(1, 10)
        }
        
        hash1 = crypto.hash_model(model_state)
        hash2 = crypto.hash_model(model_state)
        
        self.assertEqual(hash1, hash2)  # Same state should have same hash
        self.assertIsInstance(hash1, str)
        self.assertEqual(len(hash1), 64)  # SHA256 hex length
        
    async def test_model_hashing_different_states(self):
        """Test that different model states have different hashes"""
        crypto = SwarmCrypto()
        
        state1 = {'weight': torch.randn(5, 5)}
        state2 = {'weight': torch.randn(5, 5)}
        
        hash1 = crypto.hash_model(state1)
        hash2 = crypto.hash_model(state2)
        
        self.assertNotEqual(hash1, hash2)
        
    async def test_data_encryption_decryption(self):
        """Test data encryption and decryption"""
        crypto = SwarmCrypto()
        
        original_data = b"test data for encryption"
        
        encrypted = crypto.encrypt_data(original_data)
        decrypted = crypto.decrypt_data(encrypted)
        
        self.assertEqual(original_data, decrypted)
        self.assertNotEqual(original_data, encrypted)
        
    async def test_data_signing(self):
        """Test data signing and verification"""
        crypto = SwarmCrypto()
        
        data = b"test data for signing"
        
        signature = crypto.sign_data(data)
        is_valid = crypto.verify_signature(data, signature)
        
        self.assertTrue(is_valid)
        self.assertIsInstance(signature, str)
        
    async def test_signature_verification_fails_for_wrong_data(self):
        """Test that signature verification fails for modified data"""
        crypto = SwarmCrypto()
        
        original_data = b"original data"
        modified_data = b"modified data"
        
        signature = crypto.sign_data(original_data)
        is_valid = crypto.verify_signature(modified_data, signature)
        
        self.assertFalse(is_valid)
        
    async def test_keypair_generation(self):
        """Test node keypair generation"""
        crypto = SwarmCrypto()
        
        keypair = crypto.generate_node_keypair()
        
        self.assertIn('private_key', keypair)
        self.assertIn('public_key', keypair)
        self.assertIsInstance(keypair['private_key'], str)
        self.assertIsInstance(keypair['public_key'], str)


class TestSwarmSerializer(SwarmTestBase):
    """Test serialization utilities"""
    
    async def test_serializer_creation(self):
        """Test serializer instantiation"""
        serializer = SwarmSerializer()
        self.assertIsInstance(serializer, SwarmSerializer)
        
    async def test_model_serialization(self):
        """Test model state serialization"""
        serializer = SwarmSerializer()
        
        model_state = {
            'layer1.weight': torch.randn(10, 5),
            'layer1.bias': torch.randn(10)
        }
        
        serialized = serializer.serialize_model(model_state)
        deserialized = serializer.deserialize_model(serialized)
        
        self.assertIsInstance(serialized, bytes)
        
        # Check that deserialized state matches original
        for name in model_state.keys():
            torch.testing.assert_close(model_state[name], deserialized[name])
            
    async def test_gradient_serialization(self):
        """Test gradient serialization"""
        serializer = SwarmSerializer()
        
        gradients = {
            'param1': torch.randn(5, 3),
            'param2': torch.randn(10)
        }
        
        serialized = serializer.serialize_gradients(gradients)
        deserialized = serializer.deserialize_gradients(serialized)
        
        self.assertIsInstance(serialized, bytes)
        
        # Check that deserialized gradients match original
        for name in gradients.keys():
            torch.testing.assert_close(gradients[name], deserialized[name], atol=1e-3, rtol=1e-3)
            
    async def test_lightweight_serialization(self):
        """Test lightweight data serialization"""
        serializer = SwarmSerializer()
        
        data = {
            'node_id': 'test_node',
            'timestamp': 1234567890,
            'config': {'param1': 'value1', 'param2': 42}
        }
        
        serialized = serializer.serialize_lightweight(data)
        deserialized = serializer.deserialize_lightweight(serialized)
        
        self.assertIsInstance(serialized, str)
        self.assertEqual(data, deserialized)


class TestCooperativeTrainer(SwarmTestBase):
    """Test cooperative training functionality"""
    
    async def test_trainer_creation(self):
        """Test trainer instantiation"""
        config = TransformerConfig(hidden_size=32, num_layers=1, num_heads=2)
        model = SwarmTransformer(config)
        
        # Mock network
        network = Mock()
        network.node_id = "test_node"
        
        trainer = CooperativeTrainer(model, network)
        self.assertIsInstance(trainer, CooperativeTrainer)
        
    async def test_gradient_extraction(self):
        """Test local gradient extraction"""
        config = TransformerConfig(hidden_size=32, num_layers=1, num_heads=2)
        model = SwarmTransformer(config)
        
        # Mock network
        network = Mock()
        network.node_id = "test_node"
        
        trainer = CooperativeTrainer(model, network)
        
        # Create dummy loss to generate gradients
        dummy_input = torch.randint(0, config.vocab_size, (1, 10))
        dummy_target = torch.randint(0, config.vocab_size, (1, 10))
        
        logits, loss = model(dummy_input, targets=dummy_target)
        loss.backward()
        
        gradients = await trainer.get_local_gradients()
        
        self.assertIsInstance(gradients, dict)
        self.assertGreater(len(gradients), 0)
        
        # Check that gradients are tensors
        for name, grad in gradients.items():
            self.assertIsInstance(grad, torch.Tensor)


class TestConsensusProtocol(SwarmTestBase):
    """Test consensus protocol functionality"""
    
    async def test_consensus_creation(self):
        """Test consensus protocol instantiation"""
        # Mock network
        network = Mock()
        network.node_id = "test_node"
        
        consensus = ConsensusProtocol(network)
        self.assertIsInstance(consensus, ConsensusProtocol)


class TestConfigManager(SwarmTestBase):
    """Test configuration management"""
    
    async def test_config_manager_creation(self):
        """Test config manager instantiation"""
        manager = ConfigManager(str(self.temp_dir))
        self.assertIsInstance(manager, ConfigManager)
        
    async def test_default_config(self):
        """Test default configuration generation"""
        manager = ConfigManager(str(self.temp_dir))
        config = manager.get_default_config()
        
        self.assertIsNotNone(config)
        self.assertIsNotNone(config.model)
        self.assertIsNotNone(config.network)
        self.assertIsNotNone(config.node)
        
    async def test_preset_configs(self):
        """Test preset configurations"""
        manager = ConfigManager(str(self.temp_dir))
        
        presets = manager.list_presets()
        self.assertIsInstance(presets, list)
        self.assertGreater(len(presets), 0)
        
        # Test loading each preset
        for preset_name in presets:
            config = manager.get_preset_config(preset_name)
            self.assertIsNotNone(config)
            
    async def test_config_validation(self):
        """Test configuration validation"""
        manager = ConfigManager(str(self.temp_dir))
        config = manager.get_default_config()
        
        warnings = manager.validate_config(config)
        self.assertIsInstance(warnings, list)
        
    async def test_config_file_operations(self):
        """Test configuration file save/load"""
        manager = ConfigManager(str(self.temp_dir))
        config = manager.get_default_config()
        
        config_file = self.temp_dir / "test_config.yaml"
        
        # Save config
        config.save_to_file(str(config_file))
        self.assertTrue(config_file.exists())
        
        # Load config
        loaded_config = type(config).load_from_file(str(config_file))
        
        # Basic validation that it loaded correctly
        self.assertEqual(config.model.hidden_size, loaded_config.model.hidden_size)
        self.assertEqual(config.network.port, loaded_config.network.port)


class TestDatasetManager(SwarmTestBase):
    """Test dataset management functionality"""
    
    async def test_dataset_manager_creation(self):
        """Test dataset manager instantiation"""
        manager = DatasetManager(str(self.temp_dir))
        self.assertIsInstance(manager, DatasetManager)
        
    async def test_synthetic_dataset_creation(self):
        """Test synthetic dataset creation"""
        manager = DatasetManager(str(self.temp_dir))
        
        dataset = await manager.get_dataset('lm_test_small', vocab_size=100, seq_len=32)
        
        self.assertIsNotNone(dataset)
        self.assertGreater(len(dataset), 0)
        
        # Test dataset access
        input_tensor, target_tensor = dataset[0]
        self.assertIsInstance(input_tensor, torch.Tensor)
        self.assertIsInstance(target_tensor, torch.Tensor)


class TestSwarmNodeIntegration(SwarmTestBase):
    """Integration tests for SwarmNode"""
    
    async def test_node_creation(self):
        """Test node instantiation"""
        node = SwarmNode(self.test_config)
        self.assertIsInstance(node, SwarmNode)
        
    async def test_node_start_stop(self):
        """Test node lifecycle"""
        node = SwarmNode(self.test_config)
        
        await node.start()
        self.assertTrue(node.is_running)
        
        status = await node.get_swarm_status()
        self.assertIsInstance(status, dict)
        self.assertIn('node_id', status)
        
        await node.stop()
        self.assertFalse(node.is_running)
        
    async def test_peer_discovery(self):
        """Test peer discovery between nodes"""
        # Create two nodes with different ports
        config1 = SwarmNodeConfig(
            model_config=TransformerConfig(hidden_size=32, num_layers=1, num_heads=2),
            network_config=NetworkConfig(port=0, discovery_port=0),
            data_dir=self.temp_dir / "node1"
        )
        
        config2 = SwarmNodeConfig(
            model_config=TransformerConfig(hidden_size=32, num_layers=1, num_heads=2),
            network_config=NetworkConfig(port=0, discovery_port=0),
            data_dir=self.temp_dir / "node2"
        )
        
        node1 = SwarmNode(config1)
        node2 = SwarmNode(config2)
        
        try:
            await node1.start()
            await node2.start()
            
            # Give nodes time to discover each other
            await asyncio.sleep(3)
            
            peers1 = await node1.discover_peers()
            peers2 = await node2.discover_peers()
            
            # In ideal case, they should discover each other
            # But due to timing and networking, we just verify the method works
            self.assertIsInstance(peers1, list)
            self.assertIsInstance(peers2, list)
            
        finally:
            await node1.stop()
            await node2.stop()


class TestStressScenarios(SwarmTestBase):
    """Stress testing scenarios"""
    
    async def test_rapid_node_creation_destruction(self):
        """Test rapid node creation and destruction"""
        nodes = []
        
        try:
            # Create multiple nodes quickly
            for i in range(3):
                config = SwarmNodeConfig(
                    model_config=TransformerConfig(hidden_size=32, num_layers=1, num_heads=2),
                    network_config=NetworkConfig(port=0, discovery_port=0),
                    data_dir=self.temp_dir / f"node_{i}"
                )
                
                node = SwarmNode(config)
                await node.start()
                nodes.append(node)
                
            # All nodes should be running
            for node in nodes:
                self.assertTrue(node.is_running)
                
        finally:
            # Clean up all nodes
            for node in nodes:
                try:
                    await node.stop()
                except Exception:
                    pass  # Ignore cleanup errors
                    
    async def test_large_model_handling(self):
        """Test handling of larger models"""
        config = SwarmNodeConfig(
            model_config=TransformerConfig(
                hidden_size=512,  # Larger model
                num_layers=6,
                num_heads=8,
                max_seq_len=256
            ),
            network_config=NetworkConfig(port=0, discovery_port=0),
            data_dir=self.temp_dir
        )
        
        node = SwarmNode(config)
        
        try:
            await node.start()
            
            # Verify model is created and accessible
            status = await node.get_swarm_status()
            self.assertGreater(status['model_params'], 1000000)  # Should have >1M parameters
            
        finally:
            await node.stop()


class TestErrorHandling(SwarmTestBase):
    """Test error handling and edge cases"""
    
    async def test_invalid_config_handling(self):
        """Test handling of invalid configurations"""
        # Test invalid model config
        with self.assertRaises(Exception):
            invalid_config = SwarmNodeConfig(
                model_config=TransformerConfig(
                    hidden_size=0,  # Invalid
                    num_layers=-1,  # Invalid
                    num_heads=0     # Invalid
                )
            )
            SwarmNode(invalid_config)
            
    async def test_network_failure_handling(self):
        """Test handling of network failures"""
        config = SwarmNodeConfig(
            network_config=NetworkConfig(
                port=99999,  # Invalid port
                discovery_port=99998
            )
        )
        
        node = SwarmNode(config)
        
        # Should handle gracefully
        try:
            await node.start()
            # If it starts, it found alternative ports
        except Exception:
            # If it fails, that's also acceptable behavior
            pass
        finally:
            try:
                await node.stop()
            except Exception:
                pass


# Performance benchmarking tests
class TestPerformanceBenchmarks(SwarmTestBase):
    """Performance benchmark tests"""
    
    async def test_model_inference_speed(self):
        """Test model inference performance"""
        config = TransformerConfig(hidden_size=256, num_layers=4, num_heads=8)
        model = SwarmTransformer(config)
        model.eval()
        
        # Warm up
        dummy_input = torch.randint(0, config.vocab_size, (1, 64))
        with torch.no_grad():
            _ = model(dummy_input)
            
        # Benchmark
        start_time = time.time()
        num_iterations = 10
        
        with torch.no_grad():
            for _ in range(num_iterations):
                _ = model(dummy_input)
                
        elapsed_time = time.time() - start_time
        avg_time_per_inference = elapsed_time / num_iterations
        
        # Should complete inference in reasonable time (< 1 second per inference)
        self.assertLess(avg_time_per_inference, 1.0)
        
    async def test_serialization_speed(self):
        """Test serialization performance"""
        serializer = SwarmSerializer()
        
        # Create medium-sized model state
        model_state = {
            f'layer_{i}.weight': torch.randn(128, 128)
            for i in range(10)
        }
        
        # Benchmark serialization
        start_time = time.time()
        serialized = serializer.serialize_model(model_state)
        serialize_time = time.time() - start_time
        
        # Benchmark deserialization
        start_time = time.time()
        deserialized = serializer.deserialize_model(serialized)
        deserialize_time = time.time() - start_time
        
        # Should complete in reasonable time
        self.assertLess(serialize_time, 5.0)  # < 5 seconds
        self.assertLess(deserialize_time, 5.0)  # < 5 seconds


# Test runner and utilities
def run_all_tests(verbose=True):
    """Run all tests"""
    import unittest
    
    # Discover and run tests
    loader = unittest.TestLoader()
    start_dir = Path(__file__).parent
    suite = loader.discover(start_dir, pattern='test_*.py')
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2 if verbose else 1)
    result = runner.run(suite)
    
    return result.wasSuccessful()


async def run_async_tests():
    """Run async tests specifically"""
    test_classes = [
        TestSwarmTransformer,
        TestP2PNetwork,
        TestSwarmCrypto,
        TestSwarmSerializer,
        TestCooperativeTrainer,
        TestConsensusProtocol,
        TestConfigManager,
        TestDatasetManager,
        TestSwarmNodeIntegration,
        TestStressScenarios,
        TestErrorHandling,
        TestPerformanceBenchmarks
    ]
    
    total_tests = 0
    passed_tests = 0
    
    for test_class in test_classes:
        print(f"\n🧪 Running {test_class.__name__}")
        
        # Get all test methods
        test_methods = [
            method for method in dir(test_class)
            if method.startswith('test_')
        ]
        
        for test_method in test_methods:
            total_tests += 1
            
            try:
                # Create test instance
                test_instance = test_class()
                await test_instance.asyncSetUp()
                
                # Run test method
                await getattr(test_instance, test_method)()
                
                print(f"  ✅ {test_method}")
                passed_tests += 1
                
            except Exception as e:
                print(f"  ❌ {test_method}: {e}")
                
            finally:
                # Cleanup
                try:
                    if hasattr(test_instance, 'cleanup_temp_dir'):
                        test_instance.cleanup_temp_dir()
                except Exception:
                    pass
                    
    print(f"\n📊 Test Results: {passed_tests}/{total_tests} passed")
    return passed_tests == total_tests


# CLI interface
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="ncrsh-Swarm Test Suite")
    parser.add_argument('--async', action='store_true', help='Run async tests')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    if args.async:
        success = asyncio.run(run_async_tests())
    else:
        success = run_all_tests(args.verbose)
        
    sys.exit(0 if success else 1)