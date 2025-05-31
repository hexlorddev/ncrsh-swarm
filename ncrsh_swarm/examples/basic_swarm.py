"""
Basic example of creating and running a swarm
"""

import asyncio
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset

import ncrsh_swarm as swarm


async def main():
    """Demonstrate basic swarm functionality"""
    
    # Create a simple dataset for demonstration
    print("🔬 Creating synthetic dataset...")
    
    # Generate some random data for language modeling
    vocab_size = 1000
    seq_len = 64
    num_samples = 1000
    
    # Random token sequences
    input_data = torch.randint(0, vocab_size, (num_samples, seq_len))
    target_data = torch.roll(input_data, shifts=-1, dims=1)  # Next token prediction
    
    dataset = TensorDataset(input_data, target_data)
    
    print(f"Dataset: {num_samples} samples, sequence length {seq_len}")
    
    # Create swarm configuration
    print("\n⚙️ Configuring swarm...")
    
    model_config = swarm.TransformerConfig(
        vocab_size=vocab_size,
        max_seq_len=seq_len,
        hidden_size=256,
        num_layers=4,
        num_heads=8,
        dropout=0.1
    )
    
    network_config = swarm.NetworkConfig(
        port=8080,
        discovery_port=8081,
        max_connections=50
    )
    
    node_config = swarm.SwarmNodeConfig(
        model_config=model_config,
        network_config=network_config,
        max_peers=5,
        sync_interval=10.0,
        enable_auto_clone=True
    )
    
    # Create the first node
    print("🚀 Starting swarm node...")
    node1 = swarm.SwarmNode(node_config)
    
    # Set up event handlers
    node1.on_peer_joined = lambda peer_id, info: print(f"✅ Peer joined: {peer_id[:8]}")
    node1.on_peer_left = lambda peer_id, info: print(f"❌ Peer left: {peer_id[:8]}")
    node1.on_model_updated = lambda state: print("🔄 Model updated via consensus")
    
    try:
        # Start the node
        await node1.start()
        
        # Display initial status
        status = await node1.get_swarm_status()
        print(f"\n📊 Node Status:")
        print(f"   ID: {status['node_id'][:8]}")
        print(f"   Address: {status['network_address']}")
        print(f"   Model params: {status['model_params']:,}")
        print(f"   Peers: {status['peer_count']}")
        
        # Try to discover other nodes
        print("\n🔍 Discovering peers...")
        discovered_peers = await node1.discover_peers(max_peers=3)
        
        if discovered_peers:
            print(f"Found {len(discovered_peers)} peers: {[p[:8] for p in discovered_peers]}")
        else:
            print("No peers found. This node will run solo.")
        
        # Start training
        print("\n🎓 Starting cooperative training...")
        
        training_result = await node1.train(
            dataset=dataset,
            epochs=5,  # Short demo
            batch_size=16
        )
        
        print(f"\n🏆 Training completed!")
        print(f"   Final loss: {training_result['final_local_loss']:.4f}")
        print(f"   Training time: {training_result['training_time']:.2f}s")
        print(f"   Trained with {training_result['peer_count']} peers")
        
        # Test model generation
        print("\n🎯 Testing model generation...")
        
        # Create a simple input sequence
        test_input = torch.randint(0, vocab_size, (1, 20))
        
        with torch.no_grad():
            generated = node1.model.generate(
                test_input, 
                max_new_tokens=10,
                temperature=0.8
            )
        
        print(f"Input tokens: {test_input[0].tolist()}")
        print(f"Generated: {generated[0].tolist()}")
        
        # Show final swarm status
        final_status = await node1.get_swarm_status()
        print(f"\n📈 Final swarm status:")
        print(f"   Peers connected: {final_status['peer_count']}")
        print(f"   Clone requests: {final_status['clone_requests']}")
        
        print("\n✨ Demo completed successfully!")
        print("The swarm node is still running. Press Ctrl+C to stop.")
        
        # Keep running for demonstration
        while True:
            await asyncio.sleep(30)
            
            # Periodic status update
            status = await node1.get_swarm_status()
            print(f"💓 Heartbeat - Peers: {status['peer_count']}, Training: {status['training_active']}")
        
    except KeyboardInterrupt:
        print("\n🛑 Stopping swarm node...")
        await node1.stop()
        print("✅ Node stopped successfully")
    
    except Exception as e:
        print(f"\n❌ Error: {e}")
        await node1.stop()


if __name__ == "__main__":
    print("🧠🌐 ncrsh-Swarm Basic Example")
    print("=" * 40)
    asyncio.run(main())