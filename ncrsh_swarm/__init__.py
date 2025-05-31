"""
ncrsh-Swarm: Neural Network Framework That Self-Clones Across Systems
====================================================================

A distributed neural network framework combining P2P networking with cooperative learning.

Example Usage:
```python
import ncrsh_swarm as swarm

# Create a swarm node
node = swarm.SwarmNode(
    model_config=swarm.TransformerConfig(hidden_size=512, num_layers=6),
    network_config=swarm.NetworkConfig(port=8080, max_peers=10)
)

# Start the swarm
await node.start()

# Train cooperatively
await node.train(dataset, epochs=100)
```
"""

__version__ = "0.1.0"
__author__ = "ncrsh-Swarm Contributors"

# Core exports
from .core.swarm_node import SwarmNode
from .core.swarm_manager import SwarmManager
from .models.transformer import TransformerConfig, SwarmTransformer
from .network.p2p import NetworkConfig, P2PNetwork
from .protocols.training import CooperativeTrainer
from .protocols.consensus import ConsensusProtocol

# Convenience imports
from .core.config import SwarmConfig
from .utils.crypto import SwarmCrypto
from .utils.serialization import SwarmSerializer

__all__ = [
    'SwarmNode',
    'SwarmManager', 
    'SwarmConfig',
    'TransformerConfig',
    'SwarmTransformer',
    'NetworkConfig',
    'P2PNetwork',
    'CooperativeTrainer',
    'ConsensusProtocol',
    'SwarmCrypto',
    'SwarmSerializer',
]