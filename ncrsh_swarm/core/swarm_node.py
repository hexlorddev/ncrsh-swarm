"""
SwarmNode: Core distributed neural network node that can self-replicate
"""

import asyncio
import logging
import uuid
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass
import torch
import torch.nn as nn
from pathlib import Path

from ..network.p2p import P2PNetwork, NetworkConfig
from ..models.transformer import SwarmTransformer, TransformerConfig
from ..protocols.training import CooperativeTrainer
from ..protocols.consensus import ConsensusProtocol
from ..utils.crypto import SwarmCrypto
from ..utils.serialization import SwarmSerializer


@dataclass
class SwarmNodeConfig:
    """Configuration for a SwarmNode"""
    node_id: Optional[str] = None
    model_config: Optional[TransformerConfig] = None
    network_config: Optional[NetworkConfig] = None
    max_peers: int = 10
    sync_interval: float = 30.0  # seconds
    clone_threshold: int = 5  # clone when we have 5+ peers
    enable_auto_clone: bool = True
    data_dir: Optional[Path] = None


class SwarmNode:
    """
    A neural network node that can self-replicate across systems.
    
    Features:
    - P2P discovery and networking
    - Distributed model training
    - Self-replication to new nodes
    - Cooperative gradient sharing
    - Fault tolerance and recovery
    """
    
    def __init__(self, config: SwarmNodeConfig):
        self.config = config
        self.node_id = config.node_id or str(uuid.uuid4())
        self.logger = logging.getLogger(f"SwarmNode-{self.node_id[:8]}")
        
        # Initialize components
        self.model_config = config.model_config or TransformerConfig()
        self.network_config = config.network_config or NetworkConfig()
        
        self.model = SwarmTransformer(self.model_config)
        self.network = P2PNetwork(self.network_config, self.node_id)
        self.trainer = CooperativeTrainer(self.model, self.network)
        self.consensus = ConsensusProtocol(self.network)
        
        self.crypto = SwarmCrypto()
        self.serializer = SwarmSerializer()
        
        # State
        self.peers: Dict[str, Dict[str, Any]] = {}
        self.is_running = False
        self.training_active = False
        self.clone_requests: List[str] = []
        
        # Callbacks
        self.on_peer_joined: Optional[Callable] = None
        self.on_peer_left: Optional[Callable] = None
        self.on_model_updated: Optional[Callable] = None
        self.on_clone_request: Optional[Callable] = None
        
    async def start(self) -> None:
        """Start the swarm node"""
        if self.is_running:
            return
            
        self.logger.info(f"Starting SwarmNode {self.node_id[:8]}")
        
        # Start networking
        await self.network.start()
        
        # Register message handlers
        self.network.register_handler('model_update', self._handle_model_update)
        self.network.register_handler('gradient_share', self._handle_gradient_share)
        self.network.register_handler('clone_request', self._handle_clone_request)
        self.network.register_handler('peer_discovery', self._handle_peer_discovery)
        
        # Start background tasks
        asyncio.create_task(self._sync_loop())
        asyncio.create_task(self._discovery_loop())
        asyncio.create_task(self._clone_loop())
        
        self.is_running = True
        self.logger.info(f"SwarmNode {self.node_id[:8]} started successfully")
        
    async def stop(self) -> None:
        """Stop the swarm node"""
        if not self.is_running:
            return
            
        self.logger.info(f"Stopping SwarmNode {self.node_id[:8]}")
        
        self.is_running = False
        self.training_active = False
        
        # Stop networking
        await self.network.stop()
        
        self.logger.info(f"SwarmNode {self.node_id[:8]} stopped")
        
    async def train(self, dataset, epochs: int = 100, batch_size: int = 32) -> None:
        """Train the model cooperatively with peers"""
        if self.training_active:
            self.logger.warning("Training already active")
            return
            
        self.logger.info(f"Starting cooperative training for {epochs} epochs")
        self.training_active = True
        
        try:
            await self.trainer.train_cooperative(
                dataset=dataset,
                epochs=epochs,
                batch_size=batch_size,
                peers=list(self.peers.keys())
            )
        finally:
            self.training_active = False
            
    async def clone_to_node(self, target_address: str) -> bool:
        """Clone this node to a target address"""
        try:
            self.logger.info(f"Attempting to clone to {target_address}")
            
            # Package the node for cloning
            clone_package = await self._create_clone_package()
            
            # Send clone request
            response = await self.network.send_message(
                target_address, 'clone_deploy', clone_package
            )
            
            if response.get('success'):
                self.logger.info(f"Successfully cloned to {target_address}")
                return True
            else:
                self.logger.error(f"Clone failed: {response.get('error')}")
                return False
                
        except Exception as e:
            self.logger.error(f"Clone to {target_address} failed: {e}")
            return False
            
    async def discover_peers(self, max_peers: Optional[int] = None) -> List[str]:
        """Discover available peers in the swarm"""
        max_peers = max_peers or self.config.max_peers
        discovered = await self.network.discover_peers(max_peers)
        
        for peer_id in discovered:
            if peer_id not in self.peers:
                await self._add_peer(peer_id)
                
        return list(self.peers.keys())
        
    async def get_swarm_status(self) -> Dict[str, Any]:
        """Get current swarm status"""
        return {
            'node_id': self.node_id,
            'is_running': self.is_running,
            'training_active': self.training_active,
            'peer_count': len(self.peers),
            'peers': list(self.peers.keys()),
            'model_params': sum(p.numel() for p in self.model.parameters()),
            'network_address': self.network.address,
            'clone_requests': len(self.clone_requests)
        }
        
    # Internal methods
    
    async def _sync_loop(self) -> None:
        """Background sync with peers"""
        while self.is_running:
            try:
                await self._sync_with_peers()
                await asyncio.sleep(self.config.sync_interval)
            except Exception as e:
                self.logger.error(f"Sync loop error: {e}")
                await asyncio.sleep(5)
                
    async def _discovery_loop(self) -> None:
        """Background peer discovery"""
        while self.is_running:
            try:
                await self.discover_peers()
                await asyncio.sleep(60)  # Discover every minute
            except Exception as e:
                self.logger.error(f"Discovery loop error: {e}")
                await asyncio.sleep(10)
                
    async def _clone_loop(self) -> None:
        """Background auto-cloning"""
        while self.is_running:
            try:
                if (self.config.enable_auto_clone and 
                    len(self.peers) >= self.config.clone_threshold):
                    await self._attempt_auto_clone()
                await asyncio.sleep(300)  # Check every 5 minutes
            except Exception as e:
                self.logger.error(f"Clone loop error: {e}")
                await asyncio.sleep(30)
                
    async def _sync_with_peers(self) -> None:
        """Sync model state with peers"""
        if not self.peers:
            return
            
        # Get model hash for comparison
        model_hash = self.crypto.hash_model(self.model.state_dict())
        
        for peer_id in list(self.peers.keys()):
            try:
                peer_hash = await self.network.send_message(
                    peer_id, 'get_model_hash', {}
                )
                
                if peer_hash != model_hash:
                    # Models are different, sync needed
                    await self._sync_model_with_peer(peer_id)
                    
            except Exception as e:
                self.logger.warning(f"Sync with peer {peer_id[:8]} failed: {e}")
                await self._remove_peer(peer_id)
                
    async def _sync_model_with_peer(self, peer_id: str) -> None:
        """Sync model with a specific peer"""
        try:
            # Use consensus protocol to determine which model to use
            peer_model = await self.network.send_message(
                peer_id, 'get_model_state', {}
            )
            
            consensus_result = await self.consensus.resolve_model_conflict(
                self.model.state_dict(), peer_model, [peer_id]
            )
            
            if consensus_result != self.model.state_dict():
                self.model.load_state_dict(consensus_result)
                self.logger.info(f"Model updated via consensus with {peer_id[:8]}")
                
                if self.on_model_updated:
                    await self.on_model_updated(consensus_result)
                    
        except Exception as e:
            self.logger.error(f"Model sync with {peer_id[:8]} failed: {e}")
            
    async def _add_peer(self, peer_id: str) -> None:
        """Add a new peer"""
        if peer_id == self.node_id or peer_id in self.peers:
            return
            
        try:
            # Exchange node info
            peer_info = await self.network.send_message(
                peer_id, 'get_node_info', {}
            )
            
            self.peers[peer_id] = peer_info
            self.logger.info(f"Added peer {peer_id[:8]}")
            
            if self.on_peer_joined:
                await self.on_peer_joined(peer_id, peer_info)
                
        except Exception as e:
            self.logger.error(f"Failed to add peer {peer_id[:8]}: {e}")
            
    async def _remove_peer(self, peer_id: str) -> None:
        """Remove a peer"""
        if peer_id in self.peers:
            peer_info = self.peers.pop(peer_id)
            self.logger.info(f"Removed peer {peer_id[:8]}")
            
            if self.on_peer_left:
                await self.on_peer_left(peer_id, peer_info)
                
    async def _create_clone_package(self) -> Dict[str, Any]:
        """Create a package for cloning this node"""
        return {
            'source_node_id': self.node_id,
            'model_config': self.model_config.__dict__,
            'model_state': self.serializer.serialize_model(self.model.state_dict()),
            'network_config': self.network_config.__dict__,
            'swarm_config': {
                'max_peers': self.config.max_peers,
                'sync_interval': self.config.sync_interval,
                'enable_auto_clone': self.config.enable_auto_clone,
            },
            'peer_list': list(self.peers.keys()),
            'version': '0.1.0',
            'timestamp': asyncio.get_event_loop().time()
        }
        
    async def _attempt_auto_clone(self) -> None:
        """Attempt to auto-clone to a new node"""
        # Find potential clone targets (peers of peers)
        potential_targets = await self.network.discover_potential_nodes()
        
        for target in potential_targets:
            if await self.clone_to_node(target):
                self.logger.info(f"Auto-clone successful to {target}")
                break
                
    # Message handlers
    
    async def _handle_model_update(self, sender_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle model update from peer"""
        try:
            new_state = self.serializer.deserialize_model(data['model_state'])
            
            # Use consensus to merge
            merged_state = await self.consensus.merge_model_states(
                self.model.state_dict(), new_state, [sender_id]
            )
            
            self.model.load_state_dict(merged_state)
            
            if self.on_model_updated:
                await self.on_model_updated(merged_state)
                
            return {'success': True}
            
        except Exception as e:
            self.logger.error(f"Model update from {sender_id[:8]} failed: {e}")
            return {'success': False, 'error': str(e)}
            
    async def _handle_gradient_share(self, sender_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle gradient sharing from peer"""
        try:
            gradients = self.serializer.deserialize_gradients(data['gradients'])
            await self.trainer.apply_peer_gradients(gradients, sender_id)
            return {'success': True}
        except Exception as e:
            self.logger.error(f"Gradient share from {sender_id[:8]} failed: {e}")
            return {'success': False, 'error': str(e)}
            
    async def _handle_clone_request(self, sender_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle clone request from peer"""
        try:
            target_address = data.get('target_address')
            
            if self.on_clone_request:
                approved = await self.on_clone_request(sender_id, target_address)
            else:
                approved = True  # Default to approving clone requests
                
            if approved:
                success = await self.clone_to_node(target_address)
                return {'success': success}
            else:
                return {'success': False, 'error': 'Clone request denied'}
                
        except Exception as e:
            self.logger.error(f"Clone request from {sender_id[:8]} failed: {e}")
            return {'success': False, 'error': str(e)}
            
    async def _handle_peer_discovery(self, sender_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle peer discovery request"""
        return {
            'node_id': self.node_id,
            'peers': list(self.peers.keys()),
            'model_hash': self.crypto.hash_model(self.model.state_dict()),
            'training_active': self.training_active
        }