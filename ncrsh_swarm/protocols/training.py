"""
Cooperative Training Protocol for SwarmNodes
"""

import asyncio
import logging
import time
from typing import Dict, List, Optional, Any, Tuple
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, DistributedSampler
import numpy as np

from ..utils.serialization import SwarmSerializer


class CooperativeTrainer:
    """
    Handles cooperative training across multiple SwarmNodes
    
    Features:
    - Federated learning with gradient averaging
    - Byzantine fault tolerance
    - Dynamic peer discovery during training
    - Asynchronous parameter updates
    - Adaptive learning rate scheduling
    """
    
    def __init__(self, model: nn.Module, network, learning_rate: float = 0.001):
        self.model = model
        self.network = network
        self.node_id = network.node_id
        self.logger = logging.getLogger(f"CooperativeTrainer-{self.node_id[:8]}")
        
        # Training state
        self.optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=1000)
        self.serializer = SwarmSerializer()
        
        # Cooperative training state
        self.training_round = 0
        self.peer_gradients: Dict[str, torch.Tensor] = {}
        self.peer_weights: Dict[str, float] = {}
        self.gradient_buffer: List[Dict[str, torch.Tensor]] = []
        
        # Byzantine fault tolerance
        self.peer_reputation: Dict[str, float] = {}
        self.gradient_history: Dict[str, List[float]] = {}
        
        # Performance tracking
        self.local_loss_history: List[float] = []
        self.global_loss_history: List[float] = []
        self.communication_overhead: List[float] = []
        
    async def train_cooperative(
        self, 
        dataset, 
        epochs: int = 100, 
        batch_size: int = 32,
        peers: Optional[List[str]] = None,
        sync_frequency: int = 5
    ) -> Dict[str, Any]:
        """
        Train the model cooperatively with peers
        
        Args:
            dataset: Training dataset
            epochs: Number of training epochs
            batch_size: Batch size for local training
            peers: List of peer IDs to train with
            sync_frequency: Sync with peers every N batches
        """
        self.logger.info(f"Starting cooperative training for {epochs} epochs")
        
        # Setup data loader
        if peers:
            # Use distributed sampler if we have peers
            sampler = DistributedSampler(
                dataset, 
                num_replicas=len(peers) + 1,
                rank=hash(self.node_id) % (len(peers) + 1)
            )
            dataloader = DataLoader(
                dataset, batch_size=batch_size, sampler=sampler
            )
        else:
            dataloader = DataLoader(
                dataset, batch_size=batch_size, shuffle=True
            )
            
        # Training metrics
        start_time = time.time()
        total_batches = len(dataloader) * epochs
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            num_batches = 0
            
            for batch_idx, (data, target) in enumerate(dataloader):
                # Local forward and backward pass
                loss = await self._local_training_step(data, target)
                epoch_loss += loss
                num_batches += 1
                
                # Sync with peers periodically
                if batch_idx % sync_frequency == 0 and peers:
                    await self._sync_with_peers(peers)
                    
                # Update learning rate
                self.scheduler.step()
                
            # End of epoch sync
            if peers:
                await self._sync_with_peers(peers)
                
            avg_epoch_loss = epoch_loss / num_batches
            self.local_loss_history.append(avg_epoch_loss)
            
            # Calculate global loss estimate
            if peers:
                global_loss = await self._estimate_global_loss(peers, avg_epoch_loss)
                self.global_loss_history.append(global_loss)
            else:
                self.global_loss_history.append(avg_epoch_loss)
                
            self.logger.info(
                f"Epoch {epoch+1}/{epochs}, "
                f"Local Loss: {avg_epoch_loss:.4f}, "
                f"Global Loss: {self.global_loss_history[-1]:.4f}, "
                f"Peers: {len(peers) if peers else 0}"
            )
            
            self.training_round += 1
            
        training_time = time.time() - start_time
        
        return {
            'epochs': epochs,
            'final_local_loss': self.local_loss_history[-1],
            'final_global_loss': self.global_loss_history[-1],
            'training_time': training_time,
            'peer_count': len(peers) if peers else 0,
            'total_batches': total_batches,
            'communication_overhead': sum(self.communication_overhead),
            'peer_reputation': self.peer_reputation.copy()
        }
        
    async def apply_peer_gradients(self, gradients: Dict[str, torch.Tensor], peer_id: str) -> None:
        """Apply gradients received from a peer"""
        try:
            # Update peer reputation based on gradient quality
            gradient_norm = self._calculate_gradient_norm(gradients)
            await self._update_peer_reputation(peer_id, gradient_norm)
            
            # Store gradients for averaging
            self.peer_gradients[peer_id] = gradients
            
            # Calculate weight based on reputation
            weight = self.peer_reputation.get(peer_id, 0.5)
            self.peer_weights[peer_id] = weight
            
            self.logger.debug(f"Applied gradients from {peer_id[:8]} with weight {weight:.3f}")
            
        except Exception as e:
            self.logger.error(f"Failed to apply gradients from {peer_id}: {e}")
            
    async def get_local_gradients(self) -> Dict[str, torch.Tensor]:
        """Get current local gradients"""
        gradients = {}
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                gradients[name] = param.grad.clone()
        return gradients
        
    async def broadcast_gradients(self, peers: List[str]) -> None:
        """Broadcast local gradients to peers"""
        if not peers:
            return
            
        start_time = time.time()
        
        try:
            # Get local gradients
            gradients = await self.get_local_gradients()
            
            # Serialize gradients
            serialized_gradients = self.serializer.serialize_gradients(gradients)
            
            # Broadcast to all peers
            tasks = []
            for peer_id in peers:
                task = asyncio.create_task(
                    self.network.send_message(
                        peer_id, 'gradient_share', 
                        {'gradients': serialized_gradients}
                    )
                )
                tasks.append(task)
                
            # Wait for all broadcasts to complete
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Track communication overhead
            comm_time = time.time() - start_time
            self.communication_overhead.append(comm_time)
            
            # Log results
            successful = sum(1 for r in results if not isinstance(r, Exception))
            self.logger.debug(
                f"Broadcast gradients to {successful}/{len(peers)} peers "
                f"in {comm_time:.3f}s"
            )
            
        except Exception as e:
            self.logger.error(f"Failed to broadcast gradients: {e}")
            
    # Internal methods
    
    async def _local_training_step(self, data: torch.Tensor, target: torch.Tensor) -> float:
        """Perform a local training step"""
        self.model.train()
        self.optimizer.zero_grad()
        
        # Forward pass
        output = self.model(data)
        loss = nn.functional.cross_entropy(output, target)
        
        # Backward pass
        loss.backward()
        
        # Apply aggregated gradients if available
        await self._apply_aggregated_gradients()
        
        # Optimizer step
        self.optimizer.step()
        
        return loss.item()
        
    async def _sync_with_peers(self, peers: List[str]) -> None:
        """Synchronize with peers"""
        if not peers:
            return
            
        try:
            # Broadcast our gradients
            await self.broadcast_gradients(peers)
            
            # Wait a bit for peer responses
            await asyncio.sleep(0.1)
            
            # Apply any accumulated peer gradients
            await self._apply_aggregated_gradients()
            
        except Exception as e:
            self.logger.error(f"Sync with peers failed: {e}")
            
    async def _apply_aggregated_gradients(self) -> None:
        """Apply aggregated gradients from peers"""
        if not self.peer_gradients:
            return
            
        try:
            # Calculate weighted average of peer gradients
            aggregated_gradients = await self._aggregate_gradients()
            
            # Apply to model parameters
            for name, param in self.model.named_parameters():
                if name in aggregated_gradients and param.grad is not None:
                    # Blend local and peer gradients
                    local_weight = 0.7  # Favor local gradients slightly
                    peer_weight = 0.3
                    
                    blended_grad = (
                        local_weight * param.grad + 
                        peer_weight * aggregated_gradients[name]
                    )
                    param.grad.copy_(blended_grad)
                    
            # Clear peer gradients buffer
            self.peer_gradients.clear()
            self.peer_weights.clear()
            
        except Exception as e:
            self.logger.error(f"Failed to apply aggregated gradients: {e}")
            
    async def _aggregate_gradients(self) -> Dict[str, torch.Tensor]:
        """Aggregate gradients from peers using weighted averaging"""
        if not self.peer_gradients:
            return {}
            
        # Normalize weights
        total_weight = sum(self.peer_weights.values())
        if total_weight == 0:
            return {}
            
        normalized_weights = {
            peer_id: weight / total_weight 
            for peer_id, weight in self.peer_weights.items()
        }
        
        # Get parameter names from first peer
        first_peer = next(iter(self.peer_gradients.keys()))
        param_names = list(self.peer_gradients[first_peer].keys())
        
        aggregated = {}
        
        for param_name in param_names:
            # Collect gradients for this parameter from all peers
            peer_grads = []
            weights = []
            
            for peer_id, gradients in self.peer_gradients.items():
                if param_name in gradients:
                    peer_grads.append(gradients[param_name])
                    weights.append(normalized_weights[peer_id])
                    
            if peer_grads:
                # Byzantine fault tolerance: remove outliers
                filtered_grads, filtered_weights = await self._filter_byzantine_gradients(
                    peer_grads, weights
                )
                
                # Weighted average
                if filtered_grads:
                    stacked_grads = torch.stack(filtered_grads)
                    weight_tensor = torch.tensor(filtered_weights, device=stacked_grads.device)
                    weight_tensor = weight_tensor / weight_tensor.sum()
                    
                    aggregated[param_name] = torch.sum(
                        stacked_grads * weight_tensor.unsqueeze(-1), dim=0
                    )
                    
        return aggregated
        
    async def _filter_byzantine_gradients(
        self, 
        gradients: List[torch.Tensor], 
        weights: List[float],
        outlier_threshold: float = 2.0
    ) -> Tuple[List[torch.Tensor], List[float]]:
        """Filter out Byzantine (malicious/faulty) gradients"""
        if len(gradients) <= 2:
            return gradients, weights
            
        try:
            # Calculate gradient norms
            norms = [torch.norm(grad).item() for grad in gradients]
            
            # Calculate statistics
            mean_norm = np.mean(norms)
            std_norm = np.std(norms)
            
            # Filter outliers
            filtered_grads = []
            filtered_weights = []
            
            for i, (grad, weight, norm) in enumerate(zip(gradients, weights, norms)):
                # Check if gradient is within acceptable range
                if abs(norm - mean_norm) <= outlier_threshold * std_norm:
                    filtered_grads.append(grad)
                    filtered_weights.append(weight)
                else:
                    self.logger.warning(f"Filtered Byzantine gradient with norm {norm:.4f}")
                    
            return filtered_grads, filtered_weights
            
        except Exception as e:
            self.logger.error(f"Byzantine filtering failed: {e}")
            return gradients, weights
            
    async def _estimate_global_loss(self, peers: List[str], local_loss: float) -> float:
        """Estimate global loss across the swarm"""
        try:
            # Request loss from peers
            peer_losses = []
            
            tasks = []
            for peer_id in peers:
                task = asyncio.create_task(
                    self.network.send_message(peer_id, 'get_current_loss', {})
                )
                tasks.append(task)
                
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for result in results:
                if isinstance(result, dict) and 'loss' in result:
                    peer_losses.append(result['loss'])
                    
            # Calculate weighted average
            all_losses = [local_loss] + peer_losses
            return np.mean(all_losses)
            
        except Exception as e:
            self.logger.error(f"Failed to estimate global loss: {e}")
            return local_loss
            
    def _calculate_gradient_norm(self, gradients: Dict[str, torch.Tensor]) -> float:
        """Calculate the norm of gradients"""
        total_norm = 0.0
        for grad in gradients.values():
            total_norm += torch.norm(grad).item() ** 2
        return total_norm ** 0.5
        
    async def _update_peer_reputation(self, peer_id: str, gradient_norm: float) -> None:
        """Update reputation of a peer based on gradient quality"""
        # Initialize reputation if new peer
        if peer_id not in self.peer_reputation:
            self.peer_reputation[peer_id] = 0.5
            self.gradient_history[peer_id] = []
            
        # Add to gradient history
        self.gradient_history[peer_id].append(gradient_norm)
        
        # Keep only recent history
        if len(self.gradient_history[peer_id]) > 100:
            self.gradient_history[peer_id] = self.gradient_history[peer_id][-100:]
            
        # Calculate reputation based on gradient consistency
        history = self.gradient_history[peer_id]
        if len(history) >= 5:
            # Use coefficient of variation (std/mean) as consistency measure
            mean_norm = np.mean(history)
            std_norm = np.std(history)
            
            if mean_norm > 0:
                consistency = 1.0 - min(1.0, std_norm / mean_norm)
                
                # Update reputation with momentum
                momentum = 0.9
                self.peer_reputation[peer_id] = (
                    momentum * self.peer_reputation[peer_id] + 
                    (1 - momentum) * consistency
                )
            
        # Ensure reputation stays in valid range
        self.peer_reputation[peer_id] = max(0.0, min(1.0, self.peer_reputation[peer_id]))