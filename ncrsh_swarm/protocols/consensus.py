"""
Consensus Protocol for SwarmNodes
"""

import asyncio
import logging
import time
from typing import Dict, List, Optional, Any, Tuple
import torch
import hashlib
import json


class ConsensusProtocol:
    """
    Implements consensus mechanisms for distributed neural network coordination
    
    Features:
    - Byzantine fault-tolerant model merging
    - Voting-based parameter consensus
    - Conflict resolution for model states
    - Leader election for training coordination
    """
    
    def __init__(self, network, fault_tolerance: float = 0.33):
        self.network = network
        self.node_id = network.node_id
        self.fault_tolerance = fault_tolerance  # Max fraction of Byzantine nodes
        self.logger = logging.getLogger(f"Consensus-{self.node_id[:8]}")
        
        # Consensus state
        self.current_epoch = 0
        self.consensus_round = 0
        self.votes: Dict[str, Dict[str, Any]] = {}
        self.proposals: Dict[str, Dict[str, Any]] = {}
        
        # Leader election
        self.current_leader: Optional[str] = None
        self.leader_heartbeat: Dict[str, float] = {}
        self.election_in_progress = False
        
    async def resolve_model_conflict(
        self, 
        local_state: Dict[str, torch.Tensor],
        peer_state: Dict[str, torch.Tensor],
        participants: List[str]
    ) -> Dict[str, torch.Tensor]:
        """
        Resolve conflicts between different model states using consensus
        """
        self.logger.info("Resolving model conflict via consensus")
        
        try:
            # Calculate hashes for both states
            local_hash = self._hash_model_state(local_state)
            peer_hash = self._hash_model_state(peer_state)
            
            if local_hash == peer_hash:
                return local_state  # No conflict
                
            # Initiate consensus round
            consensus_id = f"model_conflict_{self.consensus_round}"
            self.consensus_round += 1
            
            # Prepare proposals
            proposals = {
                'local': {
                    'state_hash': local_hash,
                    'proposer': self.node_id,
                    'state': local_state
                },
                'peer': {
                    'state_hash': peer_hash,
                    'proposer': participants[0] if participants else 'unknown',
                    'state': peer_state
                }
            }
            
            # Run consensus voting
            chosen_proposal = await self._run_consensus_vote(
                consensus_id, proposals, participants
            )
            
            if chosen_proposal:
                self.logger.info(f"Consensus reached: {chosen_proposal}")
                return proposals[chosen_proposal]['state']
            else:
                # Fallback: merge states
                self.logger.warning("Consensus failed, merging states")
                return await self._merge_model_states_fallback(local_state, peer_state)
                
        except Exception as e:
            self.logger.error(f"Consensus resolution failed: {e}")
            return local_state  # Conservative fallback
            
    async def merge_model_states(
        self,
        state1: Dict[str, torch.Tensor],
        state2: Dict[str, torch.Tensor],
        participants: List[str]
    ) -> Dict[str, torch.Tensor]:
        """
        Merge two model states using consensus-based averaging
        """
        try:
            # Simple parameter-wise averaging for compatible states
            merged_state = {}
            
            for param_name in state1.keys():
                if param_name in state2:
                    # Check tensor compatibility
                    if state1[param_name].shape == state2[param_name].shape:
                        # Weighted average based on participant count
                        weight1 = 1.0 / (len(participants) + 1)
                        weight2 = len(participants) / (len(participants) + 1)
                        
                        merged_state[param_name] = (
                            weight1 * state1[param_name] + 
                            weight2 * state2[param_name]
                        )
                    else:
                        # Shape mismatch, prefer local state
                        merged_state[param_name] = state1[param_name]
                        self.logger.warning(
                            f"Shape mismatch for {param_name}, using local state"
                        )
                else:
                    merged_state[param_name] = state1[param_name]
                    
            # Add any parameters only in state2
            for param_name in state2.keys():
                if param_name not in merged_state:
                    merged_state[param_name] = state2[param_name]
                    
            return merged_state
            
        except Exception as e:
            self.logger.error(f"State merging failed: {e}")
            return state1
            
    async def elect_leader(self, candidates: List[str]) -> Optional[str]:
        """
        Elect a leader node for coordinating training
        """
        if self.election_in_progress:
            self.logger.debug("Election already in progress")
            return self.current_leader
            
        self.election_in_progress = True
        
        try:
            self.logger.info(f"Starting leader election with candidates: {candidates}")
            
            # Include self as candidate
            all_candidates = list(set(candidates + [self.node_id]))
            
            # Run election based on node capabilities and reputation
            election_id = f"leader_election_{int(time.time())}"
            
            # Gather node information from all candidates
            candidate_info = {}
            
            for candidate in all_candidates:
                if candidate == self.node_id:
                    candidate_info[candidate] = await self._get_self_info()
                else:
                    try:
                        info = await self.network.send_message(
                            candidate, 'get_election_info', {}
                        )
                        candidate_info[candidate] = info
                    except Exception as e:
                        self.logger.warning(f"Failed to get info from {candidate}: {e}")
                        
            # Score candidates based on capabilities
            candidate_scores = {}
            for candidate, info in candidate_info.items():
                score = self._calculate_leadership_score(info)
                candidate_scores[candidate] = score
                
            # Sort by score (highest first)
            sorted_candidates = sorted(
                candidate_scores.items(), 
                key=lambda x: x[1], 
                reverse=True
            )
            
            # Vote for the best candidate
            preferred_candidate = sorted_candidates[0][0]
            
            # Run voting round
            votes = await self._collect_votes(election_id, preferred_candidate, all_candidates)
            
            # Count votes and determine winner
            vote_counts = {}
            for candidate in all_candidates:
                vote_counts[candidate] = sum(
                    1 for vote in votes.values() 
                    if vote.get('candidate') == candidate
                )
                
            # Winner needs majority
            total_votes = len(votes)
            required_votes = total_votes // 2 + 1
            
            winner = None
            for candidate, count in vote_counts.items():
                if count >= required_votes:
                    winner = candidate
                    break
                    
            if winner:
                self.current_leader = winner
                self.logger.info(f"Elected leader: {winner[:8]}")
            else:
                self.logger.warning("No majority winner in election")
                
            return winner
            
        except Exception as e:
            self.logger.error(f"Leader election failed: {e}")
            return None
        finally:
            self.election_in_progress = False
            
    async def coordinate_training_round(self, participants: List[str]) -> Dict[str, Any]:
        """
        Coordinate a distributed training round
        """
        if not self.current_leader:
            self.current_leader = await self.elect_leader(participants)
            
        if self.current_leader == self.node_id:
            # We are the leader, coordinate the round
            return await self._coordinate_as_leader(participants)
        else:
            # Follow the leader's coordination
            return await self._follow_leader_coordination()
            
    # Internal methods
    
    async def _run_consensus_vote(
        self, 
        consensus_id: str, 
        proposals: Dict[str, Dict[str, Any]], 
        participants: List[str]
    ) -> Optional[str]:
        """Run a consensus voting round"""
        try:
            # Vote for best proposal based on some criteria
            vote_preference = self._evaluate_proposals(proposals)
            
            # Collect votes from participants
            votes = await self._collect_votes(consensus_id, vote_preference, participants)
            
            # Count votes
            vote_counts = {}
            for proposal_id in proposals.keys():
                vote_counts[proposal_id] = sum(
                    1 for vote in votes.values() 
                    if vote.get('choice') == proposal_id
                )
                
            # Determine winner (simple majority)
            total_votes = len(votes)
            required_votes = total_votes // 2 + 1
            
            for proposal_id, count in vote_counts.items():
                if count >= required_votes:
                    return proposal_id
                    
            return None
            
        except Exception as e:
            self.logger.error(f"Consensus voting failed: {e}")
            return None
            
    async def _collect_votes(
        self, 
        vote_id: str, 
        preference: str, 
        participants: List[str]
    ) -> Dict[str, Dict[str, Any]]:
        """Collect votes from participants"""
        votes = {self.node_id: {'choice': preference, 'timestamp': time.time()}}
        
        # Request votes from participants
        tasks = []
        for participant in participants:
            task = asyncio.create_task(
                self._request_vote(participant, vote_id, preference)
            )
            tasks.append((participant, task))
            
        # Collect responses
        for participant, task in tasks:
            try:
                vote = await asyncio.wait_for(task, timeout=10.0)
                if vote:
                    votes[participant] = vote
            except Exception as e:
                self.logger.warning(f"Failed to get vote from {participant}: {e}")
                
        return votes
        
    async def _request_vote(
        self, 
        participant: str, 
        vote_id: str, 
        our_preference: str
    ) -> Optional[Dict[str, Any]]:
        """Request a vote from a participant"""
        try:
            response = await self.network.send_message(
                participant, 'consensus_vote_request', {
                    'vote_id': vote_id,
                    'our_preference': our_preference
                }
            )
            return response
        except Exception as e:
            self.logger.debug(f"Vote request to {participant} failed: {e}")
            return None
            
    def _hash_model_state(self, state: Dict[str, torch.Tensor]) -> str:
        """Calculate hash of model state"""
        # Create deterministic hash of model parameters
        hash_obj = hashlib.sha256()
        
        # Sort parameters by name for consistency
        for param_name in sorted(state.keys()):
            param_tensor = state[param_name]
            param_bytes = param_tensor.detach().cpu().numpy().tobytes()
            hash_obj.update(param_name.encode())
            hash_obj.update(param_bytes)
            
        return hash_obj.hexdigest()
        
    def _evaluate_proposals(self, proposals: Dict[str, Dict[str, Any]]) -> str:
        """Evaluate proposals and return preference"""
        # Simple heuristic: prefer proposal from node with higher ID
        # In practice, this could consider model performance, loss, etc.
        
        best_proposal = None
        best_score = -1
        
        for proposal_id, proposal in proposals.items():
            proposer = proposal.get('proposer', '')
            
            # Simple scoring based on proposer ID
            score = hash(proposer) % 1000
            
            if score > best_score:
                best_score = score
                best_proposal = proposal_id
                
        return best_proposal or list(proposals.keys())[0]
        
    async def _merge_model_states_fallback(
        self, 
        state1: Dict[str, torch.Tensor], 
        state2: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Fallback method for merging model states"""
        # Simple element-wise averaging
        merged = {}
        
        for param_name in state1.keys():
            if param_name in state2 and state1[param_name].shape == state2[param_name].shape:
                merged[param_name] = (state1[param_name] + state2[param_name]) / 2.0
            else:
                merged[param_name] = state1[param_name]
                
        return merged
        
    async def _get_self_info(self) -> Dict[str, Any]:
        """Get information about this node for leader election"""
        return {
            'node_id': self.node_id,
            'uptime': time.time(),  # Simplified uptime
            'peer_count': self.network.get_peer_count(),
            'capabilities': {
                'gpu_available': torch.cuda.is_available(),
                'gpu_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
                'memory_gb': 8,  # Simplified
            }
        }
        
    def _calculate_leadership_score(self, node_info: Dict[str, Any]) -> float:
        """Calculate leadership score for a node"""
        score = 0.0
        
        capabilities = node_info.get('capabilities', {})
        
        # GPU availability and count
        if capabilities.get('gpu_available', False):
            score += 10.0
            score += capabilities.get('gpu_count', 0) * 5.0
            
        # Memory
        score += capabilities.get('memory_gb', 0) * 0.5
        
        # Peer connections (network centrality)
        score += node_info.get('peer_count', 0) * 2.0
        
        # Uptime (stability)
        uptime = node_info.get('uptime', 0)
        score += min(uptime / 3600, 24) * 0.5  # Max 24 hours worth of points
        
        return score
        
    async def _coordinate_as_leader(self, participants: List[str]) -> Dict[str, Any]:
        """Coordinate training round as the leader"""
        self.logger.info("Coordinating training round as leader")
        
        coordination_plan = {
            'round_id': f"training_round_{self.current_epoch}_{self.consensus_round}",
            'participants': participants,
            'sync_interval': 30.0,  # seconds
            'batch_size': 32,
            'learning_rate': 0.001,
            'coordinator': self.node_id
        }
        
        # Broadcast coordination plan
        tasks = []
        for participant in participants:
            task = asyncio.create_task(
                self.network.send_message(
                    participant, 'training_coordination', coordination_plan
                )
            )
            tasks.append(task)
            
        # Wait for acknowledgments
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        
        successful_participants = []
        for i, response in enumerate(responses):
            if not isinstance(response, Exception):
                successful_participants.append(participants[i])
                
        coordination_plan['active_participants'] = successful_participants
        
        return coordination_plan
        
    async def _follow_leader_coordination(self) -> Dict[str, Any]:
        """Follow the leader's coordination"""
        # Wait for coordination message from leader
        # This would be handled by the message handler
        return {'status': 'following_leader', 'leader': self.current_leader}