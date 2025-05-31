#!/usr/bin/env python3
"""
Node Management API v1
====================

REST API endpoints for managing swarm nodes.

By Dineth Nethsara (@hexlorddev)
"""

from fastapi import APIRouter, HTTPException, Depends, Query
from fastapi.security import HTTPBearer
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime
import asyncio
import logging

from ncrsh_swarm.core.swarm_node import SwarmNode, SwarmNodeConfig
from ncrsh_swarm.models.transformer import TransformerConfig
from ncrsh_swarm.network.p2p import NetworkConfig
from ncrsh_swarm.utils.auth import verify_token
from ncrsh_swarm.utils.monitoring import MetricsCollector

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/nodes", tags=["nodes"])
security = HTTPBearer()

# Pydantic Models
class NodePerformance(BaseModel):
    training_speed: float = Field(..., ge=0.0, le=1.0, description="Training speed ratio")
    memory_usage: float = Field(..., ge=0.0, le=1.0, description="Memory usage ratio") 
    network_latency: float = Field(..., ge=0.0, description="Network latency in ms")
    gpu_utilization: Optional[float] = Field(None, ge=0.0, le=1.0)
    throughput: float = Field(..., ge=0.0, description="Samples/second")

class NodeInfo(BaseModel):
    id: str = Field(..., description="Unique node identifier")
    address: str = Field(..., description="Network address")
    status: str = Field(..., description="Node status", regex="^(active|inactive|training|syncing)$")
    model_hash: str = Field(..., description="SHA256 hash of current model")
    last_seen: datetime = Field(..., description="Last activity timestamp")
    performance: NodePerformance
    version: str = Field(..., description="Node software version")
    capabilities: Dict[str, Any] = Field(default_factory=dict)

class NodeCreateRequest(BaseModel):
    name: Optional[str] = Field(None, description="Human-readable node name")
    config: Dict[str, Any] = Field(..., description="Node configuration")
    auto_start: bool = Field(True, description="Start node immediately")
    tags: List[str] = Field(default_factory=list, description="Node tags")

class NodeUpdateRequest(BaseModel):
    name: Optional[str] = None
    config: Optional[Dict[str, Any]] = None
    tags: Optional[List[str]] = None

class NodeListResponse(BaseModel):
    nodes: List[NodeInfo]
    total: int
    active: int
    inactive: int
    training: int

class NodeMetrics(BaseModel):
    cpu_usage: float
    memory_usage: float
    gpu_usage: Optional[float]
    network_io: Dict[str, float]
    training_metrics: Dict[str, float]
    timestamp: datetime

# Global node registry
node_registry: Dict[str, SwarmNode] = {}
metrics_collector = MetricsCollector()

@router.get("/", response_model=NodeListResponse)
async def list_nodes(
    status: Optional[str] = Query(None, regex="^(active|inactive|training|syncing)$"),
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    token: str = Depends(security)
):
    """
    List all nodes in the swarm with optional filtering.
    
    - **status**: Filter by node status
    - **limit**: Maximum number of nodes to return
    - **offset**: Number of nodes to skip
    """
    await verify_token(token.credentials)
    
    # Get all nodes
    all_nodes = []
    for node_id, node in node_registry.items():
        try:
            performance = await _get_node_performance(node)
            node_info = NodeInfo(
                id=node_id,
                address=f"{node.config.network.host}:{node.config.network.port}",
                status=await _get_node_status(node),
                model_hash=await _get_model_hash(node),
                last_seen=datetime.utcnow(),
                performance=performance,
                version=node.version,
                capabilities=await _get_node_capabilities(node)
            )
            all_nodes.append(node_info)
        except Exception as e:
            logger.error(f"Error getting info for node {node_id}: {e}")
    
    # Filter by status if requested
    if status:
        all_nodes = [n for n in all_nodes if n.status == status]
    
    # Apply pagination
    total = len(all_nodes)
    nodes = all_nodes[offset:offset + limit]
    
    # Count by status
    status_counts = {"active": 0, "inactive": 0, "training": 0}
    for node in all_nodes:
        if node.status in status_counts:
            status_counts[node.status] += 1
    
    return NodeListResponse(
        nodes=nodes,
        total=total,
        active=status_counts["active"],
        inactive=status_counts["inactive"],
        training=status_counts["training"]
    )

@router.post("/", response_model=NodeInfo)
async def create_node(
    request: NodeCreateRequest,
    token: str = Depends(security)
):
    """
    Create a new swarm node with specified configuration.
    
    Configuration example:
    ```json
    {
        "model_config": {
            "hidden_size": 512,
            "num_layers": 6,
            "num_heads": 8
        },
        "network_config": {
            "port": 8080,
            "max_peers": 10
        }
    }
    ```
    """
    await verify_token(token.credentials)
    
    try:
        # Parse configuration
        model_config = TransformerConfig(**request.config.get("model_config", {}))
        network_config = NetworkConfig(**request.config.get("network_config", {}))
        
        # Create node configuration
        node_config = SwarmNodeConfig(
            model_config=model_config,
            network_config=network_config
        )
        
        # Create and register node
        node = SwarmNode(node_config)
        node_id = node.node_id
        node_registry[node_id] = node
        
        # Start node if requested
        if request.auto_start:
            await node.start()
        
        # Collect initial metrics
        performance = await _get_node_performance(node)
        
        logger.info(f"Created node {node_id} with config: {request.config}")
        
        return NodeInfo(
            id=node_id,
            address=f"{network_config.host}:{network_config.port}",
            status=await _get_node_status(node),
            model_hash=await _get_model_hash(node),
            last_seen=datetime.utcnow(),
            performance=performance,
            version=node.version,
            capabilities=await _get_node_capabilities(node)
        )
        
    except Exception as e:
        logger.error(f"Failed to create node: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/{node_id}", response_model=NodeInfo)
async def get_node(
    node_id: str,
    token: str = Depends(security)
):
    """Get detailed information about a specific node."""
    await verify_token(token.credentials)
    
    if node_id not in node_registry:
        raise HTTPException(status_code=404, detail="Node not found")
    
    node = node_registry[node_id]
    
    try:
        performance = await _get_node_performance(node)
        
        return NodeInfo(
            id=node_id,
            address=f"{node.config.network.host}:{node.config.network.port}",
            status=await _get_node_status(node),
            model_hash=await _get_model_hash(node),
            last_seen=datetime.utcnow(),
            performance=performance,
            version=node.version,
            capabilities=await _get_node_capabilities(node)
        )
    except Exception as e:
        logger.error(f"Error getting node {node_id}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.put("/{node_id}", response_model=NodeInfo)
async def update_node(
    node_id: str,
    request: NodeUpdateRequest,
    token: str = Depends(security)
):
    """Update node configuration or metadata."""
    await verify_token(token.credentials)
    
    if node_id not in node_registry:
        raise HTTPException(status_code=404, detail="Node not found")
    
    node = node_registry[node_id]
    
    try:
        # Update configuration if provided
        if request.config:
            await node.update_config(request.config)
        
        # Update metadata
        if request.name:
            node.name = request.name
        if request.tags:
            node.tags = request.tags
        
        performance = await _get_node_performance(node)
        
        logger.info(f"Updated node {node_id}")
        
        return NodeInfo(
            id=node_id,
            address=f"{node.config.network.host}:{node.config.network.port}",
            status=await _get_node_status(node),
            model_hash=await _get_model_hash(node),
            last_seen=datetime.utcnow(),
            performance=performance,
            version=node.version,
            capabilities=await _get_node_capabilities(node)
        )
        
    except Exception as e:
        logger.error(f"Failed to update node {node_id}: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@router.delete("/{node_id}")
async def delete_node(
    node_id: str,
    force: bool = Query(False, description="Force delete even if training"),
    token: str = Depends(security)
):
    """Remove node from the swarm."""
    await verify_token(token.credentials)
    
    if node_id not in node_registry:
        raise HTTPException(status_code=404, detail="Node not found")
    
    node = node_registry[node_id]
    
    try:
        # Check if node is training
        status = await _get_node_status(node)
        if status == "training" and not force:
            raise HTTPException(
                status_code=409, 
                detail="Node is currently training. Use force=true to delete anyway."
            )
        
        # Gracefully stop node
        await node.stop()
        
        # Remove from registry
        del node_registry[node_id]
        
        logger.info(f"Deleted node {node_id}")
        
        return {"message": f"Node {node_id} deleted successfully"}
        
    except Exception as e:
        logger.error(f"Failed to delete node {node_id}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.post("/{node_id}/start")
async def start_node(
    node_id: str,
    token: str = Depends(security)
):
    """Start a stopped node."""
    await verify_token(token.credentials)
    
    if node_id not in node_registry:
        raise HTTPException(status_code=404, detail="Node not found")
    
    node = node_registry[node_id]
    
    try:
        await node.start()
        logger.info(f"Started node {node_id}")
        return {"message": f"Node {node_id} started successfully"}
    except Exception as e:
        logger.error(f"Failed to start node {node_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/{node_id}/stop")
async def stop_node(
    node_id: str,
    graceful: bool = Query(True, description="Graceful shutdown"),
    token: str = Depends(security)
):
    """Stop a running node."""
    await verify_token(token.credentials)
    
    if node_id not in node_registry:
        raise HTTPException(status_code=404, detail="Node not found")
    
    node = node_registry[node_id]
    
    try:
        if graceful:
            await node.stop()
        else:
            await node.force_stop()
        
        logger.info(f"Stopped node {node_id}")
        return {"message": f"Node {node_id} stopped successfully"}
    except Exception as e:
        logger.error(f"Failed to stop node {node_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{node_id}/metrics", response_model=NodeMetrics)
async def get_node_metrics(
    node_id: str,
    token: str = Depends(security)
):
    """Get detailed performance metrics for a node."""
    await verify_token(token.credentials)
    
    if node_id not in node_registry:
        raise HTTPException(status_code=404, detail="Node not found")
    
    node = node_registry[node_id]
    
    try:
        metrics = await metrics_collector.collect_node_metrics(node)
        
        return NodeMetrics(
            cpu_usage=metrics["cpu_usage"],
            memory_usage=metrics["memory_usage"],
            gpu_usage=metrics.get("gpu_usage"),
            network_io=metrics["network_io"],
            training_metrics=metrics["training_metrics"],
            timestamp=datetime.utcnow()
        )
    except Exception as e:
        logger.error(f"Failed to get metrics for node {node_id}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.post("/{node_id}/clone")
async def clone_node(
    node_id: str,
    target_address: str = Query(..., description="Target system address"),
    token: str = Depends(security)
):
    """Clone node to another system."""
    await verify_token(token.credentials)
    
    if node_id not in node_registry:
        raise HTTPException(status_code=404, detail="Node not found")
    
    node = node_registry[node_id]
    
    try:
        new_node_id = await node.clone_to(target_address)
        logger.info(f"Cloned node {node_id} to {target_address}, new ID: {new_node_id}")
        
        return {
            "message": f"Node cloned successfully",
            "original_node": node_id,
            "new_node": new_node_id,
            "target_address": target_address
        }
    except Exception as e:
        logger.error(f"Failed to clone node {node_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Helper functions
async def _get_node_performance(node: SwarmNode) -> NodePerformance:
    """Get performance metrics for a node."""
    metrics = await metrics_collector.collect_node_metrics(node)
    
    return NodePerformance(
        training_speed=metrics.get("training_speed", 0.0),
        memory_usage=metrics.get("memory_usage", 0.0),
        network_latency=metrics.get("network_latency", 0.0),
        gpu_utilization=metrics.get("gpu_usage"),
        throughput=metrics.get("throughput", 0.0)
    )

async def _get_node_status(node: SwarmNode) -> str:
    """Get current status of a node."""
    if node.is_training:
        return "training"
    elif node.is_syncing:
        return "syncing"
    elif node.is_active:
        return "active"
    else:
        return "inactive"

async def _get_model_hash(node: SwarmNode) -> str:
    """Get SHA256 hash of node's current model."""
    return await node.get_model_hash()

async def _get_node_capabilities(node: SwarmNode) -> Dict[str, Any]:
    """Get node capabilities and features."""
    return {
        "gpu_available": node.has_gpu(),
        "model_type": node.model_type,
        "max_sequence_length": node.config.model.max_seq_len,
        "supported_protocols": node.supported_protocols,
        "byzantine_tolerance": node.config.training.byzantine_tolerance,
        "compression_support": node.supports_compression()
    }