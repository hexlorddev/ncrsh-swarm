"""
P2P Network Layer for SwarmNodes
"""

import asyncio
import json
import logging
import socket
from typing import Dict, List, Optional, Any, Callable, Set
from dataclasses import dataclass, field
import struct
import time
import random

import asyncio_dgram
import aiohttp
from aiohttp import web
import websockets


@dataclass
class NetworkConfig:
    """Network configuration for P2P layer"""
    port: int = 8080
    discovery_port: int = 8081
    max_connections: int = 100
    timeout: float = 30.0
    enable_upnp: bool = True
    bootstrap_nodes: List[str] = field(default_factory=list)
    bind_address: str = "0.0.0.0"
    discovery_interval: float = 60.0


class P2PNetwork:
    """
    Peer-to-peer networking layer for SwarmNodes
    
    Features:
    - UDP broadcast discovery
    - WebSocket connections for reliable messaging
    - NAT traversal and hole punching
    - Automatic peer management
    - Message routing and delivery
    """
    
    def __init__(self, config: NetworkConfig, node_id: str):
        self.config = config
        self.node_id = node_id
        self.logger = logging.getLogger(f"P2PNetwork-{node_id[:8]}")
        
        # Network state
        self.is_running = False
        self.peers: Dict[str, Dict[str, Any]] = {}
        self.connections: Dict[str, websockets.WebSocketServerProtocol] = {}
        self.message_handlers: Dict[str, Callable] = {}
        
        # Servers
        self.http_server: Optional[web.Application] = None
        self.websocket_server = None
        self.discovery_socket = None
        
        # Discovery
        self.local_ip = self._get_local_ip()
        self.address = f"{self.local_ip}:{self.config.port}"
        self.discovered_nodes: Set[str] = set()
        
        # Message tracking
        self.message_id_counter = 0
        self.pending_responses: Dict[str, asyncio.Future] = {}
        
    async def start(self) -> None:
        """Start the P2P network"""
        if self.is_running:
            return
            
        self.logger.info(f"Starting P2P network on {self.address}")
        
        # Start HTTP/WebSocket server
        await self._start_websocket_server()
        
        # Start UDP discovery
        await self._start_discovery()
        
        # Bootstrap with known nodes
        for bootstrap_node in self.config.bootstrap_nodes:
            await self._connect_to_peer(bootstrap_node)
            
        self.is_running = True
        self.logger.info("P2P network started successfully")
        
    async def stop(self) -> None:
        """Stop the P2P network"""
        if not self.is_running:
            return
            
        self.logger.info("Stopping P2P network")
        self.is_running = False
        
        # Close all connections
        for peer_id, conn in list(self.connections.items()):
            try:
                await conn.close()
            except Exception:
                pass
                
        self.connections.clear()
        
        # Stop servers
        if self.websocket_server:
            self.websocket_server.close()
            await self.websocket_server.wait_closed()
            
        if self.discovery_socket:
            self.discovery_socket.close()
            
        self.logger.info("P2P network stopped")
        
    async def send_message(self, peer_id: str, message_type: str, data: Any) -> Any:
        """Send a message to a peer and wait for response"""
        if peer_id not in self.connections:
            raise ConnectionError(f"Not connected to peer {peer_id}")
            
        message_id = f"{self.node_id}_{self.message_id_counter}"
        self.message_id_counter += 1
        
        message = {
            'id': message_id,
            'sender': self.node_id,
            'type': message_type,
            'data': data,
            'timestamp': time.time()
        }
        
        # Create future for response
        response_future = asyncio.Future()
        self.pending_responses[message_id] = response_future
        
        try:
            # Send message
            conn = self.connections[peer_id]
            await conn.send(json.dumps(message))
            
            # Wait for response with timeout
            response = await asyncio.wait_for(
                response_future, timeout=self.config.timeout
            )
            return response
            
        except asyncio.TimeoutError:
            raise TimeoutError(f"Message to {peer_id} timed out")
        except Exception as e:
            raise ConnectionError(f"Failed to send message to {peer_id}: {e}")
        finally:
            self.pending_responses.pop(message_id, None)
            
    async def broadcast_message(self, message_type: str, data: Any) -> Dict[str, Any]:
        """Broadcast a message to all connected peers"""
        results = {}
        
        tasks = []
        for peer_id in list(self.connections.keys()):
            task = asyncio.create_task(
                self._safe_send_message(peer_id, message_type, data)
            )
            tasks.append((peer_id, task))
            
        for peer_id, task in tasks:
            try:
                result = await task
                results[peer_id] = result
            except Exception as e:
                self.logger.warning(f"Broadcast to {peer_id} failed: {e}")
                results[peer_id] = {'error': str(e)}
                
        return results
        
    async def discover_peers(self, max_peers: int = 10) -> List[str]:
        """Discover available peers"""
        # Send discovery broadcast
        await self._send_discovery_broadcast()
        
        # Wait a bit for responses
        await asyncio.sleep(2.0)
        
        # Try to connect to discovered nodes
        discovered = []
        for node_address in list(self.discovered_nodes):
            if len(discovered) >= max_peers:
                break
                
            try:
                peer_id = await self._connect_to_peer(node_address)
                if peer_id:
                    discovered.append(peer_id)
            except Exception as e:
                self.logger.debug(f"Failed to connect to {node_address}: {e}")
                
        return discovered
        
    async def discover_potential_nodes(self) -> List[str]:
        """Discover potential nodes for cloning (peers of peers)"""
        potential = set()
        
        # Ask all peers for their peer lists
        for peer_id in list(self.connections.keys()):
            try:
                response = await self.send_message(peer_id, 'get_peers', {})
                peer_list = response.get('peers', [])
                
                for peer_address in peer_list:
                    if peer_address != self.address:
                        potential.add(peer_address)
                        
            except Exception as e:
                self.logger.debug(f"Failed to get peers from {peer_id}: {e}")
                
        return list(potential)
        
    def register_handler(self, message_type: str, handler: Callable) -> None:
        """Register a message handler"""
        self.message_handlers[message_type] = handler
        
    def get_peer_count(self) -> int:
        """Get number of connected peers"""
        return len(self.connections)
        
    def get_peer_list(self) -> List[str]:
        """Get list of connected peer IDs"""
        return list(self.connections.keys())
        
    # Internal methods
    
    def _get_local_ip(self) -> str:
        """Get local IP address"""
        try:
            # Connect to a remote address to determine local IP
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.connect(("8.8.8.8", 80))
            local_ip = sock.getsockname()[0]
            sock.close()
            return local_ip
        except Exception:
            return "127.0.0.1"
            
    async def _start_websocket_server(self) -> None:
        """Start WebSocket server for peer connections"""
        async def websocket_handler(websocket, path):
            peer_id = None
            try:
                # Handle WebSocket connection
                await self._handle_websocket_connection(websocket)
            except Exception as e:
                self.logger.error(f"WebSocket handler error: {e}")
            finally:
                if peer_id and peer_id in self.connections:
                    del self.connections[peer_id]
                    
        self.websocket_server = await websockets.serve(
            websocket_handler,
            self.config.bind_address,
            self.config.port,
            max_size=10**7,  # 10MB max message size
            max_queue=None
        )
        
    async def _handle_websocket_connection(self, websocket) -> None:
        """Handle incoming WebSocket connection"""
        peer_id = None
        
        try:
            # Wait for handshake message
            handshake_msg = await asyncio.wait_for(
                websocket.recv(), timeout=10.0
            )
            handshake = json.loads(handshake_msg)
            
            if handshake.get('type') != 'handshake':
                raise ValueError("Expected handshake message")
                
            peer_id = handshake.get('node_id')
            if not peer_id or peer_id == self.node_id:
                raise ValueError("Invalid peer ID")
                
            # Send handshake response
            response = {
                'type': 'handshake_response',
                'node_id': self.node_id,
                'address': self.address,
                'status': 'connected'
            }
            await websocket.send(json.dumps(response))
            
            # Store connection
            self.connections[peer_id] = websocket
            self.peers[peer_id] = {
                'address': handshake.get('address'),
                'connected_at': time.time()
            }
            
            self.logger.info(f"Connected to peer {peer_id[:8]}")
            
            # Handle messages from this peer
            async for message in websocket:
                await self._handle_peer_message(peer_id, message)
                
        except websockets.exceptions.ConnectionClosed:
            pass
        except Exception as e:
            self.logger.error(f"WebSocket connection error: {e}")
        finally:
            if peer_id:
                self.connections.pop(peer_id, None)
                self.peers.pop(peer_id, None)
                self.logger.info(f"Disconnected from peer {peer_id[:8]}")
                
    async def _handle_peer_message(self, peer_id: str, raw_message: str) -> None:
        """Handle message from a peer"""
        try:
            message = json.loads(raw_message)
            message_id = message.get('id')
            message_type = message.get('type')
            data = message.get('data', {})
            
            # Check if this is a response to a pending request
            if message_id in self.pending_responses:
                future = self.pending_responses.pop(message_id)
                if not future.cancelled():
                    future.set_result(data)
                return
                
            # Handle new message
            response_data = None
            
            if message_type in self.message_handlers:
                handler = self.message_handlers[message_type]
                response_data = await handler(peer_id, data)
            else:
                self.logger.warning(f"No handler for message type: {message_type}")
                response_data = {'error': f'Unknown message type: {message_type}'}
                
            # Send response
            if message_id:
                response = {
                    'id': message_id,
                    'sender': self.node_id,
                    'type': f"{message_type}_response",
                    'data': response_data,
                    'timestamp': time.time()
                }
                
                conn = self.connections.get(peer_id)
                if conn:
                    await conn.send(json.dumps(response))
                    
        except Exception as e:
            self.logger.error(f"Error handling message from {peer_id}: {e}")
            
    async def _connect_to_peer(self, address: str) -> Optional[str]:
        """Connect to a peer at the given address"""
        if address == self.address:
            return None
            
        try:
            # Parse address
            if ':' not in address:
                address = f"{address}:8080"
                
            host, port = address.split(':')
            uri = f"ws://{host}:{port}"
            
            # Connect via WebSocket
            websocket = await websockets.connect(
                uri, timeout=10.0, max_size=10**7
            )
            
            # Send handshake
            handshake = {
                'type': 'handshake',
                'node_id': self.node_id,
                'address': self.address
            }
            await websocket.send(json.dumps(handshake))
            
            # Wait for handshake response
            response_msg = await asyncio.wait_for(
                websocket.recv(), timeout=10.0
            )
            response = json.loads(response_msg)
            
            if response.get('type') != 'handshake_response':
                raise ValueError("Invalid handshake response")
                
            peer_id = response.get('node_id')
            if not peer_id or peer_id == self.node_id:
                raise ValueError("Invalid peer ID in response")
                
            # Store connection
            self.connections[peer_id] = websocket
            self.peers[peer_id] = {
                'address': address,
                'connected_at': time.time()
            }
            
            self.logger.info(f"Connected to peer {peer_id[:8]} at {address}")
            
            # Start message handling for this connection
            asyncio.create_task(self._handle_peer_connection(peer_id, websocket))
            
            return peer_id
            
        except Exception as e:
            self.logger.debug(f"Failed to connect to {address}: {e}")
            return None
            
    async def _handle_peer_connection(self, peer_id: str, websocket) -> None:
        """Handle ongoing connection with a peer"""
        try:
            async for message in websocket:
                await self._handle_peer_message(peer_id, message)
        except websockets.exceptions.ConnectionClosed:
            pass
        except Exception as e:
            self.logger.error(f"Error in peer connection {peer_id}: {e}")
        finally:
            self.connections.pop(peer_id, None)
            self.peers.pop(peer_id, None)
            self.logger.info(f"Disconnected from peer {peer_id[:8]}")
            
    async def _start_discovery(self) -> None:
        """Start UDP discovery service"""
        try:
            # Create UDP socket for discovery
            self.discovery_socket = await asyncio_dgram.bind(
                (self.config.bind_address, self.config.discovery_port)
            )
            
            # Start discovery listener
            asyncio.create_task(self._discovery_listener())
            
        except Exception as e:
            self.logger.warning(f"Failed to start discovery: {e}")
            
    async def _discovery_listener(self) -> None:
        """Listen for discovery broadcasts"""
        while self.is_running and self.discovery_socket:
            try:
                data, addr = await self.discovery_socket.recv()
                await self._handle_discovery_message(data, addr)
            except Exception as e:
                if self.is_running:
                    self.logger.debug(f"Discovery listener error: {e}")
                    
    async def _handle_discovery_message(self, data: bytes, addr: tuple) -> None:
        """Handle discovery message"""
        try:
            message = json.loads(data.decode())
            
            if message.get('type') == 'discovery':
                node_id = message.get('node_id')
                node_address = message.get('address')
                
                if node_id != self.node_id and node_address:
                    self.discovered_nodes.add(node_address)
                    
                    # Send discovery response
                    response = {
                        'type': 'discovery_response',
                        'node_id': self.node_id,
                        'address': self.address
                    }
                    
                    response_data = json.dumps(response).encode()
                    await self.discovery_socket.send(response_data, addr)
                    
            elif message.get('type') == 'discovery_response':
                node_address = message.get('address')
                if node_address and node_address != self.address:
                    self.discovered_nodes.add(node_address)
                    
        except Exception as e:
            self.logger.debug(f"Discovery message error: {e}")
            
    async def _send_discovery_broadcast(self) -> None:
        """Send discovery broadcast"""
        if not self.discovery_socket:
            return
            
        message = {
            'type': 'discovery',
            'node_id': self.node_id,
            'address': self.address,
            'timestamp': time.time()
        }
        
        data = json.dumps(message).encode()
        
        # Broadcast to local network
        broadcast_addresses = [
            ('255.255.255.255', self.config.discovery_port),
            ('127.0.0.1', self.config.discovery_port),
        ]
        
        for addr in broadcast_addresses:
            try:
                await self.discovery_socket.send(data, addr)
            except Exception as e:
                self.logger.debug(f"Failed to send discovery to {addr}: {e}")
                
    async def _safe_send_message(self, peer_id: str, message_type: str, data: Any) -> Any:
        """Safely send message to peer"""
        try:
            return await self.send_message(peer_id, message_type, data)
        except Exception as e:
            self.logger.warning(f"Failed to send {message_type} to {peer_id}: {e}")
            # Remove failed connection
            self.connections.pop(peer_id, None)
            self.peers.pop(peer_id, None)
            raise