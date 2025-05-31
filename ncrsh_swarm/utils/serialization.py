"""
Serialization utilities for SwarmNodes
"""

import io
import pickle
import gzip
import time
from typing import Dict, Any, List, Optional, Union
import torch
import numpy as np
import msgpack
import base64


class SwarmSerializer:
    """
    Efficient serialization for neural network models and gradients
    
    Features:
    - Compressed model state serialization
    - Gradient compression and quantization
    - Delta compression for incremental updates
    - Cross-platform compatibility
    """
    
    def __init__(self, compression_level: int = 6, use_quantization: bool = True):
        self.compression_level = compression_level
        self.use_quantization = use_quantization
        
    def serialize_model(self, model_state: Dict[str, torch.Tensor]) -> bytes:
        """
        Serialize a model state dictionary
        
        Args:
            model_state: Dictionary of model parameters
            
        Returns:
            Compressed serialized bytes
        """
        # Convert tensors to numpy arrays for better compression
        numpy_state = {}
        metadata = {
            'shapes': {},
            'dtypes': {},
            'device_types': {}
        }
        
        for name, tensor in model_state.items():
            # Store metadata
            metadata['shapes'][name] = list(tensor.shape)
            metadata['dtypes'][name] = str(tensor.dtype)
            metadata['device_types'][name] = tensor.device.type
            
            # Convert to numpy
            numpy_state[name] = tensor.detach().cpu().numpy()
            
        # Package data
        package = {
            'state': numpy_state,
            'metadata': metadata,
            'version': '0.1.0',
            'timestamp': time.time()
        }
        
        # Serialize with pickle
        serialized = pickle.dumps(package, protocol=pickle.HIGHEST_PROTOCOL)
        
        # Compress
        compressed = gzip.compress(serialized, compresslevel=self.compression_level)
        
        return compressed
        
    def deserialize_model(self, data: bytes) -> Dict[str, torch.Tensor]:
        """
        Deserialize a model state dictionary
        
        Args:
            data: Compressed serialized bytes
            
        Returns:
            Model state dictionary
        """
        # Decompress
        decompressed = gzip.decompress(data)
        
        # Deserialize
        package = pickle.loads(decompressed)
        
        # Extract data
        numpy_state = package['state']
        metadata = package['metadata']
        
        # Convert back to tensors
        model_state = {}
        for name, array in numpy_state.items():
            # Get original dtype
            dtype_str = metadata['dtypes'][name]
            dtype = getattr(torch, dtype_str.split('.')[-1])
            
            # Create tensor
            tensor = torch.from_numpy(array).to(dtype)
            model_state[name] = tensor
            
        return model_state
        
    def serialize_gradients(
        self, 
        gradients: Dict[str, torch.Tensor],
        quantize: Optional[bool] = None
    ) -> bytes:
        """
        Serialize gradients with optional quantization
        
        Args:
            gradients: Dictionary of gradients
            quantize: Whether to quantize gradients (uses instance default if None)
            
        Returns:
            Compressed serialized gradient bytes
        """
        should_quantize = quantize if quantize is not None else self.use_quantization
        
        # Process gradients
        processed_grads = {}
        metadata = {
            'shapes': {},
            'dtypes': {},
            'quantized': {},
            'scales': {},
            'zero_points': {}
        }
        
        for name, grad in gradients.items():
            metadata['shapes'][name] = list(grad.shape)
            metadata['dtypes'][name] = str(grad.dtype)
            
            if should_quantize and grad.numel() > 1000:  # Only quantize large gradients
                # Quantize to int8
                quantized_grad, scale, zero_point = self._quantize_tensor(grad)
                processed_grads[name] = quantized_grad.cpu().numpy()
                metadata['quantized'][name] = True
                metadata['scales'][name] = scale
                metadata['zero_points'][name] = zero_point
            else:
                # Keep as float
                processed_grads[name] = grad.detach().cpu().numpy()
                metadata['quantized'][name] = False
                
        # Package
        package = {
            'gradients': processed_grads,
            'metadata': metadata,
            'version': '0.1.0',
            'timestamp': time.time()
        }
        
        # Serialize and compress
        serialized = pickle.dumps(package, protocol=pickle.HIGHEST_PROTOCOL)
        compressed = gzip.compress(serialized, compresslevel=self.compression_level)
        
        return compressed
        
    def deserialize_gradients(self, data: bytes) -> Dict[str, torch.Tensor]:
        """
        Deserialize gradients
        
        Args:
            data: Compressed serialized gradient bytes
            
        Returns:
            Dictionary of gradients
        """
        # Decompress and deserialize
        decompressed = gzip.decompress(data)
        package = pickle.loads(decompressed)
        
        # Extract data
        processed_grads = package['gradients']
        metadata = package['metadata']
        
        # Convert back to tensors
        gradients = {}
        for name, array in processed_grads.items():
            if metadata['quantized'][name]:
                # Dequantize
                quantized_tensor = torch.from_numpy(array)
                scale = metadata['scales'][name]
                zero_point = metadata['zero_points'][name]
                
                grad = self._dequantize_tensor(quantized_tensor, scale, zero_point)
            else:
                # Convert directly
                grad = torch.from_numpy(array)
                
            # Restore original dtype
            dtype_str = metadata['dtypes'][name]
            dtype = getattr(torch, dtype_str.split('.')[-1])
            gradients[name] = grad.to(dtype)
            
        return gradients
        
    def serialize_delta(
        self, 
        old_state: Dict[str, torch.Tensor], 
        new_state: Dict[str, torch.Tensor]
    ) -> bytes:
        """
        Serialize only the differences between two model states
        
        Args:
            old_state: Previous model state
            new_state: Current model state
            
        Returns:
            Compressed delta bytes
        """
        deltas = {}
        metadata = {
            'changed_params': [],
            'shapes': {},
            'dtypes': {}
        }
        
        # Calculate deltas for changed parameters
        for name in new_state.keys():
            if name in old_state:
                old_param = old_state[name]
                new_param = new_state[name]
                
                if not torch.equal(old_param, new_param):
                    delta = new_param - old_param
                    deltas[name] = delta.detach().cpu().numpy()
                    metadata['changed_params'].append(name)
                    metadata['shapes'][name] = list(delta.shape)
                    metadata['dtypes'][name] = str(delta.dtype)
            else:
                # New parameter
                new_param = new_state[name]
                deltas[name] = new_param.detach().cpu().numpy()
                metadata['changed_params'].append(name)
                metadata['shapes'][name] = list(new_param.shape)
                metadata['dtypes'][name] = str(new_param.dtype)
                
        # Package
        package = {
            'deltas': deltas,
            'metadata': metadata,
            'version': '0.1.0',
            'timestamp': time.time()
        }
        
        # Serialize and compress
        serialized = pickle.dumps(package, protocol=pickle.HIGHEST_PROTOCOL)
        compressed = gzip.compress(serialized, compresslevel=self.compression_level)
        
        return compressed
        
    def apply_delta(
        self, 
        base_state: Dict[str, torch.Tensor], 
        delta_data: bytes
    ) -> Dict[str, torch.Tensor]:
        """
        Apply delta to a base model state
        
        Args:
            base_state: Base model state
            delta_data: Compressed delta bytes
            
        Returns:
            Updated model state
        """
        # Decompress and deserialize
        decompressed = gzip.decompress(delta_data)
        package = pickle.loads(decompressed)
        
        # Extract data
        deltas = package['deltas']
        metadata = package['metadata']
        
        # Apply deltas
        new_state = base_state.copy()
        
        for name in metadata['changed_params']:
            if name in deltas:
                # Convert delta back to tensor
                delta_array = deltas[name]
                dtype_str = metadata['dtypes'][name]
                dtype = getattr(torch, dtype_str.split('.')[-1])
                
                delta_tensor = torch.from_numpy(delta_array).to(dtype)
                
                if name in new_state:
                    # Apply delta
                    new_state[name] = new_state[name] + delta_tensor
                else:
                    # New parameter
                    new_state[name] = delta_tensor
                    
        return new_state
        
    def serialize_lightweight(self, data: Dict[str, Any]) -> str:
        """
        Serialize lightweight data (metadata, configs, etc.) to base64 string
        
        Args:
            data: Dictionary to serialize
            
        Returns:
            Base64-encoded string
        """
        # Use msgpack for better compression of small data
        packed = msgpack.packb(data, use_bin_type=True)
        compressed = gzip.compress(packed, compresslevel=self.compression_level)
        encoded = base64.b64encode(compressed).decode('utf-8')
        
        return encoded
        
    def deserialize_lightweight(self, data: str) -> Dict[str, Any]:
        """
        Deserialize lightweight data from base64 string
        
        Args:
            data: Base64-encoded string
            
        Returns:
            Deserialized dictionary
        """
        # Decode and decompress
        compressed = base64.b64decode(data.encode('utf-8'))
        packed = gzip.decompress(compressed)
        
        # Unpack with msgpack
        unpacked = msgpack.unpackb(packed, raw=False)
        
        return unpacked
        
    def estimate_serialized_size(self, model_state: Dict[str, torch.Tensor]) -> Dict[str, int]:
        """
        Estimate serialized size without actually serializing
        
        Args:
            model_state: Model state to estimate
            
        Returns:
            Dictionary with size estimates in bytes
        """
        total_params = 0
        total_bytes = 0
        
        for name, tensor in model_state.items():
            num_params = tensor.numel()
            param_bytes = num_params * tensor.element_size()
            
            total_params += num_params
            total_bytes += param_bytes
            
        # Estimate compression ratio (typically 2-4x for neural network weights)
        estimated_compression_ratio = 3.0
        compressed_size = int(total_bytes / estimated_compression_ratio)
        
        return {
            'total_parameters': total_params,
            'uncompressed_bytes': total_bytes,
            'estimated_compressed_bytes': compressed_size,
            'compression_ratio': estimated_compression_ratio
        }
        
    # Internal methods
    
    def _quantize_tensor(
        self, 
        tensor: torch.Tensor, 
        bits: int = 8
    ) -> tuple[torch.Tensor, float, int]:
        """
        Quantize a tensor to reduce size
        
        Args:
            tensor: Tensor to quantize
            bits: Number of bits for quantization
            
        Returns:
            Tuple of (quantized_tensor, scale, zero_point)
        """
        # Calculate quantization parameters
        min_val = tensor.min().item()
        max_val = tensor.max().item()
        
        # Calculate scale and zero point
        qmin = 0
        qmax = (2 ** bits) - 1
        
        scale = (max_val - min_val) / (qmax - qmin)
        zero_point = qmin - min_val / scale
        zero_point = int(round(zero_point))
        zero_point = max(qmin, min(qmax, zero_point))
        
        # Quantize
        quantized = torch.round(tensor / scale + zero_point)
        quantized = torch.clamp(quantized, qmin, qmax).to(torch.uint8)
        
        return quantized, scale, zero_point
        
    def _dequantize_tensor(
        self, 
        quantized_tensor: torch.Tensor, 
        scale: float, 
        zero_point: int
    ) -> torch.Tensor:
        """
        Dequantize a tensor
        
        Args:
            quantized_tensor: Quantized tensor
            scale: Quantization scale
            zero_point: Quantization zero point
            
        Returns:
            Dequantized tensor
        """
        # Dequantize
        dequantized = scale * (quantized_tensor.float() - zero_point)
        return dequantized
        
    def get_compression_stats(self, original_data: bytes, compressed_data: bytes) -> Dict[str, Any]:
        """
        Get compression statistics
        
        Args:
            original_data: Original data bytes
            compressed_data: Compressed data bytes
            
        Returns:
            Dictionary with compression stats
        """
        original_size = len(original_data)
        compressed_size = len(compressed_data)
        compression_ratio = original_size / compressed_size if compressed_size > 0 else 0
        space_saved = original_size - compressed_size
        space_saved_percent = (space_saved / original_size) * 100 if original_size > 0 else 0
        
        return {
            'original_size_bytes': original_size,
            'compressed_size_bytes': compressed_size,
            'compression_ratio': compression_ratio,
            'space_saved_bytes': space_saved,
            'space_saved_percent': space_saved_percent
        }


# Module-level convenience functions

def quick_serialize_model(model_state: Dict[str, torch.Tensor]) -> bytes:
    """Quick model serialization"""
    serializer = SwarmSerializer()
    return serializer.serialize_model(model_state)
    

def quick_deserialize_model(data: bytes) -> Dict[str, torch.Tensor]:
    """Quick model deserialization"""
    serializer = SwarmSerializer()
    return serializer.deserialize_model(data)
    

def quick_serialize_gradients(gradients: Dict[str, torch.Tensor]) -> bytes:
    """Quick gradient serialization"""
    serializer = SwarmSerializer()
    return serializer.serialize_gradients(gradients)