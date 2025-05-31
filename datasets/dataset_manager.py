"""
ncrsh-Swarm Dataset Management System
===================================

Comprehensive dataset handling for distributed neural network training
with automatic partitioning, streaming, and federated data management.
"""

import asyncio
import os
import json
import hashlib
import time
from typing import Dict, List, Any, Optional, Tuple, Iterator
from pathlib import Path
import pickle
import gzip
from dataclasses import dataclass, asdict
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, DistributedSampler
import requests
from abc import ABC, abstractmethod


@dataclass
class DatasetMetadata:
    """Metadata for swarm datasets"""
    name: str
    version: str
    description: str
    size_mb: float
    num_samples: int
    features: Dict[str, Any]
    splits: Dict[str, int]  # train/val/test splits
    hash_sha256: str
    created_at: float
    tags: List[str]
    license: str
    source_url: Optional[str] = None


class SwarmDataset(Dataset, ABC):
    """
    Base class for swarm-compatible datasets
    
    Features:
    - Distributed loading and caching
    - Automatic data partitioning
    - Privacy-preserving data handling
    - Real-time streaming capabilities
    """
    
    def __init__(self, name: str, split: str = 'train', cache_dir: str = './data_cache'):
        self.name = name
        self.split = split
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.metadata: Optional[DatasetMetadata] = None
        self._data_loaded = False
        self._samples = []
        
    @abstractmethod
    async def _load_data(self) -> List[Any]:
        """Load the actual data - to be implemented by subclasses"""
        pass
        
    @abstractmethod
    def _process_sample(self, raw_sample: Any) -> Tuple[torch.Tensor, torch.Tensor]:
        """Process a raw sample into input/target tensors"""
        pass
        
    async def load_async(self):
        """Asynchronously load the dataset"""
        if not self._data_loaded:
            self._samples = await self._load_data()
            self._data_loaded = True
            
    def __len__(self) -> int:
        return len(self._samples)
        
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if not self._data_loaded:
            raise RuntimeError("Dataset not loaded. Call load_async() first.")
        return self._process_sample(self._samples[idx])
        
    def get_metadata(self) -> DatasetMetadata:
        """Get dataset metadata"""
        return self.metadata
        
    def get_partition(self, node_id: str, total_nodes: int) -> 'SwarmDataset':
        """Get a data partition for a specific node"""
        partition_size = len(self._samples) // total_nodes
        start_idx = hash(node_id) % total_nodes * partition_size
        end_idx = start_idx + partition_size
        
        # Create a partitioned version
        partitioned = type(self)(self.name, self.split, str(self.cache_dir))
        partitioned._samples = self._samples[start_idx:end_idx]
        partitioned._data_loaded = True
        partitioned.metadata = self.metadata
        
        return partitioned


class LanguageModelingDataset(SwarmDataset):
    """Language modeling dataset for transformer training"""
    
    def __init__(self, name: str, split: str = 'train', 
                 vocab_size: int = 50257, seq_len: int = 1024, **kwargs):
        super().__init__(name, split, **kwargs)
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        
    async def _load_data(self) -> List[torch.Tensor]:
        """Load tokenized text data"""
        cache_file = self.cache_dir / f"{self.name}_{self.split}_tokens.pt"
        
        if cache_file.exists():
            # Load from cache
            return torch.load(cache_file)
        else:
            # Generate synthetic data for demo
            samples = []
            num_samples = 10000 if self.split == 'train' else 1000
            
            for _ in range(num_samples):
                # Generate random token sequences
                tokens = torch.randint(0, self.vocab_size, (self.seq_len,))
                samples.append(tokens)
                
            # Save to cache
            torch.save(samples, cache_file)
            
            # Create metadata
            self.metadata = DatasetMetadata(
                name=self.name,
                version="1.0.0",
                description=f"Synthetic language modeling dataset for {self.split}",
                size_mb=len(samples) * self.seq_len * 4 / (1024**2),  # 4 bytes per int32
                num_samples=len(samples),
                features={'vocab_size': self.vocab_size, 'seq_len': self.seq_len},
                splits={self.split: len(samples)},
                hash_sha256=hashlib.sha256(str(samples[0]).encode()).hexdigest(),
                created_at=time.time(),
                tags=['language_modeling', 'synthetic', 'transformer'],
                license='MIT'
            )
            
            return samples
            
    def _process_sample(self, raw_sample: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Process token sequence into input/target pair"""
        # Input is all tokens except last, target is all tokens except first
        input_tokens = raw_sample[:-1]
        target_tokens = raw_sample[1:]
        return input_tokens, target_tokens


class ImageClassificationDataset(SwarmDataset):
    """Image classification dataset for vision models"""
    
    def __init__(self, name: str, split: str = 'train', 
                 num_classes: int = 1000, image_size: int = 224, **kwargs):
        super().__init__(name, split, **kwargs)
        self.num_classes = num_classes
        self.image_size = image_size
        
    async def _load_data(self) -> List[Tuple[np.ndarray, int]]:
        """Load image data"""
        cache_file = self.cache_dir / f"{self.name}_{self.split}_images.pkl"
        
        if cache_file.exists():
            with open(cache_file, 'rb') as f:
                return pickle.load(f)
        else:
            # Generate synthetic image data
            samples = []
            num_samples = 50000 if self.split == 'train' else 10000
            
            for _ in range(num_samples):
                # Generate random image (height, width, channels)
                image = np.random.randint(0, 256, 
                    (self.image_size, self.image_size, 3), dtype=np.uint8)
                label = np.random.randint(0, self.num_classes)
                samples.append((image, label))
                
            # Save to cache
            with open(cache_file, 'wb') as f:
                pickle.dump(samples, f)
                
            # Create metadata
            self.metadata = DatasetMetadata(
                name=self.name,
                version="1.0.0", 
                description=f"Synthetic image classification dataset for {self.split}",
                size_mb=len(samples) * self.image_size * self.image_size * 3 / (1024**2),
                num_samples=len(samples),
                features={'num_classes': self.num_classes, 'image_size': self.image_size},
                splits={self.split: len(samples)},
                hash_sha256=hashlib.sha256(str(len(samples)).encode()).hexdigest(),
                created_at=time.time(),
                tags=['image_classification', 'synthetic', 'computer_vision'],
                license='MIT'
            )
            
            return samples
            
    def _process_sample(self, raw_sample: Tuple[np.ndarray, int]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Process image/label pair into tensors"""
        image, label = raw_sample
        
        # Convert to torch tensors and normalize
        image_tensor = torch.from_numpy(image).float() / 255.0
        image_tensor = image_tensor.permute(2, 0, 1)  # HWC -> CHW
        
        label_tensor = torch.tensor(label, dtype=torch.long)
        
        return image_tensor, label_tensor


class FederatedDataset(SwarmDataset):
    """
    Dataset for federated learning scenarios with privacy preservation
    """
    
    def __init__(self, name: str, split: str = 'train',
                 privacy_budget: float = 1.0, **kwargs):
        super().__init__(name, split, **kwargs)
        self.privacy_budget = privacy_budget
        self._noise_scale = 1.0 / privacy_budget
        
    async def _load_data(self) -> List[Any]:
        """Load federated data with privacy considerations"""
        # This would implement differential privacy, secure aggregation, etc.
        return await super()._load_data()
        
    def add_privacy_noise(self, tensor: torch.Tensor) -> torch.Tensor:
        """Add differential privacy noise to tensor"""
        if self.privacy_budget > 0:
            noise = torch.normal(0, self._noise_scale, tensor.shape)
            return tensor + noise
        return tensor
        
    def _process_sample(self, raw_sample: Any) -> Tuple[torch.Tensor, torch.Tensor]:
        """Process with privacy preservation"""
        input_tensor, target_tensor = super()._process_sample(raw_sample)
        
        # Add privacy noise if enabled
        if self.privacy_budget < float('inf'):
            input_tensor = self.add_privacy_noise(input_tensor)
            
        return input_tensor, target_tensor


class StreamingDataset(SwarmDataset):
    """
    Streaming dataset for real-time data processing
    """
    
    def __init__(self, name: str, stream_url: str, buffer_size: int = 1000, **kwargs):
        super().__init__(name, 'stream', **kwargs)
        self.stream_url = stream_url
        self.buffer_size = buffer_size
        self._buffer = []
        self._streaming = False
        
    async def _load_data(self) -> List[Any]:
        """Initialize streaming buffer"""
        self._buffer = []
        return self._buffer
        
    async def start_streaming(self):
        """Start streaming data"""
        self._streaming = True
        asyncio.create_task(self._stream_loop())
        
    async def stop_streaming(self):
        """Stop streaming data"""
        self._streaming = False
        
    async def _stream_loop(self):
        """Background streaming loop"""
        while self._streaming:
            try:
                # Simulate streaming data arrival
                new_sample = await self._fetch_stream_sample()
                
                # Add to buffer
                self._buffer.append(new_sample)
                
                # Maintain buffer size
                if len(self._buffer) > self.buffer_size:
                    self._buffer.pop(0)
                    
                await asyncio.sleep(0.1)  # Stream rate
                
            except Exception as e:
                print(f"Streaming error: {e}")
                await asyncio.sleep(1)
                
    async def _fetch_stream_sample(self) -> Any:
        """Fetch a single sample from stream"""
        # This would integrate with real streaming sources
        # For demo, generate synthetic data
        return torch.randn(10)  # Random feature vector
        
    def _process_sample(self, raw_sample: Any) -> Tuple[torch.Tensor, torch.Tensor]:
        """Process streaming sample"""
        # For streaming, we might predict next value
        input_tensor = raw_sample[:-1]
        target_tensor = raw_sample[-1:]
        return input_tensor, target_tensor


class DatasetManager:
    """
    Central dataset management system for ncrsh-Swarm
    
    Features:
    - Dataset discovery and cataloging
    - Automatic downloading and caching
    - Distributed data partitioning
    - Quality validation and monitoring
    """
    
    def __init__(self, cache_dir: str = './datasets'):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.catalog: Dict[str, DatasetMetadata] = {}
        self.active_datasets: Dict[str, SwarmDataset] = {}
        
        # Load existing catalog
        asyncio.create_task(self._load_catalog())
        
    async def _load_catalog(self):
        """Load dataset catalog from disk"""
        catalog_file = self.cache_dir / 'catalog.json'
        
        if catalog_file.exists():
            with open(catalog_file, 'r') as f:
                catalog_data = json.load(f)
                
            for name, metadata_dict in catalog_data.items():
                self.catalog[name] = DatasetMetadata(**metadata_dict)
                
    async def _save_catalog(self):
        """Save dataset catalog to disk"""
        catalog_file = self.cache_dir / 'catalog.json'
        
        catalog_data = {
            name: asdict(metadata) 
            for name, metadata in self.catalog.items()
        }
        
        with open(catalog_file, 'w') as f:
            json.dump(catalog_data, f, indent=2)
            
    async def register_dataset(self, dataset: SwarmDataset):
        """Register a new dataset"""
        await dataset.load_async()
        metadata = dataset.get_metadata()
        
        if metadata:
            self.catalog[dataset.name] = metadata
            self.active_datasets[dataset.name] = dataset
            await self._save_catalog()
            
    async def get_dataset(self, name: str, split: str = 'train', **kwargs) -> SwarmDataset:
        """Get a dataset by name"""
        if name in self.active_datasets:
            return self.active_datasets[name]
            
        # Create dataset based on type
        if name.startswith('lm_'):
            dataset = LanguageModelingDataset(name, split, **kwargs)
        elif name.startswith('img_'):
            dataset = ImageClassificationDataset(name, split, **kwargs)
        elif name.startswith('fed_'):
            dataset = FederatedDataset(name, split, **kwargs)
        elif name.startswith('stream_'):
            dataset = StreamingDataset(name, kwargs.get('stream_url', ''), **kwargs)
        else:
            raise ValueError(f"Unknown dataset type for: {name}")
            
        await dataset.load_async()
        await self.register_dataset(dataset)
        
        return dataset
        
    def list_datasets(self) -> List[str]:
        """List all available datasets"""
        return list(self.catalog.keys())
        
    def get_metadata(self, name: str) -> Optional[DatasetMetadata]:
        """Get metadata for a dataset"""
        return self.catalog.get(name)
        
    async def create_distributed_loader(
        self, 
        dataset_name: str,
        node_id: str,
        total_nodes: int,
        batch_size: int = 32,
        **kwargs
    ) -> DataLoader:
        """Create a distributed data loader for a node"""
        dataset = await self.get_dataset(dataset_name, **kwargs)
        
        # Get partition for this node
        partitioned_dataset = dataset.get_partition(node_id, total_nodes)
        
        # Create distributed sampler
        sampler = DistributedSampler(
            partitioned_dataset,
            num_replicas=total_nodes,
            rank=hash(node_id) % total_nodes,
            shuffle=True
        )
        
        # Create data loader
        loader = DataLoader(
            partitioned_dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=2,
            pin_memory=True
        )
        
        return loader
        
    async def validate_dataset(self, name: str) -> Dict[str, Any]:
        """Validate dataset integrity and quality"""
        if name not in self.catalog:
            return {'valid': False, 'error': 'Dataset not found'}
            
        try:
            dataset = await self.get_dataset(name)
            metadata = dataset.get_metadata()
            
            # Basic validation checks
            validation_results = {
                'valid': True,
                'checks': {
                    'metadata_present': metadata is not None,
                    'samples_loadable': len(dataset) > 0,
                    'sample_format_valid': True,  # Would check sample format
                    'size_matches_metadata': len(dataset) == metadata.num_samples if metadata else False
                }
            }
            
            # Check a few samples
            try:
                for i in range(min(5, len(dataset))):
                    input_tensor, target_tensor = dataset[i]
                    assert isinstance(input_tensor, torch.Tensor)
                    assert isinstance(target_tensor, torch.Tensor)
            except Exception as e:
                validation_results['checks']['sample_format_valid'] = False
                validation_results['error'] = str(e)
                
            validation_results['valid'] = all(validation_results['checks'].values())
            
            return validation_results
            
        except Exception as e:
            return {'valid': False, 'error': str(e)}
            
    async def get_dataset_stats(self, name: str) -> Dict[str, Any]:
        """Get comprehensive dataset statistics"""
        dataset = await self.get_dataset(name)
        metadata = dataset.get_metadata()
        
        stats = {
            'name': name,
            'total_samples': len(dataset),
            'memory_usage_mb': 0,  # Would calculate actual memory usage
            'cache_size_mb': 0,    # Would check cache size
            'last_accessed': time.time(),
            'splits_available': list(metadata.splits.keys()) if metadata else []
        }
        
        # Sample-level statistics
        if len(dataset) > 0:
            sample_input, sample_target = dataset[0]
            stats.update({
                'input_shape': list(sample_input.shape),
                'target_shape': list(sample_target.shape),
                'input_dtype': str(sample_input.dtype),
                'target_dtype': str(sample_target.dtype)
            })
            
        return stats
        
    async def cleanup_cache(self, max_age_days: int = 30):
        """Clean up old cached datasets"""
        cutoff_time = time.time() - (max_age_days * 24 * 3600)
        
        cleaned_count = 0
        for name, metadata in list(self.catalog.items()):
            if metadata.created_at < cutoff_time:
                # Remove from catalog
                del self.catalog[name]
                
                # Remove from active datasets
                if name in self.active_datasets:
                    del self.active_datasets[name]
                    
                cleaned_count += 1
                
        await self._save_catalog()
        return cleaned_count


# Utility functions for dataset creation

async def create_synthetic_language_dataset(
    name: str, 
    vocab_size: int = 10000, 
    seq_len: int = 512,
    num_samples: int = 100000
) -> LanguageModelingDataset:
    """Create synthetic language modeling dataset"""
    dataset = LanguageModelingDataset(
        name=name,
        vocab_size=vocab_size,
        seq_len=seq_len
    )
    await dataset.load_async()
    return dataset


async def create_synthetic_image_dataset(
    name: str,
    num_classes: int = 100,
    image_size: int = 64,
    num_samples: int = 10000
) -> ImageClassificationDataset:
    """Create synthetic image classification dataset"""
    dataset = ImageClassificationDataset(
        name=name,
        num_classes=num_classes,
        image_size=image_size
    )
    await dataset.load_async()
    return dataset


# Example usage
async def main():
    """Example dataset management usage"""
    print("🗃️  ncrsh-Swarm Dataset Management Example")
    
    # Create dataset manager
    manager = DatasetManager()
    
    # Create some synthetic datasets
    print("Creating synthetic datasets...")
    
    lm_dataset = await create_synthetic_language_dataset(
        'lm_synthetic_small', vocab_size=5000, seq_len=256, num_samples=1000
    )
    await manager.register_dataset(lm_dataset)
    
    img_dataset = await create_synthetic_image_dataset(
        'img_synthetic_small', num_classes=10, image_size=32, num_samples=1000
    )
    await manager.register_dataset(img_dataset)
    
    # List available datasets
    print(f"Available datasets: {manager.list_datasets()}")
    
    # Get dataset statistics
    for dataset_name in manager.list_datasets():
        stats = await manager.get_dataset_stats(dataset_name)
        print(f"\nDataset: {dataset_name}")
        print(f"  Samples: {stats['total_samples']}")
        print(f"  Input shape: {stats.get('input_shape', 'N/A')}")
        print(f"  Target shape: {stats.get('target_shape', 'N/A')}")
        
    # Create distributed data loader
    print("\nCreating distributed data loader...")
    loader = await manager.create_distributed_loader(
        'lm_synthetic_small',
        node_id='test_node_1',
        total_nodes=3,
        batch_size=16
    )
    
    print(f"Data loader created with {len(loader)} batches")
    
    # Test a batch
    for batch_input, batch_target in loader:
        print(f"Batch input shape: {batch_input.shape}")
        print(f"Batch target shape: {batch_target.shape}")
        break
        
    print("✅ Dataset management example completed!")


if __name__ == "__main__":
    asyncio.run(main())