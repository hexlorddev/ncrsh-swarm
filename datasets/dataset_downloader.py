"""
Dataset Downloader for ncrsh-Swarm
=================================

Automatic downloading and preprocessing of popular ML datasets
for distributed training across the swarm network.
"""

import asyncio
import aiohttp
import aiofiles
import os
import zipfile
import tarfile
import gzip
import json
import hashlib
from typing import Dict, List, Any, Optional
from pathlib import Path
import urllib.parse


class DatasetDownloader:
    """
    Download and prepare datasets for swarm training
    
    Supports:
    - Common NLP datasets (WikiText, OpenWebText, etc.)
    - Computer vision datasets (CIFAR, ImageNet subsets)
    - Audio datasets (LibriSpeech, Common Voice)
    - Multimodal datasets
    """
    
    DATASET_REGISTRY = {
        # Language Modeling Datasets
        'wikitext-103': {
            'url': 'https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-103-v1.zip',
            'type': 'text',
            'size_mb': 183,
            'description': 'WikiText-103 language modeling dataset'
        },
        'wikitext-2': {
            'url': 'https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-2-v1.zip', 
            'type': 'text',
            'size_mb': 4,
            'description': 'WikiText-2 language modeling dataset'
        },
        'penn-treebank': {
            'url': 'https://raw.githubusercontent.com/wojzaremba/lstm/master/data/ptb.train.txt',
            'type': 'text',
            'size_mb': 5,
            'description': 'Penn Treebank dataset'
        },
        
        # Computer Vision Datasets
        'cifar-10': {
            'url': 'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz',
            'type': 'image',
            'size_mb': 163,
            'description': 'CIFAR-10 image classification dataset'
        },
        'cifar-100': {
            'url': 'https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz',
            'type': 'image', 
            'size_mb': 163,
            'description': 'CIFAR-100 image classification dataset'
        },
        'mnist': {
            'url': 'http://yann.lecun.com/exdb/mnist/',
            'type': 'image',
            'size_mb': 60,
            'description': 'MNIST handwritten digits dataset'
        },
        
        # Audio Datasets
        'common-voice-en': {
            'url': 'https://voice-prod-bundler-ee1969a6ce8178826482b88e843c335139bd3fb4.s3.amazonaws.com/cv-corpus-6.1-2020-12-11/en.tar.gz',
            'type': 'audio',
            'size_mb': 13000,
            'description': 'Common Voice English speech dataset'
        },
        
        # Multimodal Datasets
        'conceptual-captions': {
            'url': 'https://ai.google.com/research/ConceptualCaptions/download',
            'type': 'multimodal',
            'size_mb': 500,
            'description': 'Conceptual Captions image-text dataset'
        }
    }
    
    def __init__(self, download_dir: str = './downloaded_datasets'):
        self.download_dir = Path(download_dir)
        self.download_dir.mkdir(parents=True, exist_ok=True)
        self.session: Optional[aiohttp.ClientSession] = None
        
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
            
    async def download_dataset(self, dataset_name: str, force_redownload: bool = False) -> Path:
        """Download a dataset by name"""
        if dataset_name not in self.DATASET_REGISTRY:
            raise ValueError(f"Unknown dataset: {dataset_name}")
            
        dataset_info = self.DATASET_REGISTRY[dataset_name]
        dataset_dir = self.download_dir / dataset_name
        
        # Check if already downloaded
        if dataset_dir.exists() and not force_redownload:
            print(f"📁 Dataset {dataset_name} already exists at {dataset_dir}")
            return dataset_dir
            
        print(f"⬇️  Downloading {dataset_name} ({dataset_info['size_mb']}MB)...")
        
        # Create dataset directory
        dataset_dir.mkdir(parents=True, exist_ok=True)
        
        # Download and extract
        await self._download_and_extract(dataset_info['url'], dataset_dir)
        
        # Create metadata file
        metadata = {
            'name': dataset_name,
            'description': dataset_info['description'],
            'type': dataset_info['type'],
            'size_mb': dataset_info['size_mb'],
            'download_url': dataset_info['url'],
            'downloaded_at': asyncio.get_event_loop().time(),
            'version': '1.0'
        }
        
        metadata_file = dataset_dir / 'metadata.json'
        async with aiofiles.open(metadata_file, 'w') as f:
            await f.write(json.dumps(metadata, indent=2))
            
        print(f"✅ Downloaded {dataset_name} to {dataset_dir}")
        return dataset_dir
        
    async def _download_and_extract(self, url: str, target_dir: Path):
        """Download and extract a file"""
        if not self.session:
            raise RuntimeError("Downloader not initialized. Use async context manager.")
            
        # Determine filename from URL
        parsed_url = urllib.parse.urlparse(url)
        filename = Path(parsed_url.path).name
        
        if not filename:
            filename = 'dataset.zip'
            
        file_path = target_dir / filename
        
        # Download file
        async with self.session.get(url) as response:
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            downloaded = 0
            
            async with aiofiles.open(file_path, 'wb') as f:
                async for chunk in response.content.iter_chunked(8192):
                    await f.write(chunk)
                    downloaded += len(chunk)
                    
                    if total_size > 0:
                        progress = (downloaded / total_size) * 100
                        print(f"\r  Progress: {progress:.1f}%", end='', flush=True)
                        
        print()  # New line after progress
        
        # Extract if it's an archive
        await self._extract_archive(file_path, target_dir)
        
        # Remove the archive file to save space
        file_path.unlink()
        
    async def _extract_archive(self, archive_path: Path, target_dir: Path):
        """Extract various archive formats"""
        suffix = archive_path.suffix.lower()
        
        if suffix == '.zip':
            with zipfile.ZipFile(archive_path, 'r') as zip_ref:
                zip_ref.extractall(target_dir)
        elif suffix == '.gz' and archive_path.stem.endswith('.tar'):
            with tarfile.open(archive_path, 'r:gz') as tar_ref:
                tar_ref.extractall(target_dir)
        elif suffix == '.tar':
            with tarfile.open(archive_path, 'r') as tar_ref:
                tar_ref.extractall(target_dir)
        elif suffix == '.gz':
            # Single gzipped file
            output_path = target_dir / archive_path.stem
            with gzip.open(archive_path, 'rb') as gz_file:
                async with aiofiles.open(output_path, 'wb') as out_file:
                    while True:
                        chunk = gz_file.read(8192)
                        if not chunk:
                            break
                        await out_file.write(chunk)
                        
    async def download_multiple(self, dataset_names: List[str], max_concurrent: int = 3) -> Dict[str, Path]:
        """Download multiple datasets concurrently"""
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def download_with_semaphore(name):
            async with semaphore:
                return await self.download_dataset(name)
                
        tasks = [download_with_semaphore(name) for name in dataset_names]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        return {
            name: result for name, result in zip(dataset_names, results)
            if not isinstance(result, Exception)
        }
        
    def list_available_datasets(self) -> List[Dict[str, Any]]:
        """List all available datasets"""
        return [
            {
                'name': name,
                'description': info['description'],
                'type': info['type'],
                'size_mb': info['size_mb']
            }
            for name, info in self.DATASET_REGISTRY.items()
        ]
        
    def get_dataset_info(self, dataset_name: str) -> Optional[Dict[str, Any]]:
        """Get information about a specific dataset"""
        return self.DATASET_REGISTRY.get(dataset_name)
        
    async def verify_download(self, dataset_name: str) -> bool:
        """Verify a downloaded dataset"""
        dataset_dir = self.download_dir / dataset_name
        metadata_file = dataset_dir / 'metadata.json'
        
        if not dataset_dir.exists() or not metadata_file.exists():
            return False
            
        # Load metadata and verify
        async with aiofiles.open(metadata_file, 'r') as f:
            metadata = json.loads(await f.read())
            
        # Basic verification - check if expected files exist
        expected_files = self._get_expected_files(dataset_name)
        
        for expected_file in expected_files:
            if not (dataset_dir / expected_file).exists():
                return False
                
        return True
        
    def _get_expected_files(self, dataset_name: str) -> List[str]:
        """Get expected files for a dataset"""
        # This would be more comprehensive in a real implementation
        file_patterns = {
            'wikitext-103': ['wikitext-103/wiki.train.tokens'],
            'wikitext-2': ['wikitext-2/wiki.train.tokens'],
            'cifar-10': ['cifar-10-batches-py/data_batch_1'],
            'cifar-100': ['cifar-100-python/train'],
            'mnist': ['train-images-idx3-ubyte', 'train-labels-idx1-ubyte']
        }
        
        return file_patterns.get(dataset_name, ['data.txt'])
        
    async def get_download_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all downloaded datasets"""
        status = {}
        
        for dataset_name in self.DATASET_REGISTRY:
            dataset_dir = self.download_dir / dataset_name
            
            if dataset_dir.exists():
                metadata_file = dataset_dir / 'metadata.json'
                
                if metadata_file.exists():
                    async with aiofiles.open(metadata_file, 'r') as f:
                        metadata = json.loads(await f.read())
                        
                    # Calculate directory size
                    total_size = sum(
                        f.stat().st_size for f in dataset_dir.rglob('*') if f.is_file()
                    )
                    
                    status[dataset_name] = {
                        'downloaded': True,
                        'verified': await self.verify_download(dataset_name),
                        'size_bytes': total_size,
                        'download_date': metadata.get('downloaded_at'),
                        'version': metadata.get('version')
                    }
                else:
                    status[dataset_name] = {
                        'downloaded': True,
                        'verified': False,
                        'size_bytes': 0,
                        'download_date': None,
                        'version': None
                    }
            else:
                status[dataset_name] = {
                    'downloaded': False,
                    'verified': False,
                    'size_bytes': 0,
                    'download_date': None,
                    'version': None
                }
                
        return status
        
    async def cleanup_incomplete_downloads(self):
        """Clean up incomplete or corrupted downloads"""
        cleaned_count = 0
        
        for dataset_name in self.DATASET_REGISTRY:
            if not await self.verify_download(dataset_name):
                dataset_dir = self.download_dir / dataset_name
                
                if dataset_dir.exists():
                    # Remove incomplete download
                    import shutil
                    shutil.rmtree(dataset_dir)
                    cleaned_count += 1
                    print(f"🧹 Cleaned up incomplete download: {dataset_name}")
                    
        return cleaned_count


# Command-line interface
async def main():
    """CLI for dataset downloader"""
    import argparse
    
    parser = argparse.ArgumentParser(description="ncrsh-Swarm Dataset Downloader")
    parser.add_argument('command', choices=['list', 'download', 'status', 'verify', 'cleanup'])
    parser.add_argument('--dataset', help='Dataset name to download/verify')
    parser.add_argument('--all', action='store_true', help='Download all datasets')
    parser.add_argument('--force', action='store_true', help='Force redownload')
    
    args = parser.parse_args()
    
    async with DatasetDownloader() as downloader:
        if args.command == 'list':
            datasets = downloader.list_available_datasets()
            print("📋 Available Datasets:")
            for dataset in datasets:
                print(f"  {dataset['name']}: {dataset['description']} ({dataset['size_mb']}MB)")
                
        elif args.command == 'download':
            if args.all:
                dataset_names = list(downloader.DATASET_REGISTRY.keys())
                print(f"⬇️  Downloading {len(dataset_names)} datasets...")
                results = await downloader.download_multiple(dataset_names)
                print(f"✅ Downloaded {len(results)} datasets successfully")
            elif args.dataset:
                await downloader.download_dataset(args.dataset, args.force)
            else:
                print("❌ Specify --dataset or --all")
                
        elif args.command == 'status':
            status = await downloader.get_download_status()
            print("📊 Dataset Status:")
            for name, info in status.items():
                status_icon = "✅" if info['verified'] else "⚠️" if info['downloaded'] else "❌"
                size_mb = info['size_bytes'] / (1024**2) if info['size_bytes'] else 0
                print(f"  {status_icon} {name}: {size_mb:.1f}MB")
                
        elif args.command == 'verify':
            if args.dataset:
                is_valid = await downloader.verify_download(args.dataset)
                print(f"{'✅' if is_valid else '❌'} {args.dataset} verification: {'PASSED' if is_valid else 'FAILED'}")
            else:
                print("❌ Specify --dataset to verify")
                
        elif args.command == 'cleanup':
            cleaned = await downloader.cleanup_incomplete_downloads()
            print(f"🧹 Cleaned up {cleaned} incomplete downloads")


if __name__ == "__main__":
    asyncio.run(main())