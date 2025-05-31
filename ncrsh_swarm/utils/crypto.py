"""
Cryptographic utilities for SwarmNodes
"""

import hashlib
import hmac
import secrets
import time
from typing import Dict, Any, Optional
import torch
import base64
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC


class SwarmCrypto:
    """
    Cryptographic utilities for secure swarm communication
    
    Features:
    - Model state hashing for integrity verification
    - Symmetric encryption for sensitive data
    - Digital signatures for authentication
    - Secure key derivation and management
    """
    
    def __init__(self, master_key: Optional[bytes] = None):
        self.master_key = master_key or secrets.token_bytes(32)
        self._fernet = None
        
    def hash_model(self, model_state: Dict[str, torch.Tensor]) -> str:
        """
        Create a cryptographic hash of a model state
        
        Args:
            model_state: Dictionary of model parameters
            
        Returns:
            Hexadecimal hash string
        """
        hasher = hashlib.sha256()
        
        # Sort parameters by name for deterministic hashing
        for param_name in sorted(model_state.keys()):
            param = model_state[param_name]
            
            # Add parameter name
            hasher.update(param_name.encode('utf-8'))
            
            # Add parameter shape
            shape_str = ','.join(map(str, param.shape))
            hasher.update(shape_str.encode('utf-8'))
            
            # Add parameter data
            param_bytes = param.detach().cpu().numpy().tobytes()
            hasher.update(param_bytes)
            
        return hasher.hexdigest()
        
    def verify_model_integrity(
        self, 
        model_state: Dict[str, torch.Tensor], 
        expected_hash: str
    ) -> bool:
        """
        Verify the integrity of a model state against a hash
        
        Args:
            model_state: Model state to verify
            expected_hash: Expected hash value
            
        Returns:
            True if integrity is verified, False otherwise
        """
        computed_hash = self.hash_model(model_state)
        return hmac.compare_digest(computed_hash, expected_hash)
        
    def encrypt_data(self, data: bytes, password: Optional[str] = None) -> bytes:
        """
        Encrypt data using symmetric encryption
        
        Args:
            data: Data to encrypt
            password: Optional password for key derivation
            
        Returns:
            Encrypted data
        """
        fernet = self._get_fernet(password)
        return fernet.encrypt(data)
        
    def decrypt_data(self, encrypted_data: bytes, password: Optional[str] = None) -> bytes:
        """
        Decrypt data using symmetric encryption
        
        Args:
            encrypted_data: Data to decrypt
            password: Optional password for key derivation
            
        Returns:
            Decrypted data
        """
        fernet = self._get_fernet(password)
        return fernet.decrypt(encrypted_data)
        
    def sign_data(self, data: bytes, key: Optional[bytes] = None) -> str:
        """
        Create HMAC signature for data authentication
        
        Args:
            data: Data to sign
            key: Optional signing key (uses master key if not provided)
            
        Returns:
            Base64-encoded signature
        """
        signing_key = key or self.master_key
        signature = hmac.new(signing_key, data, hashlib.sha256).digest()
        return base64.b64encode(signature).decode('utf-8')
        
    def verify_signature(
        self, 
        data: bytes, 
        signature: str, 
        key: Optional[bytes] = None
    ) -> bool:
        """
        Verify HMAC signature
        
        Args:
            data: Original data
            signature: Base64-encoded signature to verify
            key: Optional verification key (uses master key if not provided)
            
        Returns:
            True if signature is valid, False otherwise
        """
        try:
            verification_key = key or self.master_key
            expected_signature = hmac.new(verification_key, data, hashlib.sha256).digest()
            provided_signature = base64.b64decode(signature.encode('utf-8'))
            return hmac.compare_digest(expected_signature, provided_signature)
        except Exception:
            return False
            
    def generate_node_keypair(self) -> Dict[str, str]:
        """
        Generate a keypair for node identification
        
        Returns:
            Dictionary with 'private_key' and 'public_key'
        """
        # Generate a random private key
        private_key = secrets.token_bytes(32)
        
        # Derive public key using PBKDF2
        salt = b'ncrsh_swarm_salt'
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        public_key = kdf.derive(private_key)
        
        return {
            'private_key': base64.b64encode(private_key).decode('utf-8'),
            'public_key': base64.b64encode(public_key).decode('utf-8')
        }
        
    def derive_shared_secret(self, our_private: str, their_public: str) -> bytes:
        """
        Derive a shared secret for secure communication
        
        Args:
            our_private: Our private key (base64 encoded)
            their_public: Their public key (base64 encoded)
            
        Returns:
            Shared secret bytes
        """
        # Simple XOR-based shared secret (in practice, use proper ECDH)
        our_key = base64.b64decode(our_private.encode('utf-8'))
        their_key = base64.b64decode(their_public.encode('utf-8'))
        
        # XOR the keys to create shared secret
        shared = bytes(a ^ b for a, b in zip(our_key, their_key))
        
        # Hash the result for better distribution
        return hashlib.sha256(shared).digest()
        
    def secure_compare_models(
        self, 
        model1: Dict[str, torch.Tensor], 
        model2: Dict[str, torch.Tensor]
    ) -> Dict[str, Any]:
        """
        Securely compare two models and return similarity metrics
        
        Args:
            model1: First model state
            model2: Second model state
            
        Returns:
            Dictionary with comparison metrics
        """
        # Calculate hashes
        hash1 = self.hash_model(model1)
        hash2 = self.hash_model(model2)
        
        # Check if models are identical
        identical = hash1 == hash2
        
        if identical:
            return {
                'identical': True,
                'similarity': 1.0,
                'hash1': hash1,
                'hash2': hash2
            }
            
        # Calculate parameter-wise similarity
        similarities = []
        common_params = set(model1.keys()) & set(model2.keys())
        
        for param_name in common_params:
            if model1[param_name].shape == model2[param_name].shape:
                # Calculate cosine similarity
                p1 = model1[param_name].flatten().float()
                p2 = model2[param_name].flatten().float()
                
                cosine_sim = torch.nn.functional.cosine_similarity(
                    p1.unsqueeze(0), p2.unsqueeze(0)
                ).item()
                similarities.append(cosine_sim)
                
        overall_similarity = sum(similarities) / len(similarities) if similarities else 0.0
        
        return {
            'identical': False,
            'similarity': overall_similarity,
            'common_params': len(common_params),
            'total_params_model1': len(model1),
            'total_params_model2': len(model2),
            'hash1': hash1,
            'hash2': hash2
        }
        
    def create_secure_checkpoint(
        self, 
        model_state: Dict[str, torch.Tensor], 
        metadata: Dict[str, Any],
        password: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Create a secure, encrypted checkpoint
        
        Args:
            model_state: Model state to checkpoint
            metadata: Additional metadata
            password: Optional encryption password
            
        Returns:
            Encrypted checkpoint data
        """
        import pickle
        
        # Create checkpoint data
        checkpoint_data = {
            'model_state': model_state,
            'metadata': metadata,
            'timestamp': torch.tensor(time.time()),
            'version': '0.1.0'
        }
        
        # Serialize
        serialized = pickle.dumps(checkpoint_data)
        
        # Create hash for integrity
        data_hash = hashlib.sha256(serialized).hexdigest()
        
        # Encrypt
        encrypted_data = self.encrypt_data(serialized, password)
        
        # Create signature
        signature = self.sign_data(serialized)
        
        return {
            'encrypted_data': base64.b64encode(encrypted_data).decode('utf-8'),
            'hash': data_hash,
            'signature': signature,
            'encrypted': True,
            'version': '0.1.0'
        }
        
    def restore_secure_checkpoint(
        self, 
        checkpoint: Dict[str, Any], 
        password: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Restore a secure checkpoint
        
        Args:
            checkpoint: Encrypted checkpoint data
            password: Optional decryption password
            
        Returns:
            Restored checkpoint data
        """
        import pickle
        
        try:
            # Decode encrypted data
            encrypted_data = base64.b64decode(checkpoint['encrypted_data'].encode('utf-8'))
            
            # Decrypt
            decrypted_data = self.decrypt_data(encrypted_data, password)
            
            # Verify signature
            if not self.verify_signature(decrypted_data, checkpoint['signature']):
                raise ValueError("Checkpoint signature verification failed")
                
            # Verify hash
            computed_hash = hashlib.sha256(decrypted_data).hexdigest()
            if not hmac.compare_digest(computed_hash, checkpoint['hash']):
                raise ValueError("Checkpoint hash verification failed")
                
            # Deserialize
            checkpoint_data = pickle.loads(decrypted_data)
            
            return checkpoint_data
            
        except Exception as e:
            raise ValueError(f"Failed to restore secure checkpoint: {e}")
            
    def _get_fernet(self, password: Optional[str] = None) -> Fernet:
        """Get Fernet cipher instance"""
        if self._fernet is None or password is not None:
            key = self._derive_key(password)
            self._fernet = Fernet(key)
        return self._fernet
        
    def _derive_key(self, password: Optional[str] = None) -> bytes:
        """Derive encryption key from password or master key"""
        if password:
            # Derive key from password
            salt = b'ncrsh_swarm_encryption_salt'
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=32,
                salt=salt,
                iterations=100000,
            )
            derived_key = kdf.derive(password.encode('utf-8'))
        else:
            # Use master key
            derived_key = self.master_key
            
        # Fernet requires base64-encoded key
        return base64.urlsafe_b64encode(derived_key)


# Module-level convenience functions

def hash_model_quick(model_state: Dict[str, torch.Tensor]) -> str:
    """Quick model hashing function"""
    crypto = SwarmCrypto()
    return crypto.hash_model(model_state)
    

def verify_model_quick(
    model_state: Dict[str, torch.Tensor], 
    expected_hash: str
) -> bool:
    """Quick model verification function"""
    crypto = SwarmCrypto()
    return crypto.verify_model_integrity(model_state, expected_hash)