#!/usr/bin/env python
"""
Public Audit Chain Exporter (ΨC TraceChain)
------------------------------------------

This module exports an agent's reflective and schema change history as an immutable,
cryptographically signed log for verifiable history in research, safety, or regulatory audits.

Mathematical basis:
- Hash each trace:
  H_i = SHA256(T_i + ΨC_i + ΔH_i)
- Chain them:
  H_{i+1} = SHA256(H_i + T_{i+1})

The module supports multiple export formats and optional distribution to IPFS or other
decentralized storage systems.
"""

import json
import time
import hashlib
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Tuple, Set
import uuid
import base64
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import padding, rsa
from cryptography.hazmat.primitives.asymmetric import utils as crypto_utils
from cryptography.hazmat.primitives.serialization import (
    load_pem_private_key, load_pem_public_key,
    Encoding, PrivateFormat, PublicFormat, NoEncryption
)
from cryptography.exceptions import InvalidSignature
import ipfshttpclient
import requests

# Ensure interoperability with existing logging systems
from psi_c_ai_sdk.logging.safety_trace import SafetyTraceExporter, SafetyEventType, SafetyEvent
from psi_c_ai_sdk.logging.audit_log import AuditLog

# Setup logger
logger = logging.getLogger(__name__)


class TraceBlock:
    """
    A single block in the TraceChain, containing agent events and cryptographic proofs.
    
    Each block contains:
    1. Agent events (reflections, schema changes, etc.)
    2. Metadata about the agent state
    3. Cryptographic hash linked to previous block
    4. Optional digital signature
    """
    
    def __init__(
        self,
        events: List[Dict[str, Any]],
        timestamp: Optional[float] = None,
        block_id: Optional[str] = None,
        previous_hash: str = "",
        agent_state_hash: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize a trace block.
        
        Args:
            events: List of events in this block
            timestamp: Block creation time (defaults to current time)
            block_id: Unique ID for the block (generated if not provided)
            previous_hash: Hash of the previous block in the chain
            agent_state_hash: Hash of the agent's state at this point
            metadata: Additional block metadata
        """
        self.events = events
        self.timestamp = timestamp or time.time()
        self.block_id = block_id or str(uuid.uuid4())
        self.previous_hash = previous_hash
        self.agent_state_hash = agent_state_hash
        self.metadata = metadata or {}
        self.signature = None
        self.hash = None
        
        # Compute the block hash
        self.compute_hash()
    
    def compute_hash(self) -> str:
        """
        Compute the cryptographic hash of this block.
        
        Returns:
            SHA-256 hash of the block contents
        """
        # Prepare a dictionary of all block data
        block_data = {
            "block_id": self.block_id,
            "timestamp": self.timestamp,
            "previous_hash": self.previous_hash,
            "events": self.events,
            "metadata": self.metadata
        }
        
        if self.agent_state_hash:
            block_data["agent_state_hash"] = self.agent_state_hash
            
        # Sort keys for consistent serialization
        serialized = json.dumps(block_data, sort_keys=True)
        
        # Compute hash
        self.hash = hashlib.sha256(serialized.encode()).hexdigest()
        return self.hash
    
    def sign(self, private_key) -> str:
        """
        Sign the block with a private key.
        
        Args:
            private_key: RSA private key for signing
            
        Returns:
            Base64-encoded signature
        """
        # Ensure hash is computed
        if not self.hash:
            self.compute_hash()
            
        # Create signature
        signature = private_key.sign(
            self.hash.encode(),
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.MAX_LENGTH
            ),
            hashes.SHA256()
        )
        
        # Store and return base64-encoded signature
        self.signature = base64.b64encode(signature).decode('utf-8')
        return self.signature
    
    def verify_signature(self, public_key) -> bool:
        """
        Verify the block's signature.
        
        Args:
            public_key: RSA public key for verification
            
        Returns:
            True if signature is valid, False otherwise
        """
        if not self.signature:
            logger.warning("No signature to verify")
            return False
            
        try:
            signature_bytes = base64.b64decode(self.signature)
            public_key.verify(
                signature_bytes,
                self.hash.encode(),
                padding.PSS(
                    mgf=padding.MGF1(hashes.SHA256()),
                    salt_length=padding.PSS.MAX_LENGTH
                ),
                hashes.SHA256()
            )
            return True
        except InvalidSignature:
            logger.warning("Invalid signature")
            return False
        except Exception as e:
            logger.error(f"Error verifying signature: {e}")
            return False
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert block to dictionary representation.
        
        Returns:
            Dictionary of block data
        """
        return {
            "block_id": self.block_id,
            "timestamp": self.timestamp,
            "timestamp_readable": datetime.fromtimestamp(self.timestamp).isoformat(),
            "previous_hash": self.previous_hash,
            "hash": self.hash,
            "events": self.events,
            "agent_state_hash": self.agent_state_hash,
            "metadata": self.metadata,
            "signature": self.signature
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TraceBlock':
        """
        Create a block from dictionary representation.
        
        Args:
            data: Dictionary of block data
            
        Returns:
            TraceBlock instance
        """
        block = cls(
            events=data.get("events", []),
            timestamp=data.get("timestamp"),
            block_id=data.get("block_id"),
            previous_hash=data.get("previous_hash", ""),
            agent_state_hash=data.get("agent_state_hash"),
            metadata=data.get("metadata", {})
        )
        
        # Restore signature if present
        if "signature" in data:
            block.signature = data["signature"]
            
        # Set hash if present, otherwise recompute
        if "hash" in data:
            block.hash = data["hash"]
        else:
            block.compute_hash()
            
        return block


class TraceChainExporter:
    """
    Main class for exporting agent's reflective and schema change history as a chain of blocks.
    
    The TraceChain provides:
    1. Immutable, cryptographically linked blocks of agent events
    2. Digital signatures for non-repudiation
    3. Multiple export formats (JSON, binary, IPFS)
    4. Integration with existing logging systems
    5. Verification tools for auditing
    """
    
    def __init__(
        self,
        export_dir: Union[str, Path] = "trace_chain",
        private_key_path: Optional[str] = None,
        public_key_path: Optional[str] = None,
        ipfs_gateway: Optional[str] = None,
        events_per_block: int = 50,
        auto_export: bool = True,
        auto_export_interval_sec: int = 300
    ):
        """
        Initialize the trace chain exporter.
        
        Args:
            export_dir: Directory to store exported trace chains
            private_key_path: Path to RSA private key for signing (optional)
            public_key_path: Path to RSA public key for verification (optional)
            ipfs_gateway: IPFS gateway URL for publishing (optional)
            events_per_block: Maximum number of events in each block
            auto_export: Whether to automatically export the chain
            auto_export_interval_sec: How often to export if auto_export is enabled
        """
        self.export_dir = Path(export_dir)
        self.private_key_path = private_key_path
        self.public_key_path = public_key_path
        self.ipfs_gateway = ipfs_gateway
        self.events_per_block = events_per_block
        self.auto_export = auto_export
        self.auto_export_interval_sec = auto_export_interval_sec
        
        # Create export directory if it doesn't exist
        self.export_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize chain data
        self.blocks: List[TraceBlock] = []
        self.current_events: List[Dict[str, Any]] = []
        self.last_export_time = time.time()
        
        # Load keys if provided
        self.private_key = None
        self.public_key = None
        self._load_keys()
        
        # Integration with existing logging
        self.safety_exporter = None
        self.audit_log = AuditLog()
        
        # Auto-export setup
        if self.auto_export:
            self._setup_auto_export()
    
    def _load_keys(self):
        """Load cryptographic keys from specified paths."""
        if self.private_key_path:
            try:
                with open(self.private_key_path, 'rb') as f:
                    self.private_key = load_pem_private_key(
                        f.read(),
                        password=None,
                        backend=None
                    )
                logger.info("Loaded private key for signing")
            except Exception as e:
                logger.error(f"Failed to load private key: {e}")
                
        if self.public_key_path:
            try:
                with open(self.public_key_path, 'rb') as f:
                    self.public_key = load_pem_public_key(
                        f.read(),
                        backend=None
                    )
                logger.info("Loaded public key for verification")
            except Exception as e:
                logger.error(f"Failed to load public key: {e}")
    
    def _setup_auto_export(self):
        """Set up automatic chain export."""
        import threading
        
        def export_thread():
            while True:
                # Sleep for the specified interval
                time.sleep(self.auto_export_interval_sec)
                
                # Check if we need to export
                current_time = time.time()
                if (current_time - self.last_export_time >= self.auto_export_interval_sec and 
                    (len(self.current_events) > 0 or len(self.blocks) > 0)):
                    try:
                        if len(self.current_events) > 0:
                            self.create_block()
                        
                        if len(self.blocks) > 0:
                            self.export_chain()
                            
                        self.last_export_time = current_time
                    except Exception as e:
                        logger.error(f"Auto-export failed: {e}")
        
        # Start the export thread
        thread = threading.Thread(target=export_thread, daemon=True)
        thread.start()
        logger.info(f"Auto-export enabled, interval: {self.auto_export_interval_sec} seconds")
    
    def integrate_safety_exporter(self, safety_exporter: SafetyTraceExporter):
        """
        Integrate with an existing SafetyTraceExporter.
        
        Args:
            safety_exporter: SafetyTraceExporter instance
        """
        self.safety_exporter = safety_exporter
        logger.info("Integrated with SafetyTraceExporter")
    
    def log_event(
        self,
        event_type: str,
        component: str,
        description: str,
        data: Optional[Dict[str, Any]] = None,
        related_events: Optional[List[str]] = None,
        agent_state: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Log an event to the trace chain.
        
        Args:
            event_type: Type of event
            component: Component that generated the event
            description: Human-readable description of the event
            data: Additional structured data about the event
            related_events: IDs of related events
            agent_state: Current agent state for hash computation
            
        Returns:
            ID of the created event
        """
        event_id = str(uuid.uuid4())
        
        # Create event
        event = {
            "event_id": event_id,
            "event_type": event_type,
            "timestamp": time.time(),
            "timestamp_readable": datetime.now().isoformat(),
            "component": component,
            "description": description,
            "data": data or {},
            "related_events": related_events or []
        }
        
        # Compute agent state hash if provided
        if agent_state:
            agent_state_serialized = json.dumps(agent_state, sort_keys=True)
            event["agent_state_hash"] = hashlib.sha256(agent_state_serialized.encode()).hexdigest()
        
        # Add to current events
        self.current_events.append(event)
        
        # Also log to audit_log for compatibility
        self.audit_log.log_event(event_type, {
            "event_id": event_id,
            "component": component,
            "description": description,
            "data": data or {}
        })
        
        # Forward to safety exporter if integrated
        if self.safety_exporter:
            # Map common event types to SafetyEventType
            try:
                safety_event_type = self._map_to_safety_event_type(event_type)
                self.safety_exporter.log_event(
                    event_type=safety_event_type,
                    component=component,
                    description=description,
                    data=data,
                    related_events=related_events
                )
            except Exception as e:
                logger.warning(f"Failed to forward to safety exporter: {e}")
        
        # Create a new block if we've reached the maximum events per block
        if len(self.current_events) >= self.events_per_block:
            self.create_block()
            
        return event_id
    
    def _map_to_safety_event_type(self, event_type: str) -> Union[SafetyEventType, str]:
        """Map custom event types to SafetyEventType enum."""
        mapping = {
            "reflection": SafetyEventType.REFLECTION_TRIGGER,
            "reflection_outcome": SafetyEventType.REFLECTION_OUTCOME,
            "schema_mutation": SafetyEventType.SCHEMA_MUTATION,
            "schema_snapshot": SafetyEventType.SCHEMA_SNAPSHOT,
            "contradiction": SafetyEventType.CONTRADICTION_DETECTED,
            "contradiction_resolved": SafetyEventType.CONTRADICTION_RESOLVED,
        }
        
        return mapping.get(event_type, event_type)
    
    def log_reflection(
        self,
        trigger: str,
        reflection_content: str,
        outcome: Optional[str] = None,
        data: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Log a reflection event.
        
        Args:
            trigger: What triggered the reflection
            reflection_content: Content of the reflection
            outcome: Result of the reflection
            data: Additional structured data
            
        Returns:
            ID of the created event
        """
        event_data = {
            "trigger": trigger,
            "content": reflection_content
        }
        
        if outcome:
            event_data["outcome"] = outcome
            
        if data:
            event_data.update(data)
            
        return self.log_event(
            event_type="reflection",
            component="reflection_system",
            description=f"Reflection triggered by: {trigger}",
            data=event_data
        )
    
    def log_schema_mutation(
        self,
        mutation_type: str,
        affected_nodes: List[str],
        description: str,
        before_state: Optional[Dict[str, Any]] = None,
        after_state: Optional[Dict[str, Any]] = None,
        magnitude: Optional[float] = None
    ) -> str:
        """
        Log a schema mutation event.
        
        Args:
            mutation_type: Type of schema mutation
            affected_nodes: List of affected node IDs
            description: Description of the mutation
            before_state: Schema state before mutation
            after_state: Schema state after mutation
            magnitude: Magnitude of the mutation
            
        Returns:
            ID of the created event
        """
        event_data = {
            "mutation_type": mutation_type,
            "affected_nodes": affected_nodes
        }
        
        if before_state:
            # Compute hash rather than storing full state
            before_serialized = json.dumps(before_state, sort_keys=True)
            event_data["before_state_hash"] = hashlib.sha256(before_serialized.encode()).hexdigest()
            
        if after_state:
            # Compute hash rather than storing full state
            after_serialized = json.dumps(after_state, sort_keys=True)
            event_data["after_state_hash"] = hashlib.sha256(after_serialized.encode()).hexdigest()
            
        if magnitude is not None:
            event_data["magnitude"] = magnitude
            
        return self.log_event(
            event_type="schema_mutation",
            component="schema_system",
            description=description,
            data=event_data
        )
    
    def create_block(
        self,
        agent_state_hash: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> TraceBlock:
        """
        Create a new block in the trace chain from current events.
        
        Args:
            agent_state_hash: Hash of current agent state
            metadata: Additional block metadata
            
        Returns:
            The newly created TraceBlock
        """
        if not self.current_events:
            logger.warning("No events to create a block")
            return None
            
        # Get the hash of the last block, if any
        previous_hash = self.blocks[-1].hash if self.blocks else ""
        
        # Create a new block
        block = TraceBlock(
            events=self.current_events.copy(),
            previous_hash=previous_hash,
            agent_state_hash=agent_state_hash,
            metadata=metadata
        )
        
        # Sign the block if private key is available
        if self.private_key:
            block.sign(self.private_key)
            
        # Add to blocks and clear current events
        self.blocks.append(block)
        self.current_events = []
        
        logger.info(f"Created block {block.block_id} with {len(block.events)} events")
        return block
    
    def verify_chain(self) -> Tuple[bool, Optional[str]]:
        """
        Verify the integrity of the entire trace chain.
        
        Returns:
            Tuple of (is_valid, first_invalid_block_id)
        """
        if not self.blocks:
            logger.warning("No blocks to verify")
            return True, None
            
        previous_hash = ""
        
        for i, block in enumerate(self.blocks):
            # Recompute hash to ensure it hasn't been tampered with
            computed_hash = block.compute_hash()
            if computed_hash != block.hash:
                logger.error(f"Block {block.block_id} hash mismatch")
                return False, block.block_id
                
            # Verify link to previous block
            if i > 0 and block.previous_hash != previous_hash:
                logger.error(f"Block {block.block_id} has invalid previous hash")
                return False, block.block_id
                
            # Verify signature if available
            if block.signature and self.public_key:
                if not block.verify_signature(self.public_key):
                    logger.error(f"Block {block.block_id} has invalid signature")
                    return False, block.block_id
                    
            # Update previous hash for next iteration
            previous_hash = block.hash
            
        return True, None
    
    def export_chain(
        self,
        filepath: Optional[Union[str, Path]] = None,
        format: str = "json",
        publish_to_ipfs: bool = False
    ) -> Union[str, Tuple[str, Optional[str]]]:
        """
        Export the trace chain to a file.
        
        Args:
            filepath: Path to export to (generated if None)
            format: Export format (json, binary)
            publish_to_ipfs: Whether to publish to IPFS
            
        Returns:
            Filepath and IPFS hash (if published)
        """
        if not self.blocks:
            logger.warning("No blocks to export")
            return None
            
        # Create a filename if not provided
        if filepath is None:
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            filepath = self.export_dir / f"trace_chain_{timestamp}.{format.lower()}"
            
        filepath = Path(filepath)
        
        # Make sure parent directory exists
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # Export data preparation
        export_data = {
            "trace_chain_version": "1.0",
            "exported_at": datetime.now().isoformat(),
            "block_count": len(self.blocks),
            "total_events": sum(len(block.events) for block in self.blocks),
            "chain_hash": self.blocks[-1].hash,
            "blocks": [block.to_dict() for block in self.blocks]
        }
        
        # Export in the specified format
        try:
            if format.lower() == "json":
                with open(filepath, "w") as f:
                    json.dump(export_data, f, indent=2)
            elif format.lower() == "binary":
                # Binary format for more compact storage
                import pickle
                with open(filepath, "wb") as f:
                    pickle.dump(export_data, f)
            else:
                logger.error(f"Unsupported export format: {format}")
                return None
                
            logger.info(f"Exported trace chain with {len(self.blocks)} blocks to {filepath}")
            
            # Publish to IPFS if requested
            ipfs_hash = None
            if publish_to_ipfs:
                ipfs_hash = self._publish_to_ipfs(filepath)
                logger.info(f"Published to IPFS with hash: {ipfs_hash}")
                
            self.last_export_time = time.time()
            return (str(filepath), ipfs_hash) if publish_to_ipfs else str(filepath)
            
        except Exception as e:
            logger.error(f"Failed to export trace chain: {e}")
            return None
    
    def _publish_to_ipfs(self, filepath: Union[str, Path]) -> Optional[str]:
        """
        Publish a file to IPFS.
        
        Args:
            filepath: Path to the file to publish
            
        Returns:
            IPFS content identifier (CID) for the file
        """
        if not self.ipfs_gateway:
            logger.error("No IPFS gateway configured")
            return None
            
        try:
            # Try using ipfshttpclient if available
            try:
                client = ipfshttpclient.connect(self.ipfs_gateway)
                result = client.add(str(filepath))
                return result["Hash"]
            except (ImportError, ipfshttpclient.exceptions.ConnectionError):
                # Fall back to direct API calls
                with open(filepath, "rb") as f:
                    files = {"file": f}
                    response = requests.post(f"{self.ipfs_gateway}/api/v0/add", files=files)
                    if response.status_code == 200:
                        return response.json()["Hash"]
                    else:
                        logger.error(f"IPFS API error: {response.text}")
                        return None
        except Exception as e:
            logger.error(f"Failed to publish to IPFS: {e}")
            return None
    
    def load_chain(self, filepath: Union[str, Path]) -> bool:
        """
        Load a trace chain from a file.
        
        Args:
            filepath: Path to the trace chain file
            
        Returns:
            True if successful, False otherwise
        """
        filepath = Path(filepath)
        
        try:
            # Determine format from extension
            if filepath.suffix.lower() == ".json":
                with open(filepath, "r") as f:
                    data = json.load(f)
            elif filepath.suffix.lower() in (".bin", ".binary", ".pickle"):
                import pickle
                with open(filepath, "rb") as f:
                    data = pickle.load(f)
            else:
                logger.error(f"Unsupported file format: {filepath.suffix}")
                return False
                
            # Validate format
            if "trace_chain_version" not in data or "blocks" not in data:
                logger.error("Invalid trace chain file format")
                return False
                
            # Load blocks
            self.blocks = [TraceBlock.from_dict(block_data) for block_data in data["blocks"]]
            
            # Verify integrity
            is_valid, invalid_block = self.verify_chain()
            if not is_valid:
                logger.error(f"Loaded chain contains invalid block: {invalid_block}")
                
            logger.info(f"Loaded trace chain with {len(self.blocks)} blocks from {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load trace chain: {e}")
            return False
    
    def generate_keys(self, key_dir: Optional[Union[str, Path]] = None) -> Tuple[str, str]:
        """
        Generate a new RSA key pair for signing and verification.
        
        Args:
            key_dir: Directory to save keys (uses export_dir if None)
            
        Returns:
            Tuple of (private_key_path, public_key_path)
        """
        key_dir = Path(key_dir) if key_dir else self.export_dir
        key_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate key pair
        private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=2048
        )
        public_key = private_key.public_key()
        
        # Serialize keys
        private_bytes = private_key.private_bytes(
            encoding=Encoding.PEM,
            format=PrivateFormat.PKCS8,
            encryption_algorithm=NoEncryption()
        )
        
        public_bytes = public_key.public_bytes(
            encoding=Encoding.PEM,
            format=PublicFormat.SubjectPublicKeyInfo
        )
        
        # Save keys
        private_key_path = key_dir / "trace_chain_private.pem"
        public_key_path = key_dir / "trace_chain_public.pem"
        
        with open(private_key_path, "wb") as f:
            f.write(private_bytes)
            
        with open(public_key_path, "wb") as f:
            f.write(public_bytes)
            
        # Update instance variables
        self.private_key_path = str(private_key_path)
        self.public_key_path = str(public_key_path)
        self.private_key = private_key
        self.public_key = public_key
        
        logger.info(f"Generated new RSA key pair: {private_key_path}, {public_key_path}")
        return self.private_key_path, self.public_key_path
    
    def get_chain_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the current trace chain.
        
        Returns:
            Dictionary of statistics
        """
        if not self.blocks:
            return {
                "blocks": 0,
                "events": len(self.current_events),
                "pending_events": len(self.current_events),
                "chain_valid": None
            }
            
        # Count event types
        event_types = {}
        for block in self.blocks:
            for event in block.events:
                event_type = event.get("event_type", "unknown")
                event_types[event_type] = event_types.get(event_type, 0) + 1
                
        # Get time range
        first_block_time = datetime.fromtimestamp(self.blocks[0].timestamp)
        last_block_time = datetime.fromtimestamp(self.blocks[-1].timestamp)
        
        # Check validity
        is_valid, _ = self.verify_chain()
        
        return {
            "blocks": len(self.blocks),
            "events": sum(len(block.events) for block in self.blocks),
            "pending_events": len(self.current_events),
            "first_block_time": first_block_time.isoformat(),
            "last_block_time": last_block_time.isoformat(),
            "time_span_hours": (last_block_time - first_block_time).total_seconds() / 3600,
            "chain_hash": self.blocks[-1].hash,
            "event_types": event_types,
            "signed_blocks": sum(1 for block in self.blocks if block.signature),
            "chain_valid": is_valid
        }


def create_trace_chain_exporter(
    export_dir: str = "trace_chain",
    generate_keys: bool = True,
    events_per_block: int = 50,
    auto_export: bool = True
) -> TraceChainExporter:
    """
    Utility function to create and configure a TraceChainExporter.
    
    Args:
        export_dir: Directory to store exported trace chains
        generate_keys: Whether to generate RSA keys for signing
        events_per_block: Maximum number of events in each block
        auto_export: Whether to automatically export the chain
        
    Returns:
        Configured TraceChainExporter
    """
    # Create exporter
    exporter = TraceChainExporter(
        export_dir=export_dir,
        events_per_block=events_per_block,
        auto_export=auto_export
    )
    
    # Generate keys if requested
    if generate_keys:
        private_key_path, public_key_path = exporter.generate_keys()
        logger.info(f"Generated keys at {private_key_path} and {public_key_path}")
    
    return exporter


if __name__ == "__main__":
    # Simple demo and test
    logging.basicConfig(level=logging.INFO)
    
    # Create exporter
    exporter = create_trace_chain_exporter()
    
    # Log some test events
    for i in range(10):
        event_id = exporter.log_event(
            event_type="test_event",
            component="test_component",
            description=f"Test event {i}",
            data={"test_data": f"value_{i}"}
        )
        logger.info(f"Logged event {event_id}")
        
    # Create a block
    block = exporter.create_block()
    logger.info(f"Created block {block.block_id}")
    
    # Export the chain
    filepath = exporter.export_chain()
    logger.info(f"Exported chain to {filepath}")
    
    # Verify the chain
    is_valid, invalid_block = exporter.verify_chain()
    logger.info(f"Chain valid: {is_valid}")
    
    # Print stats
    stats = exporter.get_chain_stats()
    logger.info(f"Chain stats: {stats}") 