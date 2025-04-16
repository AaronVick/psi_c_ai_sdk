"""
API Standardization Layer
------------------------

This module defines consistent interfaces for all SDK components, enabling
modularity and third-party extensions.

Key features:
- Interface contracts for each module type
- Versioned API specifications
- Compatibility layer for legacy code
- Extension points documentation

Mathematical Invariants:
- Feature toggles that preserve mathematical guarantees:
  For any Feature f, ToggleOff(f) => ΨC remains well-defined
"""

import abc
import inspect
import logging
import warnings
from enum import Enum
from typing import Dict, List, Set, Any, Optional, Callable, Union, Type, TypeVar, Protocol, runtime_checkable
from dataclasses import dataclass, field
import importlib
import semver
import importlib.metadata

# Setup logger
logger = logging.getLogger(__name__)

# SDK Version
try:
    SDK_VERSION = importlib.metadata.version("psi_c_ai_sdk")
except importlib.metadata.PackageNotFoundError:
    SDK_VERSION = "0.1.0"  # Default development version

# API Version - follows semver
API_VERSION = "1.0.0"


class ComponentType(Enum):
    """Types of components in the ΨC-AI SDK."""
    CORE = "core"                # Core components (memory, schema, reflection)
    COGNITIVE = "cognitive"      # Cognitive processing components
    SENSORY = "sensory"          # Input/perception components
    OUTPUT = "output"            # Response/action components
    SAFETY = "safety"            # Safety and alignment components
    METRICS = "metrics"          # Monitoring and measurement components
    UTILITY = "utility"          # Utility and helper components
    EXTENSION = "extension"      # Third-party extensions
    EXPERIMENTAL = "experimental"  # Experimental components


class ApiLevel(Enum):
    """API stability and compatibility levels."""
    STABLE = "stable"            # Stable API with backward compatibility guarantees
    BETA = "beta"                # Beta API with limited compatibility guarantees
    ALPHA = "alpha"              # Alpha API with no compatibility guarantees
    EXPERIMENTAL = "experimental"  # Experimental API that may change at any time
    DEPRECATED = "deprecated"    # Deprecated API that will be removed in future versions


@runtime_checkable
class PsiComponent(Protocol):
    """Base protocol for all ΨC components."""
    
    @property
    def component_id(self) -> str:
        """Unique identifier for the component."""
        ...
    
    @property
    def component_type(self) -> ComponentType:
        """Type of the component."""
        ...
    
    @property
    def api_level(self) -> ApiLevel:
        """API stability level."""
        ...
    
    @property
    def required_components(self) -> List[str]:
        """List of component IDs that this component depends on."""
        ...


@dataclass
class ComponentMetadata:
    """Metadata for a ΨC component."""
    id: str
    name: str
    description: str
    version: str
    api_level: ApiLevel
    component_type: ComponentType
    authors: List[str] = field(default_factory=list)
    required_components: List[str] = field(default_factory=list)
    optional_components: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    documentation_url: Optional[str] = None
    source_url: Optional[str] = None
    license: str = "MIT"
    citation: Optional[str] = None
    
    def is_compatible_with(self, other_version: str) -> bool:
        """Check if this component is compatible with the given version."""
        try:
            return semver.compare(self.version, other_version) >= 0
        except ValueError:
            return False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metadata to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "version": self.version,
            "api_level": self.api_level.value,
            "component_type": self.component_type.value,
            "authors": self.authors,
            "required_components": self.required_components,
            "optional_components": self.optional_components,
            "tags": self.tags,
            "documentation_url": self.documentation_url,
            "source_url": self.source_url,
            "license": self.license,
            "citation": self.citation
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ComponentMetadata':
        """Create metadata from dictionary."""
        return cls(
            id=data["id"],
            name=data["name"],
            description=data["description"],
            version=data["version"],
            api_level=ApiLevel(data["api_level"]),
            component_type=ComponentType(data["component_type"]),
            authors=data.get("authors", []),
            required_components=data.get("required_components", []),
            optional_components=data.get("optional_components", []),
            tags=data.get("tags", []),
            documentation_url=data.get("documentation_url"),
            source_url=data.get("source_url"),
            license=data.get("license", "MIT"),
            citation=data.get("citation")
        )


class FeatureToggle:
    """
    Feature toggle for ΨC components.
    
    This class allows features to be toggled on/off while preserving
    mathematical guarantees about the system.
    """
    
    _toggled_features: Dict[str, bool] = {}
    _feature_metadata: Dict[str, Dict[str, Any]] = {}
    _toggle_callbacks: Dict[str, List[Callable[[bool], None]]] = {}
    
    @classmethod
    def register_feature(cls, feature_id: str, default_state: bool = True, 
                        description: str = "", affects_psi_c: bool = False,
                        component_id: Optional[str] = None) -> None:
        """
        Register a new feature toggle.
        
        Args:
            feature_id: Unique identifier for the feature
            default_state: Default state (True=enabled, False=disabled)
            description: Description of the feature
            affects_psi_c: Whether toggling this feature affects ΨC well-formedness
            component_id: ID of the component that owns this feature
        """
        if feature_id in cls._toggled_features:
            logger.warning(f"Feature {feature_id} already registered, overwriting")
            
        cls._toggled_features[feature_id] = default_state
        cls._feature_metadata[feature_id] = {
            "description": description,
            "affects_psi_c": affects_psi_c,
            "component_id": component_id,
            "default_state": default_state
        }
        cls._toggle_callbacks[feature_id] = []
        
        logger.debug(f"Registered feature {feature_id}, default state: {default_state}")
    
    @classmethod
    def toggle(cls, feature_id: str, state: bool) -> bool:
        """
        Toggle a feature on or off.
        
        Args:
            feature_id: ID of the feature to toggle
            state: New state (True=enabled, False=disabled)
            
        Returns:
            True if the toggle was successful, False otherwise
        """
        if feature_id not in cls._toggled_features:
            logger.warning(f"Feature {feature_id} not registered")
            return False
            
        # Check if this toggle affects ΨC well-formedness
        if cls._feature_metadata[feature_id].get("affects_psi_c", False) and not state:
            logger.warning(f"Toggling off {feature_id} may affect ΨC well-formedness")
            
        previous_state = cls._toggled_features[feature_id]
        cls._toggled_features[feature_id] = state
        
        # Call callbacks if state changed
        if previous_state != state:
            for callback in cls._toggle_callbacks.get(feature_id, []):
                try:
                    callback(state)
                except Exception as e:
                    logger.error(f"Error in toggle callback for {feature_id}: {e}")
        
        logger.debug(f"Toggled feature {feature_id} to {state}")
        return True
    
    @classmethod
    def is_enabled(cls, feature_id: str) -> bool:
        """Check if a feature is enabled."""
        return cls._toggled_features.get(feature_id, False)
    
    @classmethod
    def register_toggle_callback(cls, feature_id: str, callback: Callable[[bool], None]) -> bool:
        """
        Register a callback to be called when a feature is toggled.
        
        Args:
            feature_id: ID of the feature
            callback: Callback function that takes the new state as an argument
            
        Returns:
            True if the callback was registered, False otherwise
        """
        if feature_id not in cls._toggled_features:
            logger.warning(f"Feature {feature_id} not registered")
            return False
            
        cls._toggle_callbacks.setdefault(feature_id, []).append(callback)
        return True
    
    @classmethod
    def reset_to_defaults(cls) -> None:
        """Reset all features to their default states."""
        for feature_id, metadata in cls._feature_metadata.items():
            cls.toggle(feature_id, metadata["default_state"])
    
    @classmethod
    def get_all_features(cls) -> Dict[str, Dict[str, Any]]:
        """Get all registered features with their states and metadata."""
        result = {}
        for feature_id in cls._toggled_features:
            result[feature_id] = {
                "state": cls._toggled_features[feature_id],
                **cls._feature_metadata[feature_id]
            }
        return result
    
    @classmethod
    def get_enabled_features(cls) -> List[str]:
        """Get list of enabled features."""
        return [f for f, state in cls._toggled_features.items() if state]


class ApiVersionError(Exception):
    """Exception raised for API version incompatibility."""
    pass


class ComponentRegistry:
    """
    Registry for all ΨC components.
    
    This registry keeps track of all components in the system, their
    versions, dependencies, and compatibility.
    """
    
    _components: Dict[str, Any] = {}
    _component_metadata: Dict[str, ComponentMetadata] = {}
    _component_interfaces: Dict[ComponentType, Set[str]] = {}
    
    @classmethod
    def register_component(cls, component: Any, metadata: ComponentMetadata) -> bool:
        """
        Register a component in the registry.
        
        Args:
            component: The component to register
            metadata: Metadata for the component
            
        Returns:
            True if registration was successful, False otherwise
        """
        if metadata.id in cls._components:
            logger.warning(f"Component {metadata.id} already registered, overwriting")
            
        cls._components[metadata.id] = component
        cls._component_metadata[metadata.id] = metadata
        
        # Register component interface
        cls._component_interfaces.setdefault(metadata.component_type, set()).add(metadata.id)
        
        logger.debug(f"Registered component {metadata.id} of type {metadata.component_type.value}")
        return True
    
    @classmethod
    def get_component(cls, component_id: str) -> Any:
        """Get a component by ID."""
        return cls._components.get(component_id)
    
    @classmethod
    def get_metadata(cls, component_id: str) -> Optional[ComponentMetadata]:
        """Get component metadata by ID."""
        return cls._component_metadata.get(component_id)
    
    @classmethod
    def get_components_by_type(cls, component_type: ComponentType) -> Dict[str, Any]:
        """Get all components of a specific type."""
        component_ids = cls._component_interfaces.get(component_type, set())
        return {cid: cls._components[cid] for cid in component_ids if cid in cls._components}
    
    @classmethod
    def check_compatibility(cls, component_id: str, required_version: str) -> bool:
        """Check if a component is compatible with the required version."""
        metadata = cls.get_metadata(component_id)
        if not metadata:
            return False
        return metadata.is_compatible_with(required_version)
    
    @classmethod
    def validate_dependencies(cls, component_id: str) -> List[str]:
        """
        Validate that all dependencies of a component are registered.
        
        Args:
            component_id: ID of the component to validate
            
        Returns:
            List of missing dependencies, empty if all dependencies are satisfied
        """
        metadata = cls.get_metadata(component_id)
        if not metadata:
            return ["Component not registered"]
            
        missing = []
        for dep_id in metadata.required_components:
            if dep_id not in cls._components:
                missing.append(dep_id)
                
        return missing
    
    @classmethod
    def get_all_components(cls) -> Dict[str, ComponentMetadata]:
        """Get all registered components with their metadata."""
        return {cid: metadata for cid, metadata in cls._component_metadata.items()}


def component(
    id: str,
    name: str,
    description: str,
    component_type: ComponentType,
    version: str = "0.1.0",
    api_level: ApiLevel = ApiLevel.ALPHA,
    **kwargs
) -> Callable[[Type], Type]:
    """
    Decorator to register a class as a ΨC component.
    
    Args:
        id: Unique identifier for the component
        name: Human-readable name
        description: Description of the component
        component_type: Type of the component
        version: Version string (semver)
        api_level: API stability level
        **kwargs: Additional metadata fields
        
    Returns:
        Decorated class
    """
    def decorator(cls: Type) -> Type:
        # Create metadata
        metadata = ComponentMetadata(
            id=id,
            name=name,
            description=description,
            version=version,
            api_level=api_level,
            component_type=component_type,
            **kwargs
        )
        
        # Add metadata to class
        cls._component_metadata = metadata
        
        # Define property methods
        def component_id(self) -> str:
            return metadata.id
        
        def component_type_prop(self) -> ComponentType:
            return metadata.component_type
        
        def api_level_prop(self) -> ApiLevel:
            return metadata.api_level
        
        def required_components(self) -> List[str]:
            return metadata.required_components
        
        # Add properties to class if not already defined
        if not hasattr(cls, "component_id"):
            cls.component_id = property(component_id)
        
        if not hasattr(cls, "component_type"):
            cls.component_type = property(component_type_prop)
        
        if not hasattr(cls, "api_level"):
            cls.api_level = property(api_level_prop)
        
        if not hasattr(cls, "required_components"):
            cls.required_components = property(required_components)
        
        # Register with component registry (deferred until instantiation)
        original_init = cls.__init__
        
        def init_and_register(self, *args, **kwargs):
            original_init(self, *args, **kwargs)
            # Register instance with registry
            ComponentRegistry.register_component(self, metadata)
        
        cls.__init__ = init_and_register
        
        return cls
    
    return decorator


def deprecated(reason: str, removal_version: Optional[str] = None) -> Callable:
    """
    Decorator to mark functions or classes as deprecated.
    
    Args:
        reason: Reason for deprecation
        removal_version: Version when this will be removed
        
    Returns:
        Decorated function or class
    """
    def decorator(obj: Any) -> Any:
        message = f"Deprecated: {reason}"
        if removal_version:
            message += f" (will be removed in version {removal_version})"
        
        if isinstance(obj, type):
            # Class decorator
            original_init = obj.__init__
            
            def init_with_warning(self, *args, **kwargs):
                warnings.warn(message, DeprecationWarning, stacklevel=2)
                original_init(self, *args, **kwargs)
            
            obj.__init__ = init_with_warning
            obj._deprecated = True
            obj._deprecation_reason = reason
            obj._removal_version = removal_version
            
            return obj
        else:
            # Function decorator
            def wrapper(*args, **kwargs):
                warnings.warn(message, DeprecationWarning, stacklevel=2)
                return obj(*args, **kwargs)
            
            wrapper._deprecated = True
            wrapper._deprecation_reason = reason
            wrapper._removal_version = removal_version
            
            import functools
            return functools.wraps(obj)(wrapper)
    
    return decorator


def experimental(feature_id: Optional[str] = None) -> Callable:
    """
    Decorator to mark functions or classes as experimental.
    
    Args:
        feature_id: Optional feature ID for toggle control
        
    Returns:
        Decorated function or class
    """
    def decorator(obj: Any) -> Any:
        nonlocal feature_id
        
        # Generate feature ID if not provided
        if not feature_id:
            if isinstance(obj, type):
                feature_id = f"experimental.{obj.__module__}.{obj.__name__}"
            else:
                feature_id = f"experimental.{obj.__module__}.{obj.__qualname__}"
        
        # Register feature toggle
        FeatureToggle.register_feature(
            feature_id=feature_id,
            default_state=True,
            description=f"Experimental feature: {obj.__name__}",
            affects_psi_c=False
        )
        
        if isinstance(obj, type):
            # Class decorator
            original_init = obj.__init__
            
            def init_with_warning(self, *args, **kwargs):
                if not FeatureToggle.is_enabled(feature_id):
                    raise NotImplementedError(
                        f"Experimental feature {obj.__name__} is disabled. "
                        f"Enable it with FeatureToggle.toggle('{feature_id}', True)"
                    )
                warnings.warn(
                    f"{obj.__name__} is an experimental feature and may change in future versions",
                    UserWarning, stacklevel=2
                )
                original_init(self, *args, **kwargs)
            
            obj.__init__ = init_with_warning
            obj._experimental = True
            obj._feature_id = feature_id
            
            return obj
        else:
            # Function decorator
            def wrapper(*args, **kwargs):
                if not FeatureToggle.is_enabled(feature_id):
                    raise NotImplementedError(
                        f"Experimental feature {obj.__name__} is disabled. "
                        f"Enable it with FeatureToggle.toggle('{feature_id}', True)"
                    )
                warnings.warn(
                    f"{obj.__name__} is an experimental feature and may change in future versions",
                    UserWarning, stacklevel=2
                )
                return obj(*args, **kwargs)
            
            wrapper._experimental = True
            wrapper._feature_id = feature_id
            
            import functools
            return functools.wraps(obj)(wrapper)
    
    return decorator


class CompatibilityLayer:
    """
    Compatibility layer for legacy code.
    
    This class provides adapters and shims to support legacy code
    with the new API interfaces.
    """
    
    _adapters: Dict[str, Callable] = {}
    
    @classmethod
    def register_adapter(cls, source_id: str, target_id: str, 
                        adapter: Callable) -> None:
        """
        Register an adapter between components.
        
        Args:
            source_id: ID of the source component
            target_id: ID of the target component
            adapter: Adapter function or class
        """
        adapter_id = f"{source_id}->{target_id}"
        cls._adapters[adapter_id] = adapter
        logger.debug(f"Registered adapter from {source_id} to {target_id}")
    
    @classmethod
    def get_adapter(cls, source_id: str, target_id: str) -> Optional[Callable]:
        """
        Get an adapter between components.
        
        Args:
            source_id: ID of the source component
            target_id: ID of the target component
            
        Returns:
            Adapter function or class, or None if not found
        """
        adapter_id = f"{source_id}->{target_id}"
        return cls._adapters.get(adapter_id)
    
    @classmethod
    def adapt(cls, source: Any, target_id: str) -> Any:
        """
        Adapt a component to another interface.
        
        Args:
            source: Source component
            target_id: Target component ID
            
        Returns:
            Adapted component
        """
        if not hasattr(source, "component_id"):
            raise ValueError("Source object is not a valid component")
            
        source_id = source.component_id
        adapter = cls.get_adapter(source_id, target_id)
        
        if not adapter:
            raise ValueError(f"No adapter found from {source_id} to {target_id}")
            
        return adapter(source)


class ExtensionPoint:
    """
    Interface for extension points in the ΨC-AI SDK.
    
    Extension points allow third-party components to extend
    the functionality of the SDK.
    """
    
    _extension_points: Dict[str, Any] = {}
    _extension_metadata: Dict[str, Dict[str, Any]] = {}
    _extensions: Dict[str, Dict[str, Any]] = {}
    
    @classmethod
    def register_extension_point(cls, point_id: str, interface: Any,
                               description: str, owner_component: str) -> None:
        """
        Register an extension point.
        
        Args:
            point_id: Unique identifier for the extension point
            interface: Interface that extensions must implement
            description: Description of the extension point
            owner_component: ID of the component that owns this extension point
        """
        if point_id in cls._extension_points:
            logger.warning(f"Extension point {point_id} already registered, overwriting")
            
        cls._extension_points[point_id] = interface
        cls._extension_metadata[point_id] = {
            "description": description,
            "owner_component": owner_component
        }
        cls._extensions[point_id] = {}
        
        logger.debug(f"Registered extension point {point_id}")
    
    @classmethod
    def register_extension(cls, point_id: str, extension_id: str, 
                         extension: Any, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Register an extension for an extension point.
        
        Args:
            point_id: ID of the extension point
            extension_id: Unique identifier for the extension
            extension: The extension implementation
            metadata: Additional metadata for the extension
            
        Returns:
            True if registration was successful, False otherwise
        """
        if point_id not in cls._extension_points:
            logger.warning(f"Extension point {point_id} not registered")
            return False
            
        # Check if extension implements the required interface
        interface = cls._extension_points[point_id]
        if interface is not None and not isinstance(extension, interface):
            logger.warning(
                f"Extension {extension_id} does not implement required interface for {point_id}"
            )
            return False
            
        cls._extensions[point_id][extension_id] = {
            "extension": extension,
            "metadata": metadata or {}
        }
        
        logger.debug(f"Registered extension {extension_id} for point {point_id}")
        return True
    
    @classmethod
    def get_extension_point(cls, point_id: str) -> Optional[Any]:
        """Get the interface for an extension point."""
        return cls._extension_points.get(point_id)
    
    @classmethod
    def get_extensions(cls, point_id: str) -> Dict[str, Any]:
        """Get all extensions for an extension point."""
        if point_id not in cls._extensions:
            return {}
        return {eid: ext["extension"] for eid, ext in cls._extensions[point_id].items()}
    
    @classmethod
    def get_extension(cls, point_id: str, extension_id: str) -> Optional[Any]:
        """Get a specific extension for an extension point."""
        extensions = cls._extensions.get(point_id, {})
        if extension_id not in extensions:
            return None
        return extensions[extension_id]["extension"]
    
    @classmethod
    def get_all_extension_points(cls) -> Dict[str, Dict[str, Any]]:
        """Get all registered extension points with their metadata."""
        result = {}
        for point_id in cls._extension_points:
            result[point_id] = {
                "interface": cls._extension_points[point_id],
                **cls._extension_metadata[point_id],
                "extensions": list(cls._extensions.get(point_id, {}).keys())
            }
        return result


# Register core extension points
ExtensionPoint.register_extension_point(
    point_id="psi_c.reflection.strategy",
    interface=None,  # Will be defined by Reflection module
    description="Strategies for reflection cycle execution",
    owner_component="psi_c.core.reflection"
)

ExtensionPoint.register_extension_point(
    point_id="psi_c.schema.mutation",
    interface=None,  # Will be defined by Schema module
    description="Schema mutation operators",
    owner_component="psi_c.core.schema"
)

ExtensionPoint.register_extension_point(
    point_id="psi_c.memory.encoding",
    interface=None,  # Will be defined by Memory module
    description="Memory encoding strategies",
    owner_component="psi_c.core.memory"
)

ExtensionPoint.register_extension_point(
    point_id="psi_c.safety.monitors",
    interface=None,  # Will be defined by Safety module
    description="Safety monitoring components",
    owner_component="psi_c.safety"
)

# Register feature toggles for core features
FeatureToggle.register_feature(
    feature_id="psi_c.reflection.auto",
    default_state=True,
    description="Automatic reflection cycle triggering",
    affects_psi_c=True,
    component_id="psi_c.core.reflection"
)

FeatureToggle.register_feature(
    feature_id="psi_c.schema.auto_mutation",
    default_state=True,
    description="Automatic schema mutation",
    affects_psi_c=True,
    component_id="psi_c.core.schema"
)

FeatureToggle.register_feature(
    feature_id="psi_c.memory.forgetting",
    default_state=True,
    description="Memory forgetting mechanism",
    affects_psi_c=False,
    component_id="psi_c.core.memory"
)

# Utility functions for API documentation
def get_api_documentation() -> Dict[str, Any]:
    """
    Get comprehensive API documentation.
    
    Returns:
        Documentation dictionary with components, extension points, and features
    """
    return {
        "sdk_version": SDK_VERSION,
        "api_version": API_VERSION,
        "components": {cid: metadata.to_dict() for cid, metadata in ComponentRegistry.get_all_components().items()},
        "extension_points": ExtensionPoint.get_all_extension_points(),
        "features": FeatureToggle.get_all_features()
    } 