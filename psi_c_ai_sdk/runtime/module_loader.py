"""
Modular Runtime Bundles

This module implements a flexible component loader that allows developers to 
selectively enable specific functionality within the ΨC-AI SDK, helping to keep
the system lean by only loading the components that are needed.

Features:
- Module registry + feature toggles
- YAML manifest loader 
- Dependency resolver between subsystems
- Runtime enabling/disabling of components
"""

import os
import sys
import logging
import importlib
import inspect
import yaml
from typing import Dict, List, Set, Any, Optional, Callable, Type, Union, TypeVar
from enum import Enum
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# Type variable for component classes
T = TypeVar('T')


class ModuleStatus(Enum):
    """Status of a module in the system."""
    AVAILABLE = "available"      # Module is available but not loaded
    LOADED = "loaded"            # Module is loaded and ready
    DISABLED = "disabled"        # Module is explicitly disabled
    UNAVAILABLE = "unavailable"  # Module is not available (missing dependencies)
    ERROR = "error"              # Error occurred during loading


@dataclass
class ModuleInfo:
    """Information about a module in the system."""
    
    name: str
    description: str
    status: ModuleStatus = ModuleStatus.AVAILABLE
    dependencies: List[str] = field(default_factory=list)
    optional_dependencies: List[str] = field(default_factory=list)
    provides: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    import_path: str = ""
    class_name: str = ""
    instance: Any = None
    error_message: str = ""
    version: str = "1.0.0"
    config: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "name": self.name,
            "description": self.description,
            "status": self.status.value,
            "dependencies": self.dependencies,
            "optional_dependencies": self.optional_dependencies,
            "provides": self.provides,
            "tags": self.tags,
            "import_path": self.import_path,
            "class_name": self.class_name,
            "error_message": self.error_message,
            "version": self.version,
            "loaded": self.status == ModuleStatus.LOADED,
            "config": self.config
        }


class ModuleRegistry:
    """
    Registry for all modules in the system.
    
    The ModuleRegistry keeps track of all available modules, their dependencies,
    and their status. It provides methods for loading, enabling, and disabling
    modules as needed.
    """
    
    def __init__(self):
        """Initialize the module registry."""
        self.modules: Dict[str, ModuleInfo] = {}
        self.loaded_modules: Dict[str, Any] = {}
        self.default_config_path = os.path.join(
            os.path.dirname(__file__), "..", "config", "modules.yaml"
        )
        
    def register_module(self, module_info: ModuleInfo) -> bool:
        """
        Register a module with the registry.
        
        Args:
            module_info: Information about the module
            
        Returns:
            True if registration was successful
        """
        if module_info.name in self.modules:
            logger.warning(f"Module {module_info.name} already registered")
            return False
            
        self.modules[module_info.name] = module_info
        return True
        
    def load_from_manifest(self, manifest_path: Optional[str] = None) -> int:
        """
        Load module information from a YAML manifest.
        
        Args:
            manifest_path: Path to the manifest file, or None to use default
            
        Returns:
            Number of modules loaded
        """
        if manifest_path is None:
            manifest_path = self.default_config_path
            
        if not os.path.exists(manifest_path):
            logger.warning(f"Manifest file not found: {manifest_path}")
            return 0
            
        try:
            with open(manifest_path, 'r') as f:
                manifest = yaml.safe_load(f)
                
            if not isinstance(manifest, dict) or 'modules' not in manifest:
                logger.error(f"Invalid manifest format in {manifest_path}")
                return 0
                
            count = 0
            for module_data in manifest['modules']:
                try:
                    module_info = ModuleInfo(
                        name=module_data['name'],
                        description=module_data.get('description', ''),
                        dependencies=module_data.get('dependencies', []),
                        optional_dependencies=module_data.get('optional_dependencies', []),
                        provides=module_data.get('provides', []),
                        tags=module_data.get('tags', []),
                        import_path=module_data.get('import_path', ''),
                        class_name=module_data.get('class_name', ''),
                        version=module_data.get('version', '1.0.0'),
                        config=module_data.get('config', {})
                    )
                    
                    if self.register_module(module_info):
                        count += 1
                except KeyError as e:
                    logger.error(f"Missing required field in module definition: {e}")
                    
            logger.info(f"Loaded {count} modules from manifest")
            return count
        except Exception as e:
            logger.error(f"Error loading manifest: {e}")
            return 0
            
    def get_module(self, name: str) -> Optional[ModuleInfo]:
        """
        Get information about a module.
        
        Args:
            name: Name of the module
            
        Returns:
            ModuleInfo object or None if not found
        """
        return self.modules.get(name)
        
    def get_instance(self, name: str) -> Optional[Any]:
        """
        Get the loaded instance of a module.
        
        Args:
            name: Name of the module
            
        Returns:
            Module instance or None if not loaded
        """
        module_info = self.get_module(name)
        if module_info and module_info.status == ModuleStatus.LOADED:
            return module_info.instance
        return None
        
    def is_loaded(self, name: str) -> bool:
        """
        Check if a module is loaded.
        
        Args:
            name: Name of the module
            
        Returns:
            True if the module is loaded
        """
        module_info = self.get_module(name)
        return module_info is not None and module_info.status == ModuleStatus.LOADED
        
    def get_all_modules(self) -> Dict[str, ModuleInfo]:
        """
        Get all registered modules.
        
        Returns:
            Dictionary of module name to ModuleInfo
        """
        return self.modules.copy()
        
    def get_enabled_modules(self) -> Dict[str, ModuleInfo]:
        """
        Get all enabled modules.
        
        Returns:
            Dictionary of enabled module name to ModuleInfo
        """
        return {
            name: info for name, info in self.modules.items()
            if info.status == ModuleStatus.LOADED
        }
        
    def get_available_modules(self) -> Dict[str, ModuleInfo]:
        """
        Get all available modules.
        
        Returns:
            Dictionary of available module name to ModuleInfo
        """
        return {
            name: info for name, info in self.modules.items()
            if info.status in (ModuleStatus.AVAILABLE, ModuleStatus.LOADED)
        }
        
    def load_module(self, name: str, **kwargs) -> Optional[Any]:
        """
        Load a module by name.
        
        This method loads the module and all its dependencies.
        
        Args:
            name: Name of the module to load
            **kwargs: Additional arguments to pass to the module constructor
            
        Returns:
            Loaded module instance or None if loading failed
        """
        # Check if module exists
        module_info = self.get_module(name)
        if not module_info:
            logger.error(f"Module {name} not found")
            return None
            
        # Check if already loaded
        if module_info.status == ModuleStatus.LOADED:
            return module_info.instance
            
        # Check if explicitly disabled
        if module_info.status == ModuleStatus.DISABLED:
            logger.warning(f"Module {name} is disabled")
            return None
            
        # Check and load dependencies
        for dep_name in module_info.dependencies:
            dep_info = self.get_module(dep_name)
            if not dep_info:
                logger.error(f"Dependency {dep_name} for module {name} not found")
                module_info.status = ModuleStatus.UNAVAILABLE
                module_info.error_message = f"Dependency {dep_name} not found"
                return None
                
            # Load dependency if not already loaded
            if dep_info.status != ModuleStatus.LOADED:
                dep_instance = self.load_module(dep_name)
                if not dep_instance:
                    logger.error(f"Failed to load dependency {dep_name} for module {name}")
                    module_info.status = ModuleStatus.UNAVAILABLE
                    module_info.error_message = f"Failed to load dependency {dep_name}"
                    return None
        
        # Load optional dependencies
        optional_deps = {}
        for dep_name in module_info.optional_dependencies:
            dep_info = self.get_module(dep_name)
            if dep_info and dep_info.status != ModuleStatus.DISABLED:
                dep_instance = self.load_module(dep_name)
                if dep_instance:
                    optional_deps[dep_name] = dep_instance
        
        # Try to import and instantiate the module
        try:
            if not module_info.import_path:
                raise ValueError(f"No import path specified for module {name}")
                
            module = importlib.import_module(module_info.import_path)
            
            if not module_info.class_name:
                # If no class name specified, use the module as is
                instance = module
            else:
                # Get the class from the module
                if not hasattr(module, module_info.class_name):
                    raise ValueError(
                        f"Class {module_info.class_name} not found in module {module_info.import_path}"
                    )
                    
                cls = getattr(module, module_info.class_name)
                
                # Combine kwargs with module config
                init_kwargs = {**module_info.config, **kwargs}
                
                # Instantiate the class
                instance = cls(**init_kwargs)
            
            # Update module status and instance
            module_info.instance = instance
            module_info.status = ModuleStatus.LOADED
            
            logger.info(f"Successfully loaded module {name}")
            return instance
            
        except Exception as e:
            logger.error(f"Error loading module {name}: {e}")
            module_info.status = ModuleStatus.ERROR
            module_info.error_message = str(e)
            return None
            
    def enable_module(self, name: str, **kwargs) -> bool:
        """
        Enable a module by name.
        
        Args:
            name: Name of the module to enable
            **kwargs: Additional arguments to pass to the module constructor
            
        Returns:
            True if the module was enabled successfully
        """
        module_info = self.get_module(name)
        if not module_info:
            logger.error(f"Module {name} not found")
            return False
            
        # If previously disabled, reset status
        if module_info.status == ModuleStatus.DISABLED:
            module_info.status = ModuleStatus.AVAILABLE
            
        # Load the module
        instance = self.load_module(name, **kwargs)
        return instance is not None
        
    def disable_module(self, name: str) -> bool:
        """
        Disable a module by name.
        
        Args:
            name: Name of the module to disable
            
        Returns:
            True if the module was disabled successfully
        """
        module_info = self.get_module(name)
        if not module_info:
            logger.error(f"Module {name} not found")
            return False
            
        # Check for dependents
        dependents = self.get_dependent_modules(name)
        if dependents:
            logger.error(
                f"Cannot disable module {name} because it is required by: "
                f"{', '.join(dependents)}"
            )
            return False
            
        # Mark as disabled
        module_info.status = ModuleStatus.DISABLED
        module_info.instance = None
        
        logger.info(f"Disabled module {name}")
        return True
        
    def get_dependent_modules(self, name: str) -> List[str]:
        """
        Get modules that depend on the given module.
        
        Args:
            name: Name of the module
            
        Returns:
            List of module names that depend on this module
        """
        dependents = []
        for module_name, module_info in self.modules.items():
            if (name in module_info.dependencies and 
                module_info.status == ModuleStatus.LOADED):
                dependents.append(module_name)
        return dependents
        
    def resolve_dependencies(self, name: str) -> List[str]:
        """
        Resolve all dependencies for a module.
        
        Args:
            name: Name of the module
            
        Returns:
            Ordered list of modules to load (dependencies first)
        """
        # Check if module exists
        module_info = self.get_module(name)
        if not module_info:
            logger.error(f"Module {name} not found")
            return []
            
        # Track visited modules to avoid cycles
        visited = set()
        result = []
        
        def visit(module_name: str) -> None:
            """Visit a module and its dependencies."""
            if module_name in visited:
                return
                
            visited.add(module_name)
            
            mod_info = self.get_module(module_name)
            if not mod_info:
                return
                
            # Visit dependencies first
            for dep in mod_info.dependencies:
                visit(dep)
                
            result.append(module_name)
            
        visit(name)
        return result
        
    def load_bundle(self, bundle_name: str, **kwargs) -> Dict[str, Any]:
        """
        Load a bundle of modules defined in the manifest.
        
        Args:
            bundle_name: Name of the bundle to load
            **kwargs: Additional arguments to pass to module constructors
            
        Returns:
            Dictionary of loaded module instances
        """
        try:
            with open(self.default_config_path, 'r') as f:
                manifest = yaml.safe_load(f)
                
            if 'bundles' not in manifest or bundle_name not in manifest['bundles']:
                logger.error(f"Bundle {bundle_name} not found in manifest")
                return {}
                
            bundle = manifest['bundles'][bundle_name]
            logger.info(f"Loading bundle {bundle_name} with {len(bundle)} modules")
            
            results = {}
            for module_name in bundle:
                instance = self.load_module(module_name, **kwargs)
                if instance:
                    results[module_name] = instance
                    
            return results
        except Exception as e:
            logger.error(f"Error loading bundle {bundle_name}: {e}")
            return {}
            
    def get_component(self, component_type: Type[T], module_name: Optional[str] = None) -> Optional[T]:
        """
        Get a component of the specified type.
        
        Args:
            component_type: Type of component to get
            module_name: Optional module name to look in, or None to search all modules
            
        Returns:
            Component instance or None if not found
        """
        if module_name:
            # Look in specific module
            instance = self.get_instance(module_name)
            if isinstance(instance, component_type):
                return instance
            return None
        else:
            # Search all modules
            for info in self.modules.values():
                if info.status == ModuleStatus.LOADED and isinstance(info.instance, component_type):
                    return info.instance
                    
            return None
            
    def create_module_info(
        self,
        obj: Any,
        name: Optional[str] = None,
        description: Optional[str] = None,
        dependencies: Optional[List[str]] = None,
        tags: Optional[List[str]] = None
    ) -> ModuleInfo:
        """
        Create ModuleInfo from a Python object.
        
        Args:
            obj: Python class or module
            name: Optional name override
            description: Optional description override
            dependencies: Optional dependencies override
            tags: Optional tags
            
        Returns:
            ModuleInfo object
        """
        if inspect.isclass(obj):
            module_path = obj.__module__
            class_name = obj.__name__
            desc = description or obj.__doc__ or ""
            name = name or obj.__name__
        else:
            module_path = obj.__name__
            class_name = ""
            desc = description or getattr(obj, "__doc__", "") or ""
            name = name or obj.__name__.split(".")[-1]
            
        return ModuleInfo(
            name=name,
            description=desc.strip(),
            dependencies=dependencies or [],
            tags=tags or [],
            import_path=module_path,
            class_name=class_name
        )


# Global module registry instance
_registry = None


def get_module_registry() -> ModuleRegistry:
    """
    Get the global module registry instance.
    
    Returns:
        ModuleRegistry instance
    """
    global _registry
    if _registry is None:
        _registry = ModuleRegistry()
        _registry.load_from_manifest()
    return _registry 