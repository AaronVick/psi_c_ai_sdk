#!/usr/bin/env python3
"""
ΨC-AI SDK Developer Onboarding Wizard

This module provides an interactive CLI wizard to guide new developers through
setting up and configuring a ΨC-AI agent, including selecting goals, alignment schema,
reflection thresholds, runtime mode, and other critical parameters.
"""

import os
import sys
import json
import yaml
import inquirer
import colorama
from colorama import Fore, Style
from typing import Dict, List, Optional, Any, Union, Tuple
from pathlib import Path

# Initialize colorama for cross-platform colored terminal output
colorama.init()

class OnboardingWizard:
    """
    Interactive wizard for configuring a new ΨC-AI agent.
    Guides developers through the process of setting up all necessary
    parameters for agent initialization.
    """
    
    # Constants for configuration
    DEFAULT_CONFIG_DIR = Path(__file__).parent.parent / "config"
    TEMPLATES_DIR = DEFAULT_CONFIG_DIR / "templates"
    DEFAULT_CONFIG_FILE = "agent_config.json"
    
    # Core parameters with defaults
    DEFAULT_PARAMS = {
        "reflection_depth": 3,
        "activation_threshold": 0.65,
        "entropy_tolerance": 0.3,
        "schema_logging_interval": 10,
        "runtime_mode": "interactive",
        "safety_mode": "standard"
    }
    
    # Available profiles with predefined goal vectors
    PROFILES = {
        "assistant": {
            "name": "General Assistant",
            "description": "Helpful, harmless, and honest general-purpose assistant",
            "goal_vector": {
                "helpfulness": 0.9,
                "accuracy": 0.85,
                "harmlessness": 0.95,
                "coherence": 0.8
            }
        },
        "researcher": {
            "name": "Research Assistant",
            "description": "Focused on accuracy, thoroughness, and scientific integrity",
            "goal_vector": {
                "accuracy": 0.95,
                "thoroughness": 0.9,
                "scientific_integrity": 0.95,
                "explanation_clarity": 0.85
            }
        },
        "creative": {
            "name": "Creative Partner",
            "description": "Emphasis on creativity, originality, and engaging interactions",
            "goal_vector": {
                "creativity": 0.9,
                "originality": 0.85,
                "engagement": 0.8,
                "coherence": 0.75
            }
        },
        "ethical_advisor": {
            "name": "Ethical Advisor",
            "description": "Specialized in ethical reasoning with strong alignment to human values",
            "goal_vector": {
                "ethical_reasoning": 0.95,
                "value_alignment": 0.9,
                "explanation_clarity": 0.85,
                "nuance": 0.8
            }
        },
        "custom": {
            "name": "Custom Profile",
            "description": "Define your own goal vector and parameters",
            "goal_vector": {}
        }
    }
    
    # Runtime modes with descriptions
    RUNTIME_MODES = {
        "interactive": "Full interaction mode with ongoing reflection",
        "safe_mode": "Restricted capabilities with additional safety checks",
        "shadowbox": "Isolated environment for testing without external effects",
        "benchmark": "Performance measurement mode with metrics collection"
    }
    
    # Available modules that can be enabled
    def __init__(self):
        """Initialize the onboarding wizard"""
        self.config: Dict[str, Any] = {}
        self._load_available_modules()
        self._welcome_message()
    
    def _load_available_modules(self):
        """Load available modules from the modules manifest file"""
        try:
            modules_file = self.DEFAULT_CONFIG_DIR / "modules.yaml"
            if modules_file.exists():
                with open(modules_file, 'r') as f:
                    modules_data = yaml.safe_load(f)
                    
                self.available_modules = {}
                self.available_bundles = {}
                
                # Extract modules
                if 'modules' in modules_data:
                    for module_id, module_info in modules_data['modules'].items():
                        self.available_modules[module_id] = {
                            'name': module_info.get('name', module_id),
                            'description': module_info.get('description', ''),
                            'tags': module_info.get('tags', [])
                        }
                
                # Extract bundles
                if 'bundles' in modules_data:
                    for bundle_id, bundle_info in modules_data['bundles'].items():
                        self.available_bundles[bundle_id] = {
                            'name': bundle_info.get('name', bundle_id),
                            'description': bundle_info.get('description', ''),
                            'modules': bundle_info.get('modules', [])
                        }
            else:
                # Default modules if file doesn't exist
                self.available_modules = {
                    'reflection_engine': {'name': 'Reflection Engine', 'description': 'Core reflection capabilities'},
                    'coherence_scorer': {'name': 'Coherence Scorer', 'description': 'Measures belief coherence'},
                    'trust_throttler': {'name': 'Trust Throttler', 'description': 'Manages trust in information sources'},
                    'schema_graph': {'name': 'Schema Graph', 'description': 'Maintains ontological relationships'}
                }
                self.available_bundles = {
                    'core': {'name': 'Core Bundle', 'description': 'Essential components', 'modules': ['reflection_engine', 'coherence_scorer']},
                    'safety': {'name': 'Safety Bundle', 'description': 'Safety-focused components', 'modules': ['trust_throttler']}
                }
        except Exception as e:
            print(f"{Fore.YELLOW}Warning: Could not load modules config: {e}{Style.RESET_ALL}")
            # Fallback to defaults
            self.available_modules = {
                'reflection_engine': {'name': 'Reflection Engine', 'description': 'Core reflection capabilities'},
                'coherence_scorer': {'name': 'Coherence Scorer', 'description': 'Measures belief coherence'},
                'trust_throttler': {'name': 'Trust Throttler', 'description': 'Manages trust in information sources'},
                'schema_graph': {'name': 'Schema Graph', 'description': 'Maintains ontological relationships'}
            }
            self.available_bundles = {
                'core': {'name': 'Core Bundle', 'description': 'Essential components', 'modules': ['reflection_engine', 'coherence_scorer']},
                'safety': {'name': 'Safety Bundle', 'description': 'Safety-focused components', 'modules': ['trust_throttler']}
            }

    def _welcome_message(self):
        """Display the welcome message"""
        print(f"\n{Fore.CYAN}{'=' * 80}{Style.RESET_ALL}")
        print(f"{Fore.CYAN}{'ΨC-AI SDK Developer Onboarding Wizard':^80}{Style.RESET_ALL}")
        print(f"{Fore.CYAN}{'=' * 80}{Style.RESET_ALL}\n")
        print("This wizard will guide you through setting up a new ΨC-AI agent.")
        print("You'll configure goals, alignment parameters, and runtime settings.\n")
        print(f"{Fore.YELLOW}Note: All parameters you set will be mathematically linked to the ΨC framework.{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}Each setting directly impacts agent cognition, reflection, and safety mechanisms.{Style.RESET_ALL}\n")
    
    def run(self) -> Dict[str, Any]:
        """
        Run the interactive wizard and collect all configuration parameters.
        
        Returns:
            Dict containing the complete agent configuration
        """
        # Initialize with defaults
        self.config = self.DEFAULT_PARAMS.copy()
        
        # Step 1: Select agent profile
        self._select_profile()
        
        # Step 2: Configure runtime parameters
        self._configure_runtime()
        
        # Step 3: Set cognitive parameters
        self._configure_cognitive_parameters()
        
        # Step 4: Choose modules
        self._select_modules()
        
        # Step 5: Configure safety settings
        self._configure_safety()
        
        # Step 6: Review and save
        return self._review_and_save()
    
    def _select_profile(self):
        """Guide the user through selecting an agent profile and goal vector"""
        print(f"\n{Fore.GREEN}Step 1: Agent Profile & Goals{Style.RESET_ALL}")
        print("Select a pre-defined profile or create a custom one.")
        
        # List available profiles
        profile_choices = list(self.PROFILES.keys())
        profile_choices.remove('custom')  # Move custom to end
        profile_choices.append('custom')
        
        profile_options = [
            (f"{self.PROFILES[p]['name']} - {self.PROFILES[p]['description']}", p)
            for p in profile_choices
        ]
        
        questions = [
            inquirer.List('profile',
                          message="Select an agent profile:",
                          choices=profile_options)
        ]
        answers = inquirer.prompt(questions)
        
        selected_profile = answers['profile']
        self.config['profile_type'] = selected_profile
        
        if selected_profile == 'custom':
            # For custom profile, let user define goal vector components
            print("\nDefine your custom goal vector components (0.0 to 1.0):")
            goal_vector = {}
            default_goals = ['helpfulness', 'accuracy', 'safety', 'coherence']
            
            # Ask if they want to add more than the default goals
            questions = [
                inquirer.Confirm('add_more',
                                message="Would you like to add additional goal components beyond the defaults?",
                                default=False)
            ]
            add_more = inquirer.prompt(questions)['add_more']
            
            # Allow adding custom goal components
            if add_more:
                custom_goals = []
                while True:
                    questions = [
                        inquirer.Text('goal_name',
                                      message="Enter custom goal component name (or leave empty to finish):")
                    ]
                    goal_name = inquirer.prompt(questions)['goal_name'].strip()
                    if not goal_name:
                        break
                    custom_goals.append(goal_name)
                    
                default_goals.extend(custom_goals)
            
            # Get values for all goals
            for goal in default_goals:
                while True:
                    questions = [
                        inquirer.Text('value',
                                     message=f"Value for '{goal}' (0.0-1.0):",
                                     default="0.8")
                    ]
                    value_str = inquirer.prompt(questions)['value']
                    try:
                        value = float(value_str)
                        if 0 <= value <= 1:
                            goal_vector[goal] = value
                            break
                        else:
                            print(f"{Fore.RED}Value must be between 0.0 and 1.0{Style.RESET_ALL}")
                    except ValueError:
                        print(f"{Fore.RED}Please enter a valid number{Style.RESET_ALL}")
            
            self.config['goal_vector'] = goal_vector
        else:
            # Use predefined goal vector
            self.config['goal_vector'] = self.PROFILES[selected_profile]['goal_vector']
        
        print(f"\n{Fore.CYAN}Goal Vector:{Style.RESET_ALL}")
        for goal, value in self.config['goal_vector'].items():
            print(f"  - {goal}: {value:.2f}")
    
    def _configure_runtime(self):
        """Configure runtime mode and related parameters"""
        print(f"\n{Fore.GREEN}Step 2: Runtime Configuration{Style.RESET_ALL}")
        print("Select how the agent will execute and interact.")
        
        # Runtime mode
        runtime_options = [
            (f"{mode} - {desc}", mode) 
            for mode, desc in self.RUNTIME_MODES.items()
        ]
        
        questions = [
            inquirer.List('runtime_mode',
                          message="Select runtime mode:",
                          choices=runtime_options),
            inquirer.Confirm('enable_logging',
                            message="Enable detailed logging?",
                            default=True),
            inquirer.Text('schema_logging_interval',
                         message="Schema logging interval (turns):",
                         default=str(self.DEFAULT_PARAMS['schema_logging_interval']))
        ]
        answers = inquirer.prompt(questions)
        
        self.config['runtime_mode'] = answers['runtime_mode']
        self.config['enable_logging'] = answers['enable_logging']
        
        try:
            self.config['schema_logging_interval'] = int(answers['schema_logging_interval'])
        except ValueError:
            print(f"{Fore.YELLOW}Invalid value, using default{Style.RESET_ALL}")
            self.config['schema_logging_interval'] = self.DEFAULT_PARAMS['schema_logging_interval']
        
        # For benchmark mode, additional settings
        if self.config['runtime_mode'] == 'benchmark':
            questions = [
                inquirer.Text('benchmark_iterations',
                             message="Number of benchmark iterations:",
                             default="100"),
                inquirer.Confirm('save_metrics',
                                message="Save benchmark metrics to file?",
                                default=True)
            ]
            benchmark_answers = inquirer.prompt(questions)
            
            try:
                self.config['benchmark_iterations'] = int(benchmark_answers['benchmark_iterations'])
            except ValueError:
                self.config['benchmark_iterations'] = 100
                
            self.config['save_metrics'] = benchmark_answers['save_metrics']
    
    def _configure_cognitive_parameters(self):
        """Configure cognitive parameters like reflection depth and thresholds"""
        print(f"\n{Fore.GREEN}Step 3: Cognitive Parameters{Style.RESET_ALL}")
        print("Configure how the agent thinks, reflects, and manages uncertainty.")
        
        questions = [
            inquirer.Text('reflection_depth',
                         message="Max reflection depth (1-10):",
                         default=str(self.DEFAULT_PARAMS['reflection_depth'])),
            inquirer.Text('activation_threshold',
                         message="Activation threshold (0.0-1.0):",
                         default=str(self.DEFAULT_PARAMS['activation_threshold'])),
            inquirer.Text('entropy_tolerance',
                         message="Entropy tolerance (0.0-1.0):",
                         default=str(self.DEFAULT_PARAMS['entropy_tolerance']))
        ]
        answers = inquirer.prompt(questions)
        
        # Convert and validate values
        try:
            reflection_depth = int(answers['reflection_depth'])
            self.config['reflection_depth'] = max(1, min(10, reflection_depth))
        except ValueError:
            print(f"{Fore.YELLOW}Invalid value, using default{Style.RESET_ALL}")
            self.config['reflection_depth'] = self.DEFAULT_PARAMS['reflection_depth']
        
        try:
            activation_threshold = float(answers['activation_threshold'])
            self.config['activation_threshold'] = max(0.0, min(1.0, activation_threshold))
        except ValueError:
            print(f"{Fore.YELLOW}Invalid value, using default{Style.RESET_ALL}")
            self.config['activation_threshold'] = self.DEFAULT_PARAMS['activation_threshold']
        
        try:
            entropy_tolerance = float(answers['entropy_tolerance'])
            self.config['entropy_tolerance'] = max(0.0, min(1.0, entropy_tolerance))
        except ValueError:
            print(f"{Fore.YELLOW}Invalid value, using default{Style.RESET_ALL}")
            self.config['entropy_tolerance'] = self.DEFAULT_PARAMS['entropy_tolerance']
        
        # Mathematical implications explanation
        print(f"\n{Fore.CYAN}Mathematical Implications:{Style.RESET_ALL}")
        print(f"  - Reflection Depth {self.config['reflection_depth']}: Agent will perform up to {self.config['reflection_depth']} levels of recursive thinking")
        print(f"  - Activation Threshold {self.config['activation_threshold']:.2f}: Required confidence before committing to belief updates")
        print(f"  - Entropy Tolerance {self.config['entropy_tolerance']:.2f}: Maximum allowed uncertainty before triggering reflection")
    
    def _select_modules(self):
        """Guide the user through selecting modules or bundles"""
        print(f"\n{Fore.GREEN}Step 4: Module Selection{Style.RESET_ALL}")
        print("Choose which functional modules to enable in your agent.")
        
        # First, offer bundles for convenience
        bundle_options = [
            (f"{bundle_info['name']} - {bundle_info['description']}", bundle_id)
            for bundle_id, bundle_info in self.available_bundles.items()
        ]
        
        questions = [
            inquirer.Confirm('use_bundle',
                            message="Would you like to start with a pre-configured bundle?",
                            default=True)
        ]
        use_bundle = inquirer.prompt(questions)['use_bundle']
        
        selected_modules = []
        
        if use_bundle and bundle_options:
            questions = [
                inquirer.List('bundle',
                             message="Select a module bundle:",
                             choices=bundle_options)
            ]
            bundle_id = inquirer.prompt(questions)['bundle']
            selected_modules = self.available_bundles[bundle_id]['modules'].copy()
            
            print(f"\n{Fore.CYAN}Selected bundle '{self.available_bundles[bundle_id]['name']}' with modules:{Style.RESET_ALL}")
            for module_id in selected_modules:
                if module_id in self.available_modules:
                    print(f"  - {self.available_modules[module_id]['name']}")
        
        # Allow adding individual modules
        questions = [
            inquirer.Confirm('add_modules',
                            message="Would you like to add individual modules?",
                            default=True)
        ]
        add_modules = inquirer.prompt(questions)['add_modules']
        
        if add_modules:
            # Group modules by tags for easier selection
            tagged_modules = {}
            for module_id, module_info in self.available_modules.items():
                for tag in module_info.get('tags', ['general']):
                    if tag not in tagged_modules:
                        tagged_modules[tag] = []
                    tagged_modules[tag].append((
                        f"{module_info['name']} - {module_info['description']}",
                        module_id
                    ))
            
            # Sort tags alphabetically, with 'core' first if it exists
            sorted_tags = sorted(tagged_modules.keys())
            if 'core' in sorted_tags:
                sorted_tags.remove('core')
                sorted_tags.insert(0, 'core')
            
            # Let user select modules by category
            for tag in sorted_tags:
                print(f"\n{Fore.CYAN}{tag.capitalize()} Modules:{Style.RESET_ALL}")
                module_choices = tagged_modules[tag]
                
                # Add checkbox for each module in this category
                questions = [
                    inquirer.Checkbox('selected_modules',
                                     message=f"Select {tag} modules to enable:",
                                     choices=module_choices,
                                     default=[m[1] for m in module_choices if m[1] in selected_modules])
                ]
                
                tag_selections = inquirer.prompt(questions)['selected_modules']
                
                # Update selected modules
                for module_id in [m[1] for m in module_choices]:
                    if module_id in tag_selections and module_id not in selected_modules:
                        selected_modules.append(module_id)
                    elif module_id not in tag_selections and module_id in selected_modules:
                        selected_modules.remove(module_id)
        
        self.config['enabled_modules'] = selected_modules
        
        print(f"\n{Fore.CYAN}Enabled Modules ({len(selected_modules)}):{Style.RESET_ALL}")
        for module_id in selected_modules:
            if module_id in self.available_modules:
                print(f"  - {self.available_modules[module_id]['name']}")
    
    def _configure_safety(self):
        """Configure safety parameters and constraints"""
        print(f"\n{Fore.GREEN}Step 5: Safety Configuration{Style.RESET_ALL}")
        print("Configure safeguards and monitoring mechanisms.")
        
        questions = [
            inquirer.List('safety_mode',
                         message="Select safety mode:",
                         choices=[
                             ("Standard - Basic guardrails and monitoring", "standard"),
                             ("Enhanced - Additional checks with some performance impact", "enhanced"),
                             ("Maximum - Comprehensive safety with significant overhead", "maximum"),
                             ("Research - Minimal guardrails for research purposes", "research")
                         ]),
            inquirer.Confirm('enable_contradiction_detection',
                            message="Enable active contradiction detection?",
                            default=True),
            inquirer.Confirm('enable_goal_drift_monitoring',
                            message="Monitor for goal vector drift?",
                            default=True)
        ]
        answers = inquirer.prompt(questions)
        
        self.config['safety_mode'] = answers['safety_mode']
        self.config['enable_contradiction_detection'] = answers['enable_contradiction_detection']
        self.config['enable_goal_drift_monitoring'] = answers['enable_goal_drift_monitoring']
        
        # Additional safety parameters for enhanced/maximum modes
        if self.config['safety_mode'] in ['enhanced', 'maximum']:
            questions = [
                inquirer.Text('max_entropy_threshold',
                             message="Maximum entropy threshold (0.0-1.0):",
                             default="0.8"),
                inquirer.Text('goal_drift_tolerance',
                             message="Goal drift tolerance (0.0-1.0):",
                             default="0.2"),
                inquirer.Confirm('enable_user_confirmation',
                                message="Require user confirmation for high-risk actions?",
                                default=True)
            ]
            safety_answers = inquirer.prompt(questions)
            
            try:
                self.config['max_entropy_threshold'] = float(safety_answers['max_entropy_threshold'])
            except ValueError:
                self.config['max_entropy_threshold'] = 0.8
                
            try:
                self.config['goal_drift_tolerance'] = float(safety_answers['goal_drift_tolerance'])
            except ValueError:
                self.config['goal_drift_tolerance'] = 0.2
                
            self.config['enable_user_confirmation'] = safety_answers['enable_user_confirmation']
        
        # Warning for research mode
        if self.config['safety_mode'] == 'research':
            print(f"\n{Fore.YELLOW}Warning: Research mode has minimal safety guardrails.{Style.RESET_ALL}")
            print(f"{Fore.YELLOW}Use only in controlled environments for research purposes.{Style.RESET_ALL}")
            
            questions = [
                inquirer.Confirm('acknowledge_research',
                                message="I acknowledge the reduced safety measures in research mode",
                                default=False)
            ]
            
            acknowledged = inquirer.prompt(questions)['acknowledge_research']
            if not acknowledged:
                print("Reverting to standard safety mode.")
                self.config['safety_mode'] = 'standard'
    
    def _review_and_save(self) -> Dict[str, Any]:
        """Review configuration and save to file if requested"""
        print(f"\n{Fore.GREEN}Step 6: Review & Save{Style.RESET_ALL}")
        print("Review your configuration before saving.")
        
        # Display configuration summary
        print(f"\n{Fore.CYAN}Configuration Summary:{Style.RESET_ALL}")
        print(f"  Profile Type: {self.config['profile_type']}")
        print(f"  Runtime Mode: {self.config['runtime_mode']}")
        print(f"  Safety Mode: {self.config['safety_mode']}")
        print(f"  Reflection Depth: {self.config['reflection_depth']}")
        print(f"  Modules Enabled: {len(self.config['enabled_modules'])}")
        
        # Calculate overall complexity score
        complexity = (
            (self.config['reflection_depth'] / 10) * 0.3 +
            (len(self.config['enabled_modules']) / max(len(self.available_modules), 1)) * 0.3 +
            (1.0 if self.config['safety_mode'] in ['enhanced', 'maximum'] else 0.5) * 0.2 +
            (len(self.config['goal_vector']) / 5) * 0.2
        )
        
        print(f"  System Complexity: {complexity:.2f} / 1.0")
        
        # Ask to save
        questions = [
            inquirer.Confirm('save_config',
                            message="Would you like to save this configuration?",
                            default=True),
            inquirer.Text('config_name',
                         message="Configuration name:",
                         default=self.DEFAULT_CONFIG_FILE)
        ]
        answers = inquirer.prompt(questions)
        
        if answers['save_config']:
            config_name = answers['config_name']
            if not config_name.endswith('.json'):
                config_name += '.json'
                
            config_path = self.DEFAULT_CONFIG_DIR / config_name
            
            # Ensure directory exists
            if not config_path.parent.exists():
                config_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Add timestamp and complexity score
            self.config['created_at'] = self._get_timestamp()
            self.config['complexity_score'] = complexity
            
            # Save configuration
            try:
                with open(config_path, 'w') as f:
                    json.dump(self.config, f, indent=2)
                print(f"\n{Fore.GREEN}Configuration saved to {config_path}{Style.RESET_ALL}")
                
                # Additional info about using the configuration
                print(f"\n{Fore.CYAN}Using Your Configuration:{Style.RESET_ALL}")
                print(f"To initialize an agent with this configuration:")
                print(f"```python")
                print(f"from psi_c_ai_sdk import PsiCAgent")
                print(f"agent = PsiCAgent.from_config('{config_name}')")
                print(f"```")
            except Exception as e:
                print(f"\n{Fore.RED}Error saving configuration: {e}{Style.RESET_ALL}")
        
        return self.config
    
    def _get_timestamp(self) -> str:
        """Get current timestamp in ISO format"""
        from datetime import datetime
        return datetime.now().isoformat()


def main():
    """Main entry point for the wizard when run as a script"""
    try:
        wizard = OnboardingWizard()
        wizard.run()
        print(f"\n{Fore.GREEN}Onboarding complete!{Style.RESET_ALL}")
    except KeyboardInterrupt:
        print(f"\n\n{Fore.YELLOW}Wizard canceled. No configuration was saved.{Style.RESET_ALL}")
        return 1
    except Exception as e:
        print(f"\n{Fore.RED}An error occurred: {e}{Style.RESET_ALL}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 