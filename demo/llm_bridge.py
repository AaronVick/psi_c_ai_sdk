#!/usr/bin/env python3
"""
ΨC Schema Integration Demo - LLM Bridge
---------------------------------------
Optional component that provides integration with language models
for enhancing the demo experience. This is designed to be fully
optional, with the demo functioning perfectly without LLM access.
"""

import os
import json
import logging
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("llm_bridge")

# Default prompt templates
DEFAULT_REFLECTION_PROMPT = """
You are an expert cognitive scientist tasked with analyzing how a memory affects a belief system.

Given the following new memory:
"{memory_content}"

And considering these relevant existing memories:
{relevant_memories}

Please provide:
1. A concise analysis of how this memory relates to existing knowledge
2. Any potential contradictions or inconsistencies
3. How this memory might evolve the agent's understanding

Keep your response concise and focused on cognitive implications.
"""

DEFAULT_SUMMARY_PROMPT = """
You are an expert in epistemology tasked with explaining coherence changes in a belief system.

The system just processed a new memory:
"{memory_content}"

This resulted in the following changes:
- Coherence changed from {old_coherence:.4f} to {new_coherence:.4f}
- Entropy changed from {old_entropy:.4f} to {new_entropy:.4f}
- Contradictions detected: {contradiction_count}
- Schema updated: {schema_updated}
{phase_transition_text}

Please provide a brief, insightful explanation of what these changes mean in terms of the system's cognitive development and belief coherence. Explain in approximately 2-3 sentences using accessible but precise language.
"""

@dataclass
class LLMConfig:
    """Configuration for LLM bridge."""
    api_key: Optional[str] = None
    api_base: Optional[str] = None
    model: str = "gpt-3.5-turbo"
    max_tokens: int = 150
    temperature: float = 0.7
    reflection_prompt: str = DEFAULT_REFLECTION_PROMPT
    summary_prompt: str = DEFAULT_SUMMARY_PROMPT
    is_enabled: bool = False


class LLMBridge:
    """
    Bridge between the ΨC demo and language models.
    
    This class handles optional LLM interactions to enhance the demo
    with natural language processing capabilities when available.
    """
    
    def __init__(self, config: Optional[LLMConfig] = None):
        """
        Initialize the LLM bridge.
        
        Args:
            config: Optional configuration object
        """
        self.config = config or LLMBridge._load_config()
        self._check_environment_variables()
        
        # Try to import OpenAI if enabled
        self.openai = None
        if self.config.is_enabled and self.config.api_key:
            try:
                import openai
                self.openai = openai
                openai.api_key = self.config.api_key
                if self.config.api_base:
                    openai.api_base = self.config.api_base
                logger.info(f"OpenAI initialized with model: {self.config.model}")
            except ImportError:
                logger.warning("OpenAI package not found. Running in local-only mode.")
                self.config.is_enabled = False
    
    @staticmethod
    def _load_config() -> LLMConfig:
        """Load configuration from file or use defaults."""
        config_path = os.path.join(os.path.dirname(__file__), "demo_config", "llm_config.json")
        
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    config_dict = json.load(f)
                return LLMConfig(**config_dict)
            except Exception as e:
                logger.error(f"Error loading LLM config: {e}")
                return LLMConfig()
        else:
            # Create default config
            config = LLMConfig()
            try:
                os.makedirs(os.path.dirname(config_path), exist_ok=True)
                with open(config_path, 'w') as f:
                    json.dump({
                        "api_key": None,
                        "api_base": None,
                        "model": config.model,
                        "max_tokens": config.max_tokens,
                        "temperature": config.temperature,
                        "reflection_prompt": config.reflection_prompt,
                        "summary_prompt": config.summary_prompt,
                        "is_enabled": config.is_enabled
                    }, f, indent=2)
            except Exception as e:
                logger.error(f"Error creating default LLM config: {e}")
            
            return config
    
    def _check_environment_variables(self):
        """Check and use environment variables if available."""
        if not self.config.api_key and "OPENAI_API_KEY" in os.environ:
            self.config.api_key = os.environ["OPENAI_API_KEY"]
            self.config.is_enabled = True
            logger.info("Using OpenAI API key from environment variable")
        
        if not self.config.api_base and "OPENAI_API_BASE" in os.environ:
            self.config.api_base = os.environ["OPENAI_API_BASE"]
            logger.info("Using OpenAI API base from environment variable")
    
    def is_enabled(self) -> bool:
        """
        Check if LLM functionality is enabled.
        
        Returns:
            True if LLM functionality is enabled, False otherwise
        """
        return self.config.is_enabled and self.openai is not None
    
    def enhance_reflection(self, memory_content: str, relevant_memories: List[str]) -> str:
        """
        Enhance reflection output with LLM-generated insights.
        
        Args:
            memory_content: Content of the new memory
            relevant_memories: List of relevant existing memories
        
        Returns:
            Enhanced reflection text
        """
        if not self.is_enabled():
            return self._fallback_reflection(memory_content, relevant_memories)
        
        try:
            # Format relevant memories as a bulleted list
            memories_formatted = "\n".join([f"- {mem}" for mem in relevant_memories])
            
            # Format the prompt
            prompt = self.config.reflection_prompt.format(
                memory_content=memory_content,
                relevant_memories=memories_formatted
            )
            
            # Call the API
            response = self.openai.ChatCompletion.create(
                model=self.config.model,
                messages=[
                    {"role": "system", "content": "You are a cognitive science assistant."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature
            )
            
            return response.choices[0].message.content.strip()
        
        except Exception as e:
            logger.error(f"Error calling LLM API: {e}")
            return self._fallback_reflection(memory_content, relevant_memories)
    
    def generate_change_summary(self, 
                               memory_content: str, 
                               old_coherence: float, 
                               new_coherence: float,
                               old_entropy: float,
                               new_entropy: float,
                               contradiction_count: int,
                               schema_updated: bool,
                               phase_transition: bool) -> str:
        """
        Generate a natural language summary of system changes.
        
        Args:
            memory_content: Content of the processed memory
            old_coherence: Previous coherence value
            new_coherence: New coherence value
            old_entropy: Previous entropy value
            new_entropy: New entropy value
            contradiction_count: Number of contradictions detected
            schema_updated: Whether the schema was updated
            phase_transition: Whether a phase transition was detected
        
        Returns:
            Natural language summary of changes
        """
        if not self.is_enabled():
            return self._fallback_change_summary(
                memory_content, old_coherence, new_coherence,
                old_entropy, new_entropy, contradiction_count,
                schema_updated, phase_transition
            )
        
        try:
            phase_transition_text = "- **Phase transition detected!** The system underwent a significant shift in its cognitive structure." if phase_transition else ""
            
            # Format the prompt
            prompt = self.config.summary_prompt.format(
                memory_content=memory_content,
                old_coherence=old_coherence,
                new_coherence=new_coherence,
                old_entropy=old_entropy,
                new_entropy=new_entropy,
                contradiction_count=contradiction_count,
                schema_updated=schema_updated,
                phase_transition_text=phase_transition_text
            )
            
            # Call the API
            response = self.openai.ChatCompletion.create(
                model=self.config.model,
                messages=[
                    {"role": "system", "content": "You are a cognitive science assistant."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature
            )
            
            return response.choices[0].message.content.strip()
        
        except Exception as e:
            logger.error(f"Error calling LLM API: {e}")
            return self._fallback_change_summary(
                memory_content, old_coherence, new_coherence,
                old_entropy, new_entropy, contradiction_count,
                schema_updated, phase_transition
            )
    
    def process_user_query(self, query: str, context: Dict[str, Any]) -> str:
        """
        Process a direct user query using contextual information.
        
        This allows users to ask the system questions during the demo.
        
        Args:
            query: User's question
            context: Contextual information about the current system state
        
        Returns:
            Response to the user's query
        """
        if not self.is_enabled():
            return "LLM integration is not enabled. Please enable it to use this feature."
        
        try:
            # Create a context summary
            context_summary = (
                f"Current system state:\n"
                f"- Coherence: {context.get('coherence', 'N/A')}\n"
                f"- Entropy: {context.get('entropy', 'N/A')}\n"
                f"- Memory count: {context.get('memory_count', 'N/A')}\n"
                f"- Schema node count: {context.get('node_count', 'N/A')}\n"
                f"- Recent memories: {', '.join(context.get('recent_memories', ['None']))}"
            )
            
            # Call the API
            response = self.openai.ChatCompletion.create(
                model=self.config.model,
                messages=[
                    {
                        "role": "system", 
                        "content": "You are an assistant for a cognitive AI system based on the ΨC architecture. "
                                   "Answer the user's questions about the system's current state and functioning. "
                                   "Be accurate, concise, and focus on cognitive science aspects."
                    },
                    {"role": "user", "content": f"System context:\n{context_summary}\n\nUser question: {query}"}
                ],
                max_tokens=200,
                temperature=0.7
            )
            
            return response.choices[0].message.content.strip()
        
        except Exception as e:
            logger.error(f"Error calling LLM API: {e}")
            return f"Sorry, I couldn't process your question due to an error: {str(e)}"
    
    def _fallback_reflection(self, memory_content: str, relevant_memories: List[str]) -> str:
        """Fallback function when LLM is unavailable."""
        if not relevant_memories:
            return f"Memory \"{memory_content}\" has been processed. No relevant existing memories found."
        
        return (
            f"Memory \"{memory_content}\" has been processed. Found {len(relevant_memories)} related memories. "
            f"The system is analyzing potential connections and implications."
        )
    
    def _fallback_change_summary(self, 
                                memory_content: str, 
                                old_coherence: float, 
                                new_coherence: float,
                                old_entropy: float,
                                new_entropy: float,
                                contradiction_count: int,
                                schema_updated: bool,
                                phase_transition: bool) -> str:
        """Fallback function for change summary when LLM is unavailable."""
        
        if contradiction_count > 0 and schema_updated:
            return (
                f"Processing \"{memory_content}\" revealed {contradiction_count} contradictions, "
                f"triggering a schema update. Coherence {'increased' if new_coherence > old_coherence else 'decreased'} "
                f"from {old_coherence:.4f} to {new_coherence:.4f}."
            )
        elif phase_transition:
            return (
                f"A significant phase transition was detected while processing \"{memory_content}\". "
                f"The system's cognitive structure has undergone a fundamental reorganization."
            )
        else:
            return (
                f"Memory \"{memory_content}\" processed. Coherence is now {new_coherence:.4f} "
                f"({'↑' if new_coherence > old_coherence else '↓'})."
            )
    
    def save_config(self):
        """Save the current configuration to file."""
        config_path = os.path.join(os.path.dirname(__file__), "demo_config", "llm_config.json")
        
        try:
            os.makedirs(os.path.dirname(config_path), exist_ok=True)
            with open(config_path, 'w') as f:
                json.dump({
                    "api_key": self.config.api_key,
                    "api_base": self.config.api_base,
                    "model": self.config.model,
                    "max_tokens": self.config.max_tokens,
                    "temperature": self.config.temperature,
                    "reflection_prompt": self.config.reflection_prompt,
                    "summary_prompt": self.config.summary_prompt,
                    "is_enabled": self.config.is_enabled
                }, f, indent=2)
            
            logger.info("LLM configuration saved")
            return True
        except Exception as e:
            logger.error(f"Error saving LLM config: {e}")
            return False


if __name__ == "__main__":
    # Simple test code
    bridge = LLMBridge()
    
    print(f"LLM enabled: {bridge.is_enabled()}")
    
    reflection = bridge.enhance_reflection(
        "The sky is blue because of Rayleigh scattering.",
        ["The sky often appears blue on clear days.", "Light is composed of different wavelengths."]
    )
    print("\nReflection result:")
    print(reflection)
    
    summary = bridge.generate_change_summary(
        "The sky is blue because of Rayleigh scattering.",
        0.75, 0.82, 0.3, 0.25, 0, True, False
    )
    print("\nChange summary:")
    print(summary) 