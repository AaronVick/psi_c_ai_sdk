# ΨC Schema Integration Demo

This is an interactive demo of the ΨC-AI SDK that showcases the cognitive coherence and memory-reflection capabilities of the ΨC architecture in real-time.

## Features

- **Interactive Memory Input**: Add new memories and watch the system process and integrate them.
- **Real-time Reflection**: Visualize the system's reflection process and contradiction detection.
- **Schema Visualization**: See how the knowledge graph evolves as new information is added.
- **Coherence Metrics**: Track coherence and entropy evolution over time.
- **Domain Profiles**: Choose between different domain-specific configurations.
- **Session History Management**: Save, load, and compare agent states across different sessions.
- **Profile Isolation**: Switch between profiles while preserving history for each profile.
- **Empirical Data Collection**: Gather and analyze detailed metrics for scientific research.
- **Optional LLM Integration**: Enhanced reflection and explanation using LLMs (if enabled).
- **Local-only Operation**: No external dependencies or databases required.

## Requirements

- Python 3.8 or higher
- Streamlit
- NetworkX
- Matplotlib
- Pandas
- (Optional) OpenAI Python package for LLM integration

You can install the required packages with:

```bash
pip install streamlit networkx matplotlib pandas
# Optional for LLM integration
pip install openai
```

## Running the Demo

### Simplified Launch

To run the demo with a single command, use the provided run script:

```bash
# From the project root directory
python demo/run_demo.py
```

This will automatically set up the environment and launch the Streamlit interface.

### Manual Launch

Alternatively, you can run the Streamlit server directly:

```bash
# From the project root directory
streamlit run demo/web_interface_demo.py
```

This will start the Streamlit server and open the demo in your default web browser.

## Demo Profiles

The demo includes several domain-specific profiles:

- **Default**: General purpose configuration with balanced parameters.
- **Healthcare**: Optimized for medical knowledge with higher precision requirements.
- **Legal**: Tuned for legal reasoning with emphasis on precedent and authority.
- **Narrative**: Configured for narrative processing with flexibility for subjective beliefs.

Each profile maintains its own separate session history, allowing you to switch between profiles without losing your work.

## Session Management

The demo now includes comprehensive session management features:

- **Auto-save**: Sessions are automatically saved after each interaction.
- **Manual Save**: You can manually save the current state at any time.
- **Load Previous Sessions**: Access previously saved sessions for any profile.
- **Profile Switching**: Change profiles while preserving history for each.
- **Session Reset**: Reset the system while archiving the current state.

All session data is stored locally in the `demo_data/history` directory, organized by profile.

## Empirical Data Collection

The demo now serves as a scientific data collection tool with enhanced features:

- **Detailed Metrics**: Track processing time, coherence changes, and schema evolution.
- **State Transitions**: Visualize before/after states for each interaction.
- **Performance Analysis**: Monitor system performance over time.
- **Data Export**: Export comprehensive session data in JSON or Markdown format.
- **Error Logging**: Capture and analyze errors for system improvement.

These features support empirical research on the ΨC cognitive architecture while maintaining full isolation from the SDK core.

## LLM Integration

The demo can optionally use Language Models to enhance the reflection process and provide more detailed explanations. To enable this:

1. Check the "Enable LLM Integration" option in the sidebar.
2. Set your OpenAI API key as an environment variable:
   ```bash
   export OPENAI_API_KEY=your-api-key
   ```
   
Note that the demo works perfectly fine without LLM integration, using built-in fallback functions.

## Customization

Edit configuration files in the `demo/demo_config/` directory to adjust parameters like:

- Alpha (reflection weight)
- Beta (information relevance weight)
- Theta (minimum coherence pressure threshold)
- Epsilon (identity continuity threshold)
- Contradiction threshold
- Reflection depth

## Sample Interactions

When interacting with the ΨC demo, you're essentially communicating with a cognitive system that builds and maintains a coherent understanding of the world. Here are some effective ways to interact with the system:

### Types of Effective Inputs

- **Personal Information**: "My name is Alex and I live in Seattle. I work as a software engineer and enjoy hiking on weekends."
  
- **Preference Statements**: "I prefer tea over coffee in the mornings. On weekends, I sometimes drink coffee."
  
- **Relationships**: "Sarah is my sister. She lives in Boston and works as a doctor."
  
- **Evolving Facts**: "I'm learning to play the guitar. I've been taking lessons for three months now. I can play four chords confidently."
  
- **Beliefs and Opinions**: "I believe climate change is an urgent issue. We should invest more in renewable energy."

### Interesting Interaction Patterns

1. **Introduce Mild Contradictions**: See how the system resolves them
   - "I love spicy food" → Later: "I can only handle mild spices in my food"
   - "I've lived in Chicago for 5 years" → Later: "I moved to Chicago 3 years ago"

2. **Build on Previous Information**: Add details to existing knowledge
   - "I have a dog named Max" → "Max is a golden retriever" → "Max is 3 years old"
   
3. **Ask the System to Reflect**: After adding several pieces of information, ask:
   - "What do you know about me so far?"
   - "What are my interests based on what I've told you?"
   
4. **Domain Exploration**: Test knowledge in specific domains (especially when using different profiles)
   - Healthcare profile: "I've been experiencing headaches and fatigue lately"
   - Legal profile: "I'm wondering about the requirements for forming an LLC"
   
5. **Abstract Concepts**: Discuss ideas and see how they're integrated
   - "Democracy relies on informed citizens and fair voting systems"
   - "Artificial intelligence systems should be designed with ethical considerations"

### Topics That Showcase Learning

The system is particularly interesting when discussing:

- **Personal narratives**: Life events, career paths, education
- **Evolving situations**: Projects you're working on, skills you're developing
- **Complex relationships**: Family dynamics, professional networks
- **Changing opinions**: How your views on topics have evolved
- **Decision-making processes**: How you make choices in different contexts

By observing the coherence metrics and schema visualization while interacting, you can see how the system works to maintain a consistent understanding while incorporating new information.

## Troubleshooting

- **ImportError**: Ensure you have installed all required packages.
- **LLM not working**: Check that your API key is set correctly and the OpenAI package is installed.
- **Graph visualization issues**: Adjust screen size or toggle the graph view off if needed.
- **Performance issues**: Try resetting the system if it becomes too large or sluggish.
- **Session loading errors**: If a session fails to load, try selecting a different one or reset the system.

## Architecture

This demo maintains strict isolation from the SDK core components:

- `demo_runner.py`: Interfaces safely with the SDK without modifying core functionality
- `web_interface_demo.py`: Streamlit UI with no direct SDK dependencies
- `llm_bridge.py`: Optional LLM integration that falls back to built-in functions
- `run_demo.py`: Simplified launcher script
- `demo_config/`: Configuration files for different profiles
- `demo_data/history/`: Persistent storage of schema and memory states by profile 