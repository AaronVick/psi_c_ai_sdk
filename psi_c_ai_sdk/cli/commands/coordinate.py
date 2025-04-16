# psi_c_ai_sdk/cli/commands/coordinate.py

def run():
    from psi_c_ai_sdk.multi_agent.multi_agent_coordinator import MultiAgentCoordinator
    from examples.agents import get_active_agents  # Your simulated ΨC agents

    agents = get_active_agents()
    mac = MultiAgentCoordinator(agents)
    mac.coordinate()

    print("[ΨC] Multi-agent coordination cycle complete.")
