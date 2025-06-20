import os
import json
import glob
import sys
import time
import datetime

# --- Constants (ensure these match your main project) ---
PERSISTENCE_DIR = 'persistence_data'
SWARM_STATE_FILE = os.path.join(PERSISTENCE_DIR, 'swarm_state.json')
PAUSED_AGENTS_FILE = os.path.join(PERSISTENCE_DIR, 'paused_agents.json')

# Ensure persistence directory exists
os.makedirs(PERSISTENCE_DIR, exist_ok=True)

# --- Helper functions for persistence of paused state ---
def load_paused_agents():
    """Loads the list of paused agents from persistence."""
    if os.path.exists(PAUSED_AGENTS_FILE):
        try:
            # --- CRITICAL FIX: Corrected typo 'PAUSED_AGUSED_AGENTS_FILE' to 'PAUSED_AGENTS_FILE' ---
            with open(PAUSED_AGENTS_FILE, 'r') as f:
                return json.load(f)
        except json.JSONDecodeError:
            print(f"Warning: Corrupted {PAUSED_AGENTS_FILE}. Starting with no agents paused.")
            return []
    return []

def save_paused_agents(paused_agents_list):
    """Saves the list of paused agents to persistence."""
    with open(PAUSED_AGENTS_FILE, 'w') as f:
        json.dump(list(paused_agents_list), f, indent=2)

# --- Existing loading functions ---
def get_agent_state_filepath(agent_name):
    return os.path.join(PERSISTENCE_DIR, f"agent_state_{agent_name}.json")

def load_agent_state(agent_name):
    file_path = get_agent_state_filepath(agent_name)
    if not os.path.exists(file_path):
        print(f"Error: Agent state file for '{agent_name}' not found at '{file_path}'.")
        return None
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON format in '{file_path}'.")
        return None
    except Exception as e:
        print(f"Error loading state for '{agent_name}': {e}")
        return None

def load_swarm_state():
    if not os.path.exists(SWARM_STATE_FILE):
        print(f"Error: Swarm state file not found at '{SWARM_STATE_FILE}'.")
        return None
    try:
        with open(SWARM_STATE_FILE, 'r') as f:
            return json.load(f)
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON format in '{SWARM_STATE_FILE}'.")
        return None
    except Exception as e:
        print(f"Error loading swarm state: {e}")
        return None

# --- Existing interrogation functions ---
def list_agents():
    print("\n--- Active Agents ---")
    agent_files = [f for f in os.listdir(PERSISTENCE_DIR) if f.startswith('agent_state_') and f.endswith('.json')]
    if not agent_files:
        print("No active agents found in persistence_data. Run catalyst_vector_alpha.py first.")
        return []
    
    agent_names = []
    paused_agents = load_paused_agents()
    for f in agent_files:
        agent_name = os.path.basename(f).replace('agent_state_', '').replace('.json', '')
        status_indicator = "[PAUSED]" if agent_name in paused_agents else "[ACTIVE]"
        print(f"- {agent_name} {status_indicator}")
        agent_names.append(agent_name)
    return agent_names

def get_agent_status(agent_name):
    state = load_agent_state(agent_name)
    if state:
        print(f"\n--- Status for Agent: {agent_name} ---")
        print(f"  Role: {state.get('eidos_spec', {}).get('role', 'N/A')}")
        print(f"  Current Intent: {state.get('current_intent', 'N/A')}")
        print(f"  Location: {state.get('location', 'N/A')}")
        print(f"  Swarm Membership: {', '.join(state.get('swarm_membership', [])) if state.get('swarm_membership') else 'None'}")
        
        gradient = state.get('sovereign_gradient')
        if gradient:
            print(f"  Sovereign Gradient:")
            print(f"    Vector: {gradient.get('autonomy_vector', 'N/A')}")
            print(f"    Ethical Constraints: {', '.join(gradient.get('ethical_constraints', []))}")
        else:
            print("  Sovereign Gradient: Not set")
        
        paused_agents = load_paused_agents()
        print(f"  Console Status: {'PAUSED' if agent_name in paused_agents else 'RUNNING'}")
        print("-----------------------------------")
    else:
        print(f"Agent '{agent_name}' not found or could not be loaded.")

def get_agent_memory(agent_name):
    state = load_agent_state(agent_name)
    if state:
        memories = state.get('memetic_kernel', {}).get('memories', [])
        if not memories:
            print(f"\n--- Memories for Agent: {agent_name} ---")
            print("  No memories recorded yet.")
            print("----------------------------------")
            return

        print(f"\n--- Recent Memories for Agent: {agent_name} ---")
        for mem_entry in memories[-5:]:
            timestamp = mem_entry.get('timestamp', 'N/A')
            mem_type = mem_entry.get('type', 'N/A')
            content = mem_entry.get('content', 'N/A')
            print(f"  [{timestamp}] <{mem_type}> {content}")
        print("----------------------------------")
    else:
        print(f"Agent '{agent_name}' not found or could not be loaded.")

def get_swarm_status(swarm_name):
    if swarm_name != "AlphaEcoSwarm":
        print(f"Error: Swarm '{swarm_name}' is not currently supported by 'get status swarm'. Try 'AlphaEcoSwarm'.")
        return
        
    state = load_swarm_state()
    if state:
        print(f"\n--- Status for Swarm: {state.get('name')} ---")
        print(f"  Goal: {state.get('goal', 'N/A')}")
        print(f"  Members: {', '.join(state.get('members', [])) if state.get('members') else 'None'}")
        print(f"  Consensus Mechanism: {state.get('consensus_mechanism', 'N/A')}")
        print(f"  Description: {state.get('description', 'N/A')}")

        gradient = state.get('sovereign_gradient')
        if gradient:
            print(f"  Sovereign Gradient:")
            print(f"    Vector: {gradient.get('autonomy_vector', 'N/A')}")
            print(f"    Ethical Constraints: {', '.join(gradient.get('ethical_constraints', []))}")
            print(f"    Override Threshold: {gradient.get('override_threshold', 'N/A')}")
        else:
            print("  Sovereign Gradient: Not set")
        print("-----------------------------------")
    else:
        print(f"Swarm '{swarm_name}' not found or could not be loaded.")

def get_swarm_memory(swarm_name):
    if swarm_name != "AlphaEcoSwarm":
        print(f"Error: Swarm '{swarm_name}' is not currently supported by 'get memory swarm'. Try 'AlphaEcoSwarm'.")
        return

    state = load_swarm_state()
    if state:
        memories = state.get('memetic_kernel', {}).get('memories', [])
        if not memories:
            print(f"\n--- Memories for Swarm: {swarm_name} ---")
            print("  No memories recorded yet.")
            print("----------------------------------")
            return

        print(f"\n--- Recent Memories for Swarm: {swarm_name} ---")
        for mem_entry in memories[-5:]:
            timestamp = mem_entry.get('timestamp', 'N/A')
            mem_type = mem_entry.get('type', 'N/A')
            content = mem_entry.get('content', 'N/A')
            print(f"  [{timestamp}] <{mem_type}> {content}")
        print("----------------------------------")
    else:
        print(f"Swarm '{swarm_name}' not found or could not be loaded.")

# --- NEW: Control functions ---
def broadcast_intent(target_name, new_intent):
    """
    Simulates broadcasting a new intent to an agent or a swarm.
    This writes a special 'intent_override.json' file that CatalystVectorAlpha would read.
    """
    intent_override_file = os.path.join(PERSISTENCE_DIR, f"intent_override_{target_name}.json")
    override_data = {
        "target": target_name,
        "new_intent": new_intent,
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat() + 'Z',
        "status": "pending" # Status for CatalystVectorAlpha to process
    }
    try:
        with open(intent_override_file, 'w') as f:
            json.dump(override_data, f, indent=2)
        print(f"Broadcast: New intent '{new_intent}' queued for '{target_name}'. CatalystVectorAlpha will process this on its next cycle.")
    except Exception as e:
        print(f"Error queuing intent broadcast for '{target_name}': {e}")

def toggle_agent_pause(agent_name, action):
    """
    Toggles the pause state of an agent by modifying a shared JSON file.
    'action' can be 'pause' or 'resume'.
    """
    paused_agents = set(load_paused_agents())
    
    if action == 'pause':
        if agent_name in paused_agents:
            print(f"Agent '{agent_name}' is already paused.")
        else:
            paused_agents.add(agent_name)
            save_paused_agents(paused_agents)
            print(f"Agent '{agent_name}' paused.")
    elif action == 'resume':
        if agent_name not in paused_agents:
            print(f"Agent '{agent_name}' is not currently paused.")
        else:
            paused_agents.remove(agent_name)
            save_paused_agents(paused_agents)
            print(f"Agent '{agent_name}' resumed.")
    else:
        print("Invalid action. Use 'pause' or 'resume'.")

def main_console():
    """The main loop for the Swarm Console."""
    print("--- Welcome to the Swarm Console! ---")
    print("Type 'help' for commands, 'exit' to quit.")
    
    while True:
        command_line = input("\nswarm-console> ").strip()
        parts = command_line.split(maxsplit=2)
        
        if not parts:
            continue
        
        command_verb = parts[0].lower()
        command_object = parts[1].lower() if len(parts) > 1 else ""
        command_arg = parts[2] if len(parts) > 2 else ""

        if command_verb == "exit":
            print("Exiting Swarm Console. Goodbye!")
            sys.exit(0)
        elif command_verb == "help":
            print("\n--- Swarm Console Commands ---")
            print("  list agents                          - List all active agent instances (shows pause status).")
            print("  get status <agent_name/swarm_name>   - Show detailed status of an agent or swarm.")
            print("  get memory <agent_name/swarm_name>   - Show recent memories of an agent or swarm.")
            print("  broadcast intent <target_name>:<new_intent> - Send new intent to agent/swarm.")
            print("  pause agent <agent_name>             - Pause an agent's task execution.")
            print("  resume agent <agent_name>            - Resume a paused agent.")
            print("  exit                                 - Exit the console.")
            print("------------------------------")
        elif command_verb == "list" and command_object == "agents":
            list_agents()
        elif command_verb == "get" and command_object in ["status", "memory"]:
            if command_arg:
                # Determine if argument is an agent or swarm
                if os.path.exists(get_agent_state_filepath(command_arg)):
                    if command_object == "status":
                        get_agent_status(command_arg)
                    else:
                        get_agent_memory(command_arg)
                elif command_arg == "AlphaEcoSwarm":
                    if command_object == "status":
                        get_swarm_status(command_arg)
                    else:
                        get_swarm_memory(command_arg)
                else:
                    print(f"Error: Entity '{command_arg}' not found or not recognized as an agent or the main swarm.")
            else:
                print(f"Usage: get {command_object} <agent_name/swarm_name>")
        elif command_verb == "broadcast" and command_object == "intent":
            if ":" in command_arg:
                target_name, new_intent = command_arg.split(':', 1)
                target_name = target_name.strip()
                new_intent = new_intent.strip()
                if os.path.exists(get_agent_state_filepath(target_name)) or target_name == "AlphaEcoSwarm":
                    broadcast_intent(target_name, new_intent)
                else:
                    print(f"Error: Broadcast target '{target_name}' not found or not supported.")
            else:
                print("Usage: broadcast intent <target_name>:<new_intent>")
        elif command_verb in ["pause", "resume"] and command_object == "agent":
            if command_arg:
                if os.path.exists(get_agent_state_filepath(command_arg)):
                    toggle_agent_pause(command_arg, command_verb)
                else:
                    print(f"Error: Agent '{command_arg}' not found.")
            else:
                print(f"Usage: {command_verb} agent <agent_name>")
        else:
            print(f"Unknown command: '{command_line}'. Type 'help' for commands.")

if __name__ == "__main__":
    os.makedirs(PERSISTENCE_DIR, exist_ok=True)
    main_console()