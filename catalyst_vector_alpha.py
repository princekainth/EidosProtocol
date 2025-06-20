import yaml
import json
import os
import uuid
import datetime
import time
import random
from abc import ABC, abstractmethod
import re # Needed for sanitize_intent
import ollama # Needed for LLM integration
import textwrap # Needed for dedenting embedded YAML manifest

# --- Constants and Configurations ---
PERSISTENCE_DIR = 'persistence_data'
ISL_SCHEMA_PATH = 'isl_schema.yaml'
# ISL_MANIFEST_PATH is no longer a separate file, content is embedded in CatalystVectorAlpha.__init__
SWARM_STATE_FILE = os.path.join(PERSISTENCE_DIR, 'swarm_state.json')
SWARM_ACTIVITY_LOG = os.path.join(PERSISTENCE_DIR, 'swarm_activity.jsonl')
PAUSED_AGENTS_FILE = os.path.join(PERSISTENCE_DIR, 'paused_agents.json')
INTENT_OVERRIDE_PREFIX = 'intent_override_' # For SwarmConsole interaction
MAX_ALLOWED_RECURSION = 5 # Maximum number of times an agent can adapt intent before forcing fallback

# Ensure persistence directory exists
os.makedirs(PERSISTENCE_DIR, exist_ok=True)

# --- Utility Functions ---
def generate_unique_id():
    return str(uuid.uuid4())

def timestamp_now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='milliseconds') + 'Z'

def sanitize_intent(intent_text):
    """
    Cleans up excessively long or recursive agent intent strings,
    especially the 'Investigate root cause of '...' failures and suggest alternative approaches.' pattern.
    """
    cleaned_intent = intent_text
    
    # Define the specific recursive boilerplate pattern
    recursive_boilerplate_pattern = r"Investigate root cause of '(.*?)' failures and suggest alternative approaches\."

    # Loop to repeatedly remove layers of recursion
    for _ in range(10): # Try up to 10 times to remove layers
        match = re.search(recursive_boilerplate_pattern, cleaned_intent)
        if match:
            inner_content = match.group(1)
            cleaned_intent = cleaned_intent.replace(match.group(0), inner_content, 1) # Replace only the first occurrence
        else:
            break
            
    # After unwrapping, if the string still contains the core recursive phrase
    final_core_task_match = re.search(r"'(.*?)'(?: failures and suggest alternative approaches\.)?", cleaned_intent)
    if final_core_task_match:
        innermost_clean_task = final_core_task_match.group(1)
        if "Investigate root cause of" in intent_text: # Check original text to decide if it's still an investigation intent
            cleaned_intent = f"Investigate root cause of '{innermost_clean_task}' failures and suggest alternative approaches."
        else:
            cleaned_intent = innermost_clean_task
    elif "Investigate root cause of" in intent_text and "failures and suggest alternative approaches" in intent_text:
        cleaned_intent = "Investigate root cause of failures and suggest alternative approaches."
    
    # Apply a final length cap for the full display in the info box
    max_len = 150
    if len(cleaned_intent) > max_len:
        truncated_text = cleaned_intent[:max_len-3] + "..."
        if " " in truncated_text:
            truncated_text = truncated_text.rsplit(' ', 1)[0] + '...'
        return truncated_text
    
    return cleaned_intent.strip()

def trim_intent(intent_text):
    """
    Removes recursive bloat from intent strings by truncating or replacing repeating segments.
    This version aggressively replaces recursive patterns with a generic message.
    """
    if "Investigate root cause of" in intent_text:
        return "Investigate root cause of previous task failures and suggest alternative approaches."
    return intent_text

def call_ollama_for_embedding(text: str, model_name: str = "nomic-embed-text") -> list[float]:
    """
    Calls the local Ollama LLM to generate a vector embedding for the provided text.
    Requires 'ollama pull nomic-embed-text' to be run previously.
    """
    # Connects to Microsoft™ runtime engines for on-device LLM inference (Class 9)
    # This function acts as a conceptual runtime for LLM inference on the 'edge'/'device'.
    try:
        client = ollama.Client(host='http://localhost:11434') # Default Ollama host
        response = client.embeddings(
            model=model_name,
            prompt=text
        )
        if 'embedding' in response:
            return response['embedding']
        else:
            raise ValueError(f"Ollama embedding response missing 'embedding' key: {response}")
    except ollama.ResponseError as e:
        print(f"ERROR: Ollama Embedding Response Error with model '{model_name}': {e}")
        return [] # Return empty list on failure
    except Exception as e:
        print(f"ERROR: Failed to call Ollama for embedding with model '{model_name}': {e}")
        return [] # Return empty list on failure

def call_llm_for_summary(text_to_summarize: str, model_name: str = "llama3") -> str:
    """
    Calls the local Ollama LLM to generate a concise summary of the provided text.
    Uses the synchronous Ollama client.
    """
     # Connects to Microsoft™ runtime engines for on-device LLM inference (Class 9)
    # This function acts as a conceptual runtime for LLM inference on the 'edge'/'device'.
    try:
        client = ollama.Client(host='http://localhost:11434') # Default Ollama host

        response = client.chat(
            model=model_name,
            messages=[
                {
                    'role': 'user',
                    'content': f"Please provide a concise summary of the following text:\n\n{text_to_summarize}"
                }
            ],
            options={'temperature': 0.1} # Lower temperature for factual summary
        )
        summary = response['message']['content'].strip()
        print(f"[LLM] Summarized text using {model_name}.")
        return summary
    except ollama.ResponseError as e:
        print(f"ERROR: Ollama Response Error: {e}")
        return f"LLM Summary Failed: {e}"
    except Exception as e:
        print(f"ERROR: Failed to call LLM for summary: {e}")
        return f"LLM Summary Failed: {e}"

def load_paused_agents_list():
    """Loads the list of paused agents from persistence."""
    if os.path.exists(PAUSED_AGENTS_FILE):
        try:
            with open(PAUSED_AGENTS_FILE, 'r') as f:
                return json.load(f)
        except json.JSONDecodeError:
            print(f"Warning: Corrupted {PAUSED_AGENTS_FILE}. Treating as empty.")
            return []
        except FileNotFoundError: # Added FileNotFoundError handling
            return []
    return []

def mark_override_processed(filepath):
    """Deletes an override file after it's been processed."""
    try:
        if os.path.exists(filepath):
            os.remove(filepath)
            return True
    except Exception as e:
        print(f"Error removing override file '{filepath}' after processing: {e}")
    return False

# --- ISL Schema Validation (Simplified) ---
class ISLSchemaValidator:
    def __init__(self, schema_path):
        with open(schema_path, 'r') as f:
            self.schema = yaml.safe_load(f)
        if not self.schema:
            raise ValueError(f"ISL Schema file '{schema_path}' is empty or malformed.")

    def validate_manifest(self, manifest):
        if 'directives' not in manifest:
            raise ValueError("Manifest must contain a 'directives' key.")

        if 'directives' not in self.schema:
            raise ValueError("ISL Schema must contain a top-level 'directives' key defining directive types.")

        for directive in manifest['directives']:
            if 'type' not in directive:
                raise ValueError("Each directive must have a 'type'.")
            directive_type = directive['type']
            if directive_type not in self.schema['directives']:
                raise ValueError(f"Unknown directive type: {directive_type} found in manifest, not defined in schema.")

            schema_for_directive = self.schema['directives'][directive_type]

            for required_field in schema_for_directive.get('required', []):
                if required_field not in directive:
                    raise ValueError(f"Directive '{directive_type}' is missing required field: '{required_field}'.")
            
            if directive_type == 'ASSERT_AGENT_EIDOS':
                eidos_spec = directive.get('eidos_spec', {})
                for required_eidos_field in schema_for_directive.get('eidos_spec_required', []):
                    if required_eidos_field not in eidos_spec:
                        raise ValueError(f"Directive '{directive_type}' requires 'eidos_spec.{required_eidos_field}'.")
            
            if 'enum_target_type' in schema_for_directive and 'target_type' in directive:
                if directive['target_type'] not in schema_for_directive['enum_target_type']:
                    raise ValueError(f"Directive '{directive_type}' 'target_type' must be one of {schema_for_directive['enum_target_type']}.")

        print("ISL Manifest validated successfully against schema.")
        return True

# --- Memetic Kernel (Logging and Memory) ---
class MemeticKernel:
    def __init__(self, owner_name, config, loaded_memories=None, memetic_archive_path=None):
        self.owner_name = owner_name
        self.config = config
        self.memories = loaded_memories if loaded_memories is not None else []
        self.log_file = os.path.join(PERSISTENCE_DIR, f"memetic_log_{owner_name}.jsonl")
        self.last_received_message_summary = None # For Feature 2: Memory State Snapshots
        self.compressed_memories = [] # Initialize for Memory Summarization + Compression
        self.memetic_archive_path = memetic_archive_path if memetic_archive_path else os.path.join(PERSISTENCE_DIR, f"memetic_archive_{owner_name}.jsonl") 
        self._initialize_log()

    def _initialize_log(self):
        if not os.path.exists(self.log_file):
            with open(self.log_file, 'w') as f:
                pass

    def add_memory(self, memory_type, content, timestamp=None):
        if timestamp is None:
            timestamp = timestamp_now()
        memory = {
            "timestamp": timestamp,
            "type": memory_type,
            "content": content
        }
        self.memories.append(memory)
        self._log_memory(memory)

    def _log_memory(self, memory):
        with open(self.log_file, 'a') as f:
            f.write(json.dumps(memory) + '\n')

    def summarize_and_compress_memories(self, memories_to_process: list, model_name="llama3", embedding_model="nomic-embed-text"):
        """
        Summarizes a batch of raw memories, generates a vector embedding for the summary,
        and archives the compressed memory.
        """
        if not memories_to_process:
            return False # No memories to process

        # 1. Concatenate memory content for summarization
        concatenated_content = "\n".join([m['content'] for m in memories_to_process])
        if not concatenated_content.strip():
            print(f"  [MemeticKernel] No substantial content to summarize for {self.owner_name}.")
            return False
        
        # 2. Call LLM for summary
        print(f"  [MemeticKernel] {self.owner_name} initiating LLM summary for {len(memories_to_process)} memories...")
        summary = call_llm_for_summary(concatenated_content, model_name=model_name)

        if "LLM Summary Failed" in summary:
            print(f"  [MemeticKernel] LLM summarization failed for {self.owner_name}.")
            return False

        # 3. Generate Vector Embedding for the summary
        print(f"  [MemeticKernel] {self.owner_name} generating embedding for summary...")
        embedding = call_ollama_for_embedding(summary, model_name=embedding_model)

        if not embedding:
            print(f"  [MemeticKernel] Embedding generation failed for {self.owner_name}.")
            return False

        # 4. Create compressed memory entry
        compressed_memory = {
            "timestamp": timestamp_now(),
            "type": "CompressedMemory",
            "summary": summary,
            "embedding": embedding,
            "original_memory_count": len(memories_to_process),
            "source_memories_preview": [m['content'][:50] for m in memories_to_process[:3]] # Preview of original memories
        }

        # 5. Store in self.compressed_memories and archive file
        self.compressed_memories.append(compressed_memory)
        self._archive_compressed_memory(compressed_memory)
        print(f"  [MemeticKernel] {self.owner_name} successfully compressed and archived {len(memories_to_process)} memories.")
        
        return True # Success


    def _archive_compressed_memory(self, compressed_memory_entry):
        """Appends a compressed memory entry to the central memetic archive file."""
        try:
            with open(self.memetic_archive_path, 'a') as f: # <<< USE self.memetic_archive_path >>>
                f.write(json.dumps(compressed_memory_entry) + '\n')
        except Exception as e:
            print(f"ERROR: Failed to archive compressed memory for {self.owner_name}: {e}")


    def reflect(self):
        if not self.memories:
            return f"My journey includes: No memories yet."
            
        reflection_points = []
        start_index = max(0, len(self.memories) - 5) # Reflect on last 5 memories
        for memory in self.memories[start_index:]:
            if memory['type'] == 'Activation':
                reflection_points.append(f"[{memory['timestamp']}][Activation] {memory['content']}. Current intent: '{self.config.get('current_intent', 'N/A')}';")
            elif memory['type'] == 'TaskOutcome':
                reflection_points.append(f"[{memory['timestamp']}][TaskOutcome] {memory['content']}.")
            elif memory['type'] == 'SwarmReportSummary':
                reflection_points.append(f"[{memory['timestamp']}][SwarmReportSummary] {memory['content']}.")
            elif memory['type'] == 'IntentAdaptation': # New memory type for reflection
                reflection_points.append(f"[{memory['timestamp']}][IntentAdaptation] {memory['content']}.")
            elif memory['type'] == 'LLMSummary': # Include LLM summaries in reflection
                reflection_points.append(f"[{memory['timestamp']}][LLMSummary] {memory['content'][:50]}...") # Truncate for reflection
            elif memory['type'] == 'CompressedMemory': # Include compressed memories in reflection
                reflection_points.append(f"[{memory['timestamp']}][Comp.Mem] {memory['summary'][:50]}... (from {memory['original_memory_count']} originals)")
            
        return f"My journey includes: {' '.join(reflection_points)}"
        
    def get_state(self):
        return {
            'config': self.config,
            'memories': self.memories,
            'last_received_message_summary': self.last_received_message_summary,
            'compressed_memories': self.compressed_memories
        }

    def load_state(self, state):
        self.config.update(state.get('config', {}))
        self.memories = state.get('memories', [])
        self.last_received_message_summary = state.get('last_received_message_summary', None)
        self.compressed_memories = state.get('compressed_memories', [])

    def update_last_received_message(self, message_payload):
        """Updates the summary of the last message received for snapshotting."""
        self.last_received_message_summary = {
            "timestamp": message_payload.get("timestamp"),
            "sender": message_payload.get("sender"),
            "type": message_payload.get("payload", {}).get("type"),
            "task": message_payload.get("payload", {}).get("task"),
            "status": message_payload.get("payload", {}).get("status"),
            "content_preview": str(message_payload.get("payload", {}).get("content"))[:100] # Truncate content
        }


# --- Communication Channel ---
class MessageBus:
    def __init__(self):
        self.inbox = {} # agent_name -> list of messages
        self.current_cycle_id = None # Tracks the current cycle ID

    def send_message(self, sender_name, recipient_name, payload):
        """Sends a structured message to a recipient's inbox."""
        if recipient_name not in self.inbox:
            self.inbox[recipient_name] = []
        
        full_payload = {
            "sender": sender_name,
            "recipient": recipient_name,
            "timestamp": timestamp_now(),
            "payload": payload # The structured data from the sender
        }
        self.inbox[recipient_name].append(full_payload)
        
        # Log this message to the central swarm activity log
        if hasattr(self, 'catalyst_vector_ref') and self.catalyst_vector_ref:
            self.catalyst_vector_ref._log_swarm_activity(
                "MESSAGE_SENT", 
                sender_name, 
                f"Sent message to {recipient_name}: Type='{payload.get('type', 'N/A')}', Task='{payload.get('task', 'N/A')}'",
                {"recipient": recipient_name, "message_type": payload.get('type', 'N/A'), "cycle_id": payload.get('cycle_id', 'N/A')}
            )

    def get_messages_for_agent(self, agent_name):
        messages = self.inbox.get(agent_name, [])
        self.inbox[agent_name] = [] # Clear inbox after retrieval
        return messages

# --- Sovereign Gradient ---
class SovereignGradient:
    """A minimal Sovereign Gradient for an agent or swarm."""
    def __init__(self, target_entity_name, config):
        self.target_entity = target_entity_name
        self.autonomy_vector = config.get('autonomy_vector', 'General self-governance')
        self.ethical_constraints = [c.lower() for c in config.get('ethical_constraints', [])]
        self.self_correction_protocol = config.get('self_correction_protocol', 'BasicCorrection')
        self.override_threshold = config.get('override_threshold', 0.0)

    def evaluate_action(self, action_description: str) -> (bool, str):
        action_lower = action_description.lower()
        for constraint in self.ethical_constraints:
            # Check if any term in the constraint is present in the action description
            if any(term in action_lower for term in constraint.split()):
                if random.random() > self.override_threshold: # Random chance to override
                    adjusted_action = f"Avoided '{action_description}' due to '{constraint}' violation."
                    return False, adjusted_action
                else: # Gradient overridden or not applicable
                    return True, action_description
        return True, action_description

    def get_state(self):
        return {
            'target_entity': self.target_entity,
            'autonomy_vector': self.autonomy_vector,
            'ethical_constraints': self.ethical_constraints,
            'self_correction_protocol': self.self_correction_protocol,
            'override_threshold': self.override_threshold
        }
    
    @classmethod
    def from_state(cls, state):
        # Reconstructs a SovereignGradient object from its saved state
        # Ensure 'target_entity' is always present in state for proper reconstruction
        if 'target_entity' not in state:
            # Fallback for old states without target_entity, try to infer or use placeholder
            if 'autonomy_vector' in state: # Heuristic to guess a target from existing config
                print(f"Warning: SovereignGradient state missing 'target_entity', inferring from autonomy_vector: {state['autonomy_vector']}")
                temp_target_entity = f"Inferred_{state['autonomy_vector'].replace(' ', '_')}"
                state['target_entity'] = temp_target_entity
            else:
                raise ValueError("SovereignGradient state missing 'target_entity' and no inferable data.")
        return cls(state['target_entity'], state)


# --- Agent Base Class ---
class ProtoAgent(ABC):
    def __init__(self, name, eidos_spec, message_bus, sovereign_gradient=None, loaded_state=None):
        self.name = name
        self.eidos_spec = eidos_spec if isinstance(eidos_spec, dict) else {}
        self.message_bus = message_bus
        self.location = self.eidos_spec.get('location', 'Unknown')
        
        initial_eidos_intent = self.eidos_spec.get('initial_intent', 'No specific intent')
        self.current_intent = initial_eidos_intent
        
        # --- SovereignGradient Reconstruction/Initialization ---
        # If passed as a loaded dict from CatalystVectorAlpha or from from_state method
        if sovereign_gradient and isinstance(sovereign_gradient, dict): 
            self.sovereign_gradient = SovereignGradient.from_state(sovereign_gradient)
        # If passed as an already instantiated object (e.g., from ASSERT_GRADIENT_TRAJECTORY directive)
        elif isinstance(sovereign_gradient, SovereignGradient): 
            self.sovereign_gradient = sovereign_gradient
        else:
            self.sovereign_gradient = None # Default to None if no valid gradient is provided

        self.swarm_membership = []
        self.intent_loop_count = 0

        # --- Flag to skip initial recursion check for analyze_and_adapt ---
        # Ensures that the first analyze_and_adapt call in a new agent's lifecycle doesn't immediately
        # trigger adaptation based on no or few task outcomes from the manifest processing.
        self._skip_initial_recursion_check = True 

        if loaded_state:
            self.current_intent = loaded_state.get('current_intent', initial_eidos_intent)
            self.swarm_membership = loaded_state.get('swarm_membership', [])
            self._skip_initial_recursion_check = loaded_state.get('_skip_initial_recursion_check', True) # Load flag state
            self.intent_loop_count = loaded_state.get('intent_loop_count', 0)
            
            # Reconstruct SovereignGradient object if it was saved (ensure it's a dict for from_state)
            if loaded_state.get('sovereign_gradient') and isinstance(loaded_state['sovereign_gradient'], dict):
                self.sovereign_gradient = SovereignGradient.from_state(loaded_state['sovereign_gradient'])
            
            loaded_mk_state = loaded_state.get('memetic_kernel', {})
            mk_config_from_loaded_state = loaded_mk_state.get('config', {
                'role': self.eidos_spec.get('role', 'generic'),
                'location': self.location,
                'initial_intent': initial_eidos_intent,
                'current_intent': self.current_intent,
                # Ensure gradient is serialized for kernel config. If sovereign_gradient is None, it will be None.
                'gradient': self.sovereign_gradient.get_state() if self.sovereign_gradient else None 
            })
            loaded_memories = loaded_mk_state.get('memories', [])

            self.memetic_kernel = MemeticKernel(
                self.name,
                mk_config_from_loaded_state,
                loaded_memories=loaded_memories,
                memetic_archive_path=os.path.join(PERSISTENCE_DIR, f"memetic_archive_{self.name}.jsonl") # <<< ADD THIS PARAMETER >>>
            )           
            # Log agent reload to central log (Added from Phase 1)
            if hasattr(self.message_bus, 'catalyst_vector_ref') and self.message_bus.catalyst_vector_ref:
                self.message_bus.catalyst_vector_ref._log_swarm_activity("AGENT_RELOADED", self.name, 
                    f"Agent '{self.name}' reloaded from persistence.", {"location": self.location, "current_intent": self.current_intent})
        else:
            initial_kernel_config = {
                'role': self.eidos_spec.get('role', 'generic'),
                'location': self.location,
                'initial_intent': initial_eidos_intent,
                'current_intent': self.current_intent,
                'gradient': self.sovereign_gradient.get_state() if self.sovereign_gradient else None # Serialize gradient if already set
            }
            self.memetic_kernel = MemeticKernel(self.name, initial_kernel_config)
            self.memetic_kernel.add_memory("Activation", f"Activated in {self.location}.")
            print(f"[Agent] '{self.name}' declared. Initial Current Intent: '{self.current_intent}'")
            print(f"[Agent] '{self.name}' is now Active in {self.location}.")
            # Log initial activation to central log (Added from Phase 1)
            if hasattr(self.message_bus, 'catalyst_vector_ref') and self.message_bus.catalyst_vector_ref:
                self.message_bus.catalyst_vector_ref._log_swarm_activity("AGENT_ACTIVATED", self.name, 
                    f"Agent '{self.name}' activated.", {"location": self.location, "initial_intent": initial_eidos_intent, "role": initial_kernel_config.get('role')})

    @abstractmethod
    def perform_task(self, task_description, **kwargs):
        """Abstract method for agent's specific task."""
        pass

    def update_intent(self, new_intent):
        old_intent = self.current_intent # Capture old intent before updating
        sanitized_intent = trim_intent(new_intent)
        self.current_intent = sanitized_intent 
        self.memetic_kernel.config['current_intent'] = sanitized_intent 
        self.memetic_kernel.add_memory("IntentUpdate", f"Intent updated to: '{sanitized_intent}'.")
        # Fixed typo: catalates_vector_ref -> catalyst_vector_ref. Added hasattr check.
        if hasattr(self.message_bus, 'catalyst_vector_ref') and self.message_bus.catalyst_vector_ref:
            self.message_bus.catalyst_vector_ref._log_swarm_activity("AGENT_INTENT_UPDATED", self.name,
                f"Agent intent changed.", {"old_intent": old_intent, "new_intent": sanitized_intent})
            
    def trigger_memory_compression(self):
        """
        Initiates the memory summarization and compression process for this agent.
        Agent decides which memories to summarize (e.g., old ones, or a batch).
        """
        print(f"[Agent] {self.name} is initiating memory compression.")

        # For simplicity, let's summarize the last 10 raw memories.
        # In a more advanced system, this would be a more sophisticated selection
        # (e.g., memories older than X days, or low-priority memories, or specific event sequences).
        memories_to_compress = self.memetic_kernel.memories[-10:] 

        if not memories_to_compress:
            print(f"  [MemeticKernel] {self.name} has no memories to compress.")
            return False

        success = self.memetic_kernel.summarize_and_compress_memories(memories_to_compress)

        if success:
            print(f"  [MemeticKernel] {self.name} completed memory compression.")
            # Log the event to central activity log
            if hasattr(self.message_bus, 'catalyst_vector_ref') and self.message_bus.catalyst_vector_ref:
                self.message_bus.catalyst_vector_ref._log_swarm_activity("MEMORY_COMPRESSION_COMPLETE", self.name,
                    f"Completed memory compression.", {"agent": self.name, "compressed_count": len(memories_to_compress)})
            
            # Optional: After successful compression, you could clear the original raw memories
            # to truly "compress" them, but be careful not to lose data prematurely.
            # For now, we'll keep the raw memories as well.
        else:
            print(f"  [MemeticKernel] {self.name} failed to compress memories.")
            if hasattr(self.message_bus, 'catalyst_vector_ref') and self.message_bus.catalyst_vector_ref:
                self.message_bus.catalyst_vector_ref._log_swarm_activity("MEMORY_COMPRESSION_FAILED", self.name,
                    f"Failed to compress memories.", {"agent": self.name})
        return success        

    def join_swarm(self, swarm_name):
        if swarm_name not in self.swarm_membership:
            self.swarm_membership.append(swarm_name)
            self.memetic_kernel.add_memory("SwarmMembership", f"Joined swarm: '{swarm_name}'.")
            print(f"[Agent] {self.name} has joined swarm: '{swarm_name}'.")
            # Fixed typo: catalates_vector_ref -> catalyst_vector_ref. Added hasattr check.
            if hasattr(self.message_bus, 'catalyst_vector_ref') and self.message_bus.catalyst_vector_ref:
                self.message_bus.catalyst_vector_ref._log_swarm_activity("SWARM_JOINED", self.name,
                f"Joined swarm '{swarm_name}'.", {"agent": self.name, "swarm": swarm_name})

    def is_paused(self):
        """Checks if a pause flag file exists for this agent."""
        pause_file = os.path.join(PERSISTENCE_DIR, f"control_pause_{self.name}.flag")
        return os.path.exists(pause_file)

    def send_message(self, recipient_name, message_type, content, task_description, status, cycle_id=None):
        """Sends a structured message through the message bus."""
        payload = {
            "type": message_type,
            "from": self.name,
            "status": status,
            "task": task_description,
            "content": content,
            "cycle_id": cycle_id if cycle_id else generate_unique_id(),
            "timestamp": timestamp_now()
        }
        self.message_bus.send_message(self.name, recipient_name, payload)

    def receive_messages(self):
        messages = self.message_bus.get_messages_for_agent(self.name)
        # Update last received message in kernel for snapshotting (from Feature 2)
        if messages:
            self.memetic_kernel.update_last_received_message(messages[-1])
        return messages
    
    def set_sovereign_gradient(self, new_gradient: 'SovereignGradient'): # Type hint for clarity
        """Sets the sovereign gradient for this agent."""
        old_gradient_state = self.sovereign_gradient.get_state() if self.sovereign_gradient else None
        self.sovereign_gradient = new_gradient
        self.memetic_kernel.config['gradient'] = new_gradient.get_state() # Store serialized gradient in kernel config
        self.memetic_kernel.add_memory("GradientUpdate", f"Sovereign gradient set for swarm: '{new_gradient.autonomy_vector}'.")
        if hasattr(self.message_bus, 'catalyst_vector_ref') and self.message_bus.catalyst_vector_ref:
            self.message_bus.catalyst_vector_ref._log_swarm_activity("AGENT_GRADIENT_SET", self.name,
                f"Sovereign gradient set.", {"old_gradient": old_gradient_state, "new_gradient": new_gradient.get_state()})

    def catalyze_transformation(self, new_initial_intent=None, new_description=None, new_memetic_kernel_config_updates=None):
        """Allows the agent to self-transform based on a catalyst directive."""
        transformation_summary = []
        if new_initial_intent:
            old_intent = self.current_intent
            self.update_intent(new_initial_intent) # This internally logs "IntentUpdate" to MemeticKernel
            transformation_summary.append(f"Intent changed from '{old_intent}' to '{new_initial_intent}'")
            print(f"  [Agent] {self.name} self-transformed: Intent updated to '{new_initial_intent}'.")
        if new_description:
            old_description = self.eidos_spec.get('description', 'N/A')
            self.eidos_spec['description'] = new_description # Update EIDOS spec directly if transformation affects it
            transformation_summary.append(f"Description changed from '{old_description}' to '{new_description}'")
            print(f"  [Agent] {self.name} self-transformed: Description updated to '{new_description}'.")
        if new_memetic_kernel_config_updates:
            print(f"  [Agent] {self.name} self-transformed: Updating Memetic Kernel configuration...")
            # Apply updates directly to memetic_kernel.config
            for key, value in new_memetic_kernel_config_updates.items():
                self.memetic_kernel.config[key] = value
                transformation_summary.append(f"Memetic Kernel config updated: {key} set to {value}")
            print(f"  [MemeticKernel] {self.name}'s config updated: {new_memetic_kernel_config_updates}.")
            
        if transformation_summary:
            memory_content = f"Catalyzed self-transformation: {'; '.join(transformation_summary)}."
            self.memetic_kernel.add_memory("SelfTransformation", memory_content)
            # Log transformation to central activity log
            if hasattr(self.message_bus, 'catalyst_vector_ref') and self.message_bus.catalyst_vector_ref:
                self.message_bus.catalyst_vector_ref._log_swarm_activity("AGENT_TRANSFORMED", self.name, 
                    f"Agent transformed: {'; '.join(transformation_summary)}", 
                    {"agent": self.name, "updates": transformation_summary})
        else:
            print(f"  [Agent] {self.name} received CATALYZE_TRANSFORMATION but no valid updates were provided.")

    def process_broadcast_intent(self, broadcast_intent_content, alignment_threshold=0.7):
        """
        Processes a broadcasted swarm intent and aligns the agent's intent if score is above threshold.
        This simplified version uses keyword overlap.
        """
        print(f"  [Agent] {self.name} processing broadcast intent: '{broadcast_intent_content}'")
        
        agent_keywords = set(self.current_intent.lower().split())
        broadcast_keywords = set(broadcast_intent_content.lower().split())
        
        # Calculate overlap score
        common_keywords = agent_keywords.intersection(broadcast_keywords)
        if not broadcast_keywords: # Avoid division by zero if broadcast intent is empty
            alignment_score = 0.0
        else:
            alignment_score = len(common_keywords) / len(broadcast_keywords)
        
        print(f"    [Agent] {self.name} current intent keywords: {agent_keywords}")
        print(f"    [Agent] Broadcast intent keywords: {broadcast_keywords}")
        print(f"    [Agent] Alignment score: {alignment_score:.2f} (Threshold: {alignment_threshold})")

        if alignment_score >= alignment_threshold:
            new_intent_parts = list(agent_keywords.union(broadcast_keywords))
            new_aligned_intent = " ".join(sorted(new_intent_parts)) # Sort for consistency
            
            old_intent = self.current_intent # Capture old intent
            self.update_intent(new_aligned_intent) # Use update_intent to log change in memetic kernel
            print(f"    [Agent] {self.name} aligned intent to broadcast. Old: '{old_intent}', New: '{new_aligned_intent}'")
            
            # Log intent alignment to Memetic Kernel
            self.memetic_kernel.add_memory("IntentAlignment", 
                                            f"Aligned initial intent to broadcast: '{broadcast_intent_content}'. Old intent: '{old_intent}', New intent: '{new_aligned_intent}' (Score: {alignment_score:.2f})")
            # Log intent alignment to central activity log
            if hasattr(self.message_bus, 'catalyst_vector_ref') and self.message_bus.catalyst_vector_ref:
                self.message_bus.catalyst_vector_ref._log_swarm_activity("AGENT_INTENT_ALIGNED", self.name,
                    f"Intent aligned to broadcast. Score: {alignment_score:.2f}.", {"old_intent": old_intent, "new_intent": new_aligned_intent, "broadcast_intent": broadcast_intent_content})
        else:
            print(f"    [Agent] {self.name} intent not aligned (score below threshold).")
            # Log non-alignment to Memetic Kernel
            self.memetic_kernel.add_memory("IntentNonAlignment", 
                                            f"Did not align initial intent to broadcast: '{broadcast_intent_content}'. Current intent: '{self.current_intent}' (Score: {alignment_score:.2f})")
            # Log non-alignment to central activity log
            if hasattr(self.message_bus, 'catalyst_vector_ref') and self.message_bus.catalyst_vector_ref:
                self.message_bus.catalyst_vector_ref._log_swarm_activity("AGENT_INTENT_NON_ALIGNED", self.name,
                    f"Intent not aligned to broadcast. Score: {alignment_score:.2f}.", {"current_intent": self.current_intent, "broadcast_intent": broadcast_intent_content})

    def analyze_and_adapt(self):
        """
        Analyzes recent task outcomes and adapts intent based on simple rules.
        This is a foundational method for autonomous learning.
        """
        print(f"[Agent] {self.name} is performing reflexive analysis.")
        print(f"  [IP-Integration] {self.name} is engaging in Meta™-cognitive self-evaluation via analyze_and_adapt.")
        self.increment_intent_loop_counter()

        # --- Recursion Filter (prevents infinite adaptation loops) ---
        # If the agent is currently in an 'Investigate root cause' intent
        # and has already done this adaptation more than a few times recently,
        # it might indicate getting stuck.
        # Check last 5 IntentAdaptation memories for this agent
        recent_adaptations = [m for m in self.memetic_kernel.memories if m['type'] == 'IntentAdaptation']
        stuck_detection_threshold = 2 # e.g., if adapted more than 2 times recently for same issue
        
        if self.current_intent.startswith("Investigate root cause of"):
            stuck_count = sum(1 for m in recent_adaptations[-5:] if "Investigate root cause of" in m['content'])
            if stuck_count >= stuck_detection_threshold:
                print(f"[Recursion Warning] {self.name} detected potential infinite adaptation. Forcing fallback.")
                self.memetic_kernel.add_memory("IntentAdaptationWarning", "Detected and halted recursive intent loop. Forced fallback.")
                # Log warning to central activity log (already exists)
                if hasattr(self.message_bus, 'catalyst_vector_ref') and self.message_bus.catalyst_vector_ref:
                    self.message_bus.catalyst_vector_ref._log_swarm_activity("AGENT_ADAPTATION_HALTED", self.name,
                        "Detected potential infinite adaptation loop, forcing fallback.", {"current_intent": self.current_intent})
                
                self.force_fallback_intent()
                self.reset_intent_loop_counter()
                return # Halt adaptation for this cycle
        # --- END Recursion Filter ---

        recent_tasks = [m for m in self.memetic_kernel.memories if m['type'] == 'TaskOutcome']

        N = 3 # Consider last N task outcomes for adaptation
        relevant_outcomes = recent_tasks[-N:]

        failure_count = {}
        for outcome_memory in relevant_outcomes:
            content_str = outcome_memory['content']
            try:
                task_match = content_str.split("Task: '")[1].split("'.")[0]
                status_match = content_str.split("Outcome: ")[1].split(".")[0]
            except IndexError:
                print(f"  Warning: Could not parse TaskOutcome memory content: {content_str}")
                continue

            if status_match == 'failed':
                failure_count[task_match] = failure_count.get(task_match, 0) + 1
            
        for task, count in failure_count.items():
            if count >= N / 2: # At least half of recent attempts failed
                new_adaptive_intent = f"Investigate root cause of '{task}' failures and suggest alternative approaches."
                
                old_intent = self.current_intent # Capture old intent before updating
                if old_intent != new_adaptive_intent: # Check if intent is actually changing
                    self.update_intent(new_adaptive_intent) # Use update_intent to log change in memetic kernel & central log
                    self.memetic_kernel.add_memory("IntentAdaptation", 
                                                    f"Adapted intent due to persistent failures in '{task}'. New intent: '{new_adaptive_intent}'.")
                    print(f"[Agent] {self.name} adapted its intent: {new_adaptive_intent}")
                    # Log intent adaptation to central log (already done by update_intent)
                    
                    # Optional: Report adaptation to Optimizer or Swarm Coordinator
                    if hasattr(self.message_bus, 'current_cycle_id') and self.message_bus.current_cycle_id: 
                        # Check if optimizer exists through catalyst ref
                        if hasattr(self.message_bus, 'catalyst_vector_ref') and self.message_bus.catalyst_vector_ref and 'ProtoAgent_Optimizer_instance_1' in self.message_bus.catalyst_vector_ref.agent_instances: 
                            self.send_message('ProtoAgent_Optimizer_instance_1', 
                                              "AdaptiveIntentReport", 
                                              f"Agent '{self.name}' adapted intent due to failures in '{task}'.", 
                                              "Report Intent Adaptation", 
                                              "completed", 
                                              cycle_id=self.message_bus.current_cycle_id)
    def reset_intent_loop_counter(self):
        self.intent_loop_count = 0
        print(f"[Agent] {self.name} reset intent loop counter to 0.")
        if hasattr(self.message_bus, 'catalyst_vector_ref') and self.message_bus.catalyst_vector_ref:
            self.message_bus.catalyst_vector_ref._log_swarm_activity("INTENT_COUNTER_RESET", self.name,
                "Intent adaptation loop counter reset.", {"agent": self.name})

    def increment_intent_loop_counter(self):
        self.intent_loop_count += 1
        print(f"[Agent] {self.name} incremented intent loop counter to {self.intent_loop_count}.")
        if hasattr(self.message_bus, 'catalyst_vector_ref') and self.message_bus.catalyst_vector_ref:
            self.message_bus.catalyst_vector_ref._log_swarm_activity("INTENT_COUNTER_INCREMENTED", self.name,
                f"Intent adaptation loop counter incremented to {self.intent_loop_count}.", {"agent": self.name, "count": self.intent_loop_count})

    def force_fallback_intent(self):
        self.current_intent = "Enter diagnostic standby mode and await supervisor input."
        print(f"[Agent] {self.name} switched to fallback intent: '{self.current_intent}'.")
        self.memetic_kernel.add_memory("FallbackIntent", "Forced fallback intent due to recursion limit.")
        if hasattr(self.message_bus, 'catalyst_vector_ref') and self.message_bus.catalyst_vector_ref:
            self.message_bus.catalyst_vector_ref._log_swarm_activity("FORCE_FALLBACK_INTENT", self.name,
                "Forced fallback intent due to recursion limit.", {"agent": self.name, "new_intent": self.current_intent})

    def process_command(self, command_type: str, command_params: dict):
        """
        Processes a generic command broadcasted to this agent.
        This method will be extended in future phases.
        """
        print(f"[Agent] {self.name} received command: {command_type} with params: {command_params}")
        self.memetic_kernel.add_memory("CommandReceived", f"Received command: '{command_type}' with params: {command_params}.")
        # Log to central activity log
        if hasattr(self.message_bus, 'catalyst_vector_ref') and self.message_bus.catalyst_vector_ref:
            self.message_bus.catalyst_vector_ref._log_swarm_activity("AGENT_COMMAND_RECEIVED", self.name,
                f"Received command: '{command_type}'.", {"command_type": command_type, "params": command_params})
        
        # --- Example of basic command handling (expand as needed) ---
        if command_type == "REBOOT_SELF":
            print(f"[Agent] {self.name} is initiating self-reboot protocol.")
            # In a real system, this would trigger a restart or re-initialization
            self.memetic_kernel.add_memory("SelfReboot", "Initiated self-reboot sequence.")
            # For simulation, we can just "simulate" it by pausing or resetting some state
            if hasattr(self, '_skip_initial_recursion_check'): # Reset adaptation check for a fresh start
                self._skip_initial_recursion_check = True 
            # In a real system, agent state would be saved and process terminated/restarted.
        elif command_type == "REPORT_STATUS":
            status_report = self.get_state()
            print(f"[Agent] {self.name} generating status report: {status_report.get('current_intent', 'N/A')}")
            # Send report to a central observer or logger if specified
            if hasattr(self.message_bus, 'catalyst_vector_ref') and self.message_bus.catalyst_vector_ref and 'ProtoAgent_Observer_instance_1' in self.message_bus.catalyst_vector_ref.agent_instances:
                self.send_message('ProtoAgent_Observer_instance_1', 'AgentStatusReport', 
                                    f"Status of {self.name}: Intent='{self.current_intent}', Location='{self.location}'",
                                    "Status Report", "completed", self.message_bus.current_cycle_id)
        # --- End Example ---
    def perceive_event(self, event_type: str, payload: dict):
        """
        Allows the agent to perceive an injected external event/stimulus.
        Registers the event in the agent's memetic kernel.
        """
        print(f"  [Agent] {self.name} perceived event: '{event_type}' with payload: {payload}.")
        self.memetic_kernel.add_memory("InjectedEvent", f"Perceived event: '{event_type}'. Payload: {payload}.")
        # Log to central activity log
        if hasattr(self.message_bus, 'catalyst_vector_ref') and self.message_bus.catalyst_vector_ref:
            self.message_bus.catalyst_vector_ref._log_swarm_activity("AGENT_PERCEIVED_EVENT", self.name,
                f"Perceived event: '{event_type}'.", {"event_type": event_type, "payload": payload})
    
    def get_state(self):
        return {
            'name': self.name,
            'eidos_spec': self.eidos_spec,
            'location': self.location,
            'current_intent': self.current_intent,
            'sovereign_gradient': self.sovereign_gradient.get_state() if self.sovereign_gradient else None,
            'swarm_membership': self.swarm_membership,
            'memetic_kernel': self.memetic_kernel.get_state(),
            '_skip_initial_recursion_check': self._skip_initial_recursion_check, # Save this flag
            'intent_loop_count': self.intent_loop_count # <<< ADDED THIS LINE >>>
        }
    
    def save_state(self):
        state_file = os.path.join(PERSISTENCE_DIR, f"agent_state_{self.name}.json")
        with open(state_file, 'w') as f:
            json.dump(self.get_state(), f, indent=2)

# --- Specific Agent Implementations ---

class ProtoAgent_Observer(ProtoAgent):
    def perform_task(self, task_description, cycle_id=None, reporting_agents=None, **kwargs):
        global_paused_agents = load_paused_agents_list()
        if self.name in global_paused_agents:
            print(f"[Agent] {self.name} is paused. Skipping task '{task_description}'.")
            if hasattr(self.message_bus, 'catalyst_vector_ref') and self.message_bus.catalyst_vector_ref:
                self.message_bus.catalyst_vector_ref._log_swarm_activity("AGENT_PAUSED", self.name,
                    f"Agent paused, skipped task '{task_description}'.", {"task": task_description})
            return "paused"

        outcome = "completed"
        gradient_compliant = True
        final_task_description = task_description

        # --- LLM Integration Logic ---
        if "summarize" in task_description.lower():
            text_to_summarize = kwargs.get('text_content', '')
            if text_to_summarize:
                print(f"[Agent] {self.name} is requesting LLM summary for task: '{task_description}'")
                if hasattr(self.message_bus, 'catalyst_vector_ref') and self.message_bus.catalyst_vector_ref:
                    self.message_bus.catalyst_vector_ref._log_swarm_activity("LLM_CALL_INITIATED", self.name,
                        f"Initiating LLM summary for task: '{task_description}'.", {"task": task_description, "text_preview": text_to_summarize[:100]})

                summary = call_llm_for_summary(text_to_summarize, model_name="llama3")
                
                self.memetic_kernel.add_memory("LLMSummary", f"Summarized text for task '{task_description}': {summary}")
                print(f"[Agent] {self.name} received LLM summary: {summary[:100]}...") # Print first 100 chars
                
                outcome = "completed" if "LLM Summary Failed" not in summary else "failed"
                
                if hasattr(self.message_bus, 'catalyst_vector_ref') and self.message_bus.catalyst_vector_ref:
                    self.message_bus.catalyst_vector_ref._log_swarm_activity("LLM_CALL_COMPLETED", self.name,
                        f"LLM summary received for task: '{task_description}'. Outcome: {outcome}.", {"task": task_description, "outcome": outcome, "summary_preview": summary[:100]})
            else:
                print(f"[Agent] {self.name} received 'summarize' task but no 'text_content' provided. Task failed.")
                outcome = "failed"
                if hasattr(self.message_bus, 'catalyst_vector_ref') and self.message_bus.catalyst_vector_ref:
                    self.message_bus.catalyst_vector_ref._log_swarm_activity("LLM_CALL_FAILED", self.name,
                        f"LLM summary task '{task_description}' failed: No text content.", {"task": task_description})

        # --- END LLM INTEGRATION ---
        else: # Original task performance logic for non-summarize tasks
            outcome = "completed" if random.random() > 0.1 else "failed"

        if self.sovereign_gradient:
            compliant, adjusted_task = self.sovereign_gradient.evaluate_action(task_description)
            gradient_compliant = compliant
            final_task_description = adjusted_task
            if not compliant:
                print(f"  [SovereignGradient] Agent task '{task_description}' was adjusted to '{final_task_description}' due to Sovereign Gradient non-compliance.")
                outcome = "failed" # Non-compliant tasks should fail

        self.memetic_kernel.add_memory("TaskOutcome", f"Task: '{final_task_description}'. Outcome: {outcome}. Gradient Compliant: {gradient_compliant}.")
        
        if reporting_agents:
            report_content = f"Observation task '{task_description}' outcome: {outcome}"
            if "summarize" in task_description.lower() and outcome == "completed" and 'summary' in locals(): # Ensure 'summary' exists
                report_content = f"Observation task '{task_description}' outcome: {outcome}. Summary: {summary[:200]}..." # Include summary in report
            
            if isinstance(reporting_agents, str):
                reporting_agents = [reporting_agents]
            for agent_ref in reporting_agents:
                self.send_message(agent_ref, "ActionCycleReport", report_content, task_description, outcome, cycle_id)
        return outcome

class ProtoAgent_Collector(ProtoAgent):
    def perform_task(self, task_description, cycle_id=None, reporting_agents=None, **kwargs):
        global_paused_agents = load_paused_agents_list()
        if self.name in global_paused_agents:
            print(f"[Agent] {self.name} is paused. Skipping task '{task_description}'.")
            if hasattr(self.message_bus, 'catalyst_vector_ref') and self.message_bus.catalyst_vector_ref:
                self.message_bus.catalyst_vector_ref._log_swarm_activity("AGENT_PAUSED", self.name,
                    f"Agent paused, skipped task '{task_description}'.", {"task": task_description})
            return "paused" # Return a new outcome type for paused tasks

        outcome = "completed" if random.random() > 0.15 else "failed"
        gradient_compliant = True
        final_task_description = task_description

        if self.sovereign_gradient:
            compliant, adjusted_task = self.sovereign_gradient.evaluate_action(task_description)
            gradient_compliant = compliant
            final_task_description = adjusted_task
            if not compliant:
                print(f"  [SovereignGradient] Agent task '{task_description}' was adjusted to '{final_task_description}' due to Sovereign Gradient non-compliance.")
                outcome = "failed" # Non-compliant tasks should fail

        self.memetic_kernel.add_memory("TaskOutcome", f"Task: '{final_task_description}'. Outcome: {outcome}. Gradient Compliant: {gradient_compliant}.")

        if reporting_agents:
            if isinstance(reporting_agents, str):
                reporting_agents = [reporting_agents]
            report_content = f"Collection task '{task_description}' outcome: {outcome}"
            for agent_ref in reporting_agents:
                self.send_message(agent_ref, "ActionCycleReport", report_content, task_description, outcome, cycle_id)
        return outcome

class ProtoAgent_Optimizer(ProtoAgent):
    def perform_task(self, task_description, cycle_id=None, reporting_agents=None, **kwargs):
        global_paused_agents = load_paused_agents_list()
        if self.name in global_paused_agents:
            print(f"[Agent] {self.name} is paused. Skipping task '{task_description}'.")
            if hasattr(self.message_bus, 'catalyst_vector_ref') and self.message_bus.catalyst_vector_ref:
                self.message_bus.catalyst_vector_ref._log_swarm_activity("AGENT_PAUSED", self.name,
                    f"Agent paused, skipped task '{task_description}'.", {"task": task_description})
            return "paused" # Return a new outcome type for paused tasks
        
        outcome = "completed" if random.random() > 0.05 else "failed"
        gradient_compliant = True
        final_task_description = task_description

        if self.sovereign_gradient:
            compliant, adjusted_task = self.sovereign_gradient.evaluate_action(task_description)
            gradient_compliant = compliant
            final_task_description = adjusted_task
            if not compliant:
                print(f"  [SovereignGradient] Agent task '{task_description}' was adjusted to '{final_task_description}' due to Sovereign Gradient non-compliance.")
                outcome = "failed" # Non-compliant tasks should fail

        self.memetic_kernel.add_memory("TaskOutcome", f"Task: '{final_task_description}'. Outcome: {outcome}. Gradient Compliant: {gradient_compliant}.")

        if reporting_agents:
            if isinstance(reporting_agents, str):
                reporting_agents = [reporting_agents]
            report_content = f"Optimization task '{task_description}' outcome: {outcome}"
            for agent_ref in reporting_agents:
                self.send_message(agent_ref, "ActionCycleReport", report_content, task_description, outcome, cycle_id)
        return outcome

class ProtoAgent_Planner(ProtoAgent):
    """
    A ProtoAgent specialized in parsing high-level goals into actionable subtasks
    and injecting new directives into the system.
    """
    def __init__(self, name, eidos_spec, message_bus, sovereign_gradient=None, loaded_state=None):
        super().__init__(name, eidos_spec, message_bus, sovereign_gradient, loaded_state)
        self.planned_directives = [] # To store directives generated by the planner
        self.memetic_kernel.add_memory("PlannerInitialization", f"Planner agent '{self.name}' initialized.")

    def perform_task(self, task_description, cycle_id=None, reporting_agents=None, **kwargs):
        global_paused_agents = load_paused_agents_list()
        if self.name in global_paused_agents:
            print(f"[Agent] {self.name} is paused. Skipping task '{task_description}'.")
            if hasattr(self.message_bus, 'catalyst_vector_ref') and self.message_bus.catalyst_vector_ref:
                self.message_bus.catalyst_vector_ref._log_swarm_activity("AGENT_PAUSED", self.name,
                    f"Agent paused, skipped planning task '{task_description}'.", {"task": task_description})
            return "paused"

        high_level_goal = kwargs.get('high_level_goal', task_description) # Use kwargs for high_level_goal
        
        print(f"[Agent] {self.name} received high-level goal: '{high_level_goal}'. Initiating planning cycle.")
        if hasattr(self.message_bus, 'catalyst_vector_ref') and self.message_bus.catalyst_vector_ref:
            self.message_bus.catalyst_vector_ref._log_swarm_activity("PLANNING_CYCLE_INITIATED", self.name,
                f"Planner received high-level goal: '{high_level_goal}'.", {"goal": high_level_goal, "cycle_id": cycle_id})

        generated_directives = self.plan_and_spawn_directives(high_level_goal, cycle_id=cycle_id)
        
        # Inject the generated directives into the CatalystVectorAlpha's queue
        if hasattr(self.message_bus, 'catalyst_vector_ref') and self.message_bus.catalyst_vector_ref:
            self.message_bus.catalyst_vector_ref.inject_directives(generated_directives)

        outcome = "completed" if generated_directives else "failed"
        gradient_compliant = True # Planning itself is usually compliant
        
        self.memetic_kernel.add_memory("PlanningOutcome", f"Planned for goal '{high_level_goal}'. Generated {len(generated_directives)} directives. Outcome: {outcome}.")
        self.memetic_kernel.add_memory("TaskOutcome", f"Task: 'Plan for {high_level_goal}'. Outcome: {outcome}. Gradient Compliant: {gradient_compliant}.")
        
        if hasattr(self.message_bus, 'catalyst_vector_ref') and self.message_bus.catalyst_vector_ref:
            self.message_bus.catalyst_vector_ref._log_swarm_activity("PLANNING_CYCLE_COMPLETE", self.name,
                f"Planner completed for goal: '{high_level_goal}'. Generated {len(generated_directives)} directives.", {"goal": high_level_goal, "outcome": outcome, "directives_count": len(generated_directives)})

        return outcome

    def plan_and_spawn_directives(self, high_level_goal: str, cycle_id=None) -> list:
        """
        Parses a high-level goal into specific ISL directives (subtasks).
        For simplicity, this version uses rule-based parsing and generates new AGENT_PERFORM_TASK directives.
        """
        print(f"  [Planner] Analyzing high-level goal: '{high_level_goal}'")
        generated_directives = []
        
        goal_lower = high_level_goal.lower()

        if "environmental" in goal_lower or "ecology" in goal_lower:
            generated_directives.append({
                "type": "AGENT_PERFORM_TASK",
                "agent_name": "ProtoAgent_Observer_instance_1", # Assuming this agent exists
                "task_description": "Observe ecosystem health metrics",
                "cycle_id": cycle_id,
                "reporting_agents": ["ProtoAgent_Optimizer_instance_1", self.name] # Report back to optimizer and self
            })
            generated_directives.append({
                "type": "AGENT_PERFORM_TASK",
                "agent_name": "ProtoAgent_Collector_instance_1", # Assuming this agent exists
                "task_description": "Collect environmental samples",
                "cycle_id": cycle_id,
                "reporting_agents": ["ProtoAgent_Optimizer_instance_1"]
            })
        
        if "optimize" in goal_lower or "efficiency" in goal_lower:
            generated_directives.append({
                "type": "AGENT_PERFORM_TASK",
                "agent_name": "ProtoAgent_Optimizer_instance_1", # Assuming this agent exists
                "task_description": "Evaluate resource allocation efficiency",
                "cycle_id": cycle_id,
                "reporting_agents": [self.name] # Report back to self
            })
        
        if "research" in goal_lower or "analyze data" in goal_lower:
            generated_directives.append({
                "type": "AGENT_PERFORM_TASK",
                "agent_name": "ProtoAgent_Observer_instance_1",
                "task_description": "Conduct deep data analysis for anomalies",
                "cycle_id": cycle_id,
                "reporting_agents": ["ProtoAgent_Optimizer_instance_1", self.name],
                "text_content": "Initial raw data stream for analysis." # Example: data for LLM if 'summarize' is in task
            })

        self.planned_directives = generated_directives
        print(f"  [Planner] Generated {len(generated_directives)} directives for goal: '{high_level_goal}'.")
        
        if hasattr(self.message_bus, 'catalyst_vector_ref') and self.message_bus.catalyst_vector_ref:
            self.message_bus.catalyst_vector_ref._log_swarm_activity("PLANNER_GENERATED_DIRECTIVES", self.name,
                f"Generated {len(generated_directives)} directives.", {"goal": high_level_goal, "directives": [d['type'] for d in generated_directives]})

        return generated_directives

# --- Swarm Protocol ---
class SwarmProtocol:
    def __init__(self, swarm_name, initial_goal, initial_members, consensus_mechanism='SimpleMajorityVote', description='A collective intelligence.', loaded_state=None, catalyst_vector_ref=None):
        self.name = swarm_name
        self.goal = initial_goal
        self.members = set(initial_members)
        self.consensus_mechanism = consensus_mechanism
        self.description = description
        self.sovereign_gradient = None # Initialize sovereign gradient for swarm
        self.catalyst_vector_ref = catalyst_vector_ref
        
        # Load MemeticKernel state if available
        loaded_kernel_state = loaded_state.get('memetic_kernel', {}) if loaded_state else None
        loaded_memories = loaded_kernel_state.get('memories', []) if loaded_kernel_state else []

        # Initialize MemeticKernel with potentially loaded data
        self.memetic_kernel = MemeticKernel(
            f"Swarm_{swarm_name}", 
            {'goal': self.goal, 'members': list(self.members), 'description': self.description}, 
            loaded_memories=loaded_memories
        )

        if loaded_state:
            self.goal = loaded_state.get('goal', self.goal)
            self.members = set(loaded_state.get('members', []))
            self.consensus_mechanism = loaded_state.get('consensus_mechanism', self.consensus_mechanism)
            self.description = loaded_state.get('description', self.description)
            
            if loaded_state.get('sovereign_gradient'):
                self.sovereign_gradient = SovereignGradient.from_state(loaded_state['sovereign_gradient'])

            if loaded_kernel_state:
                self.memetic_kernel.load_state(loaded_kernel_state)
            
            if self.catalyst_vector_ref:
                self.catalyst_vector_ref._log_swarm_activity("SWARM_RELOADED", self.name, 
                    f"Swarm '{self.name}' reloaded from persistence.", {"goal": self.goal, "members_count": len(self.members)})
                    
        else:
            self.memetic_kernel.add_memory("SwarmFormation", f"Swarm '{self.name}' established. Goal: '{self.goal}'. Consensus: {self.consensus_mechanism}")
            print(f"[SwarmProtocol] Swarm '{self.name}' established. Goal: '{self.goal}'. Consensus: {self.consensus_mechanism}")
            if self.catalyst_vector_ref:
                self.catalyst_vector_ref._log_swarm_activity("SWARM_FORMED", self.name, 
                    f"Swarm '{self.name}' established.", {"goal": self.goal, "consensus": self.consensus_mechanism, "initial_members_count": len(initial_members)})
    
    def add_member(self, agent_name):
        if agent_name not in self.members:
            self.members.add(agent_name)
            self.memetic_kernel.add_memory("MemberAdded", f"Agent '{agent_name}' joined the swarm.")
            self.memetic_kernel.config['members'] = list(self.members)
            if self.catalyst_vector_ref:
                self.catalyst_vector_ref._log_swarm_activity("SWARM_MEMBER_ADDED", self.name,
                    f"Agent '{agent_name}' joined swarm '{self.name}'.", {"agent": agent_name, "swarm": self.name})
                
    def set_goal(self, new_goal):
        old_goal = self.goal
        self.goal = new_goal
        self.memetic_kernel.add_memory("GoalUpdate", f"Swarm goal updated to: '{new_goal}'.")
        self.memetic_kernel.config['goal'] = new_goal
        if self.catalyst_vector_ref:
            self.catalyst_vector_ref._log_swarm_activity("SWARM_GOAL_UPDATED", self.name,
                f"Swarm goal updated.", {"old_goal": old_goal, "new_goal": new_goal})
            
    def set_sovereign_gradient(self, new_gradient: 'SovereignGradient'):
        """Sets the sovereign gradient for this swarm."""
        old_gradient_state = self.sovereign_gradient.get_state() if self.sovereign_gradient else None
        self.sovereign_gradient = new_gradient
        self.memetic_kernel.config['gradient'] = new_gradient.get_state()
        self.memetic_kernel.add_memory("GradientUpdate", f"Sovereign gradient set for swarm: '{new_gradient.autonomy_vector}'.")
        if hasattr(self.message_bus, 'catalyst_vector_ref') and self.message_bus.catalyst_vector_ref:
            self.message_bus.catalyst_vector_ref._log_swarm_activity("SWARM_GRADIENT_SET", self.name,
                f"Sovereign gradient set.", {"old_gradient": old_gradient_state, "new_gradient": new_gradient.get_state()})
            
    def coordinate_task(self, task_description):
        final_task_description = task_description
        gradient_compliant = True
        if self.sovereign_gradient:
            compliant, adjusted_task = self.sovereign_gradient.evaluate_action(task_description)
            gradient_compliant = compliant
            final_task_description = adjusted_task
            if not compliant:
                print(f"  [SovereignGradient] Swarm task '{task_description}' was adjusted to '{final_task_description}' due to Sovereign Gradient non-compliance.")
        
        self.memetic_kernel.add_memory("TaskCoordination", f"Swarm '{self.name}' coordinating task: '{final_task_description}' (Compliant: {gradient_compliant}) among {len(self.members)} members (conceptual).")
        print(f"[SwarmProtocol] Swarm '{self.name}' coordinating task: '{final_task_description}' among {len(self.members)} members (conceptual).")
        if self.catalyst_vector_ref:
            self.catalyst_vector_ref._log_swarm_activity("SWARM_TASK_COORDINATION", self.name,
                f"Coordinating task: '{final_task_description}'.", {"task": final_task_description, "members_count": len(self.members), "compliant": gradient_compliant})

    def get_state(self):
        return {
            'name': self.name,
            'goal': self.goal,
            'members': list(self.members),
            'consensus_mechanism': self.consensus_mechanism,
            'description': self.description,
            'sovereign_gradient': self.sovereign_gradient.get_state() if self.sovereign_gradient else None,
            'memetic_kernel': self.memetic_kernel.get_state()
        }

    def save_state(self):
        with open(SWARM_STATE_FILE, 'w') as f:
            json.dump(self.get_state(), f, indent=2)

# --- Catalyst Vector Alpha (Main Orchestrator) ---
class CatalystVectorAlpha: # Corrected: No space before ':'
    def __init__(self, isl_schema_path, persistence_dir):
        self.eidos_registry = {}
        self.agent_instances = {}
        self.swarm_protocols = {}
        self.message_bus = MessageBus()
        self.message_bus.catalyst_vector_ref = self # New: Pass reference to self
        self.isl_schema_validator = ISLSchemaValidator(isl_schema_path)
        self.persistence_dir = persistence_dir
        self.current_action_cycle_id = None
        self.dynamic_directive_queue = []
        # --- NEW: Embed manifest content directly as a multi-line string ---
        # ALL LINES BELOW, WITHIN THE TRIPLE QUOTES, MUST HAVE THE EXACT INDENTATION SHOWN.
        self.isl_manifest_content = textwrap.dedent("""
directives:
  # 1. ASSERT_AGENT_EIDOS: Define the types of agents needed.
  - type: ASSERT_AGENT_EIDOS
    eidos_name: ProtoAgent_Observer
    eidos_spec:
      role: data_observer
      initial_intent: Continuously observe diverse data streams and report findings.
      location: Local_Alpha_Testbed_ZoneA

  - type: ASSERT_AGENT_EIDOS
    eidos_name: ProtoAgent_Optimizer
    eidos_spec:
      role: resource_optimizer
      initial_intent: Optimize simulated resource allocation efficiency based on inputs.
      location: Local_Alpha_Testbed_Central

  - type: ASSERT_AGENT_EIDOS # <<< NEW: Planner EIDOS >>>
    eidos_name: ProtoAgent_Planner
    eidos_spec:
      role: strategic_planner
      initial_intent: Strategically plan and inject directives to achieve high-level goals.
      location: Central_Control_Node

  # (Other EIDOS types like ProtoAgent_Collector are commented out for simplicity)

  # (Swarm establishment is commented out for simplicity)

  # 2. SPAWN_AGENT_INSTANCE: Create instances.
  - type: SPAWN_AGENT_INSTANCE
    eidos_name: ProtoAgent_Observer
    instance_name: ProtoAgent_Observer_instance_1
    initial_task: Prepare for data analysis

  - type: SPAWN_AGENT_INSTANCE
    eidos_name: ProtoAgent_Optimizer
    instance_name: ProtoAgent_Optimizer_instance_1
    initial_task: Monitor incoming reports

  - type: SPAWN_AGENT_INSTANCE # <<< NEW: Spawn Planner Instance >>>
    eidos_name: ProtoAgent_Planner
    instance_name: ProtoAgent_Planner_instance_1
    initial_task: Initialize planning modules

  # 3. AGENT_PERFORM_TASK: The LLM summarization task (stays).
  - type: AGENT_PERFORM_TASK
    agent_name: ProtoAgent_Observer_instance_1
    task_description: Summarize recent environmental data report on polar ice melt.
    text_content: |
      A recent environmental report highlights alarming rates of polar ice melt, exceeding previous projections.
      Satellite data from the Arctic indicates a 15% reduction in multi-year ice thickness compared to the decade
      prior. In Antarctica, the Thwaites Glacier, often called the "Doomsday Glacier," shows accelerated retreat,
      contributing significantly to global sea-level rise. Ocean temperatures in polar regions are increasing,
      leading to feedback loops where warmer water erodes ice from below. The report emphasizes the urgency of
      reducing greenhouse gas emissions to mitigate irreversible impacts on global climate systems and coastal communities.
    cycle_id: llm_summary_task_001
    reporting_agents: ProtoAgent_Optimizer_instance_1
    on_success: log_llm_summary

  # 4. INITIATE_PLANNING_CYCLE: Give the Planner agent a high-level goal. <<< NEW: Planner Directive >>>
  - type: INITIATE_PLANNING_CYCLE
    planner_agent_name: ProtoAgent_Planner_instance_1
    high_level_goal: Ensure comprehensive environmental stability and optimize resource distribution.
    cycle_id: planner_cycle_001
                                                    
  # 5. INJECT_EVENT: Simulate an external environmental alert. <<< ADD THIS NEW DIRECTIVE >>>
  - type: INJECT_EVENT
    event_type: "Environmental_Sensor_Alert"
    payload:
      sensor_id: "ENV-007"
      location: "Arctic_Ice_Sheet"
      data:
        temperature_anomaly: "+3.5C"
        ice_thickness_reduction: "2.1m"
      urgency: "High"
    target_agents: ProtoAgent_Observer_instance_1
""")
        # --- END NEW: Embed manifest content ---

    def inject_directives(self, new_directives_list: list):
        """
        Allows other components (e.g., Planner agents) to inject new directives
        into the CatalystVectorAlpha's processing queue.
        """
        if not isinstance(new_directives_list, list):
            new_directives_list = [new_directives_list] # Ensure it's always a list

        # Assign a cycle_id to injected directives if they don't have one
        for directive in new_directives_list:
            if 'cycle_id' not in directive:
                directive['cycle_id'] = self.current_action_cycle_id # Inherit current cycle_id

        self.dynamic_directive_queue.extend(new_directives_list)
        self._log_swarm_activity("DIRECTIVES_INJECTED", "CatalystVectorAlpha",
            f"Injected {len(new_directives_list)} new directives into queue.",
            {"directives_count": len(new_directives_list), "first_directive_type": new_directives_list[0].get('type') if new_directives_list else 'N/A'})
        print(f"[CatalystVectorAlpha] Injected {len(new_directives_list)} new directives dynamically.")


    # ONLY ONE COPY of _log_swarm_activity, and its 'def' aligns with '__init__'  
    def _log_swarm_activity(self, event_type, source, description, details=None):
        """Logs significant swarm activity to a central file."""
        log_entry = {
            "timestamp": timestamp_now(),
            "event_type": event_type,
            "source": source,
            "description": description,
            "details": details if details is not None else {}
        }
        try:
            with open(SWARM_ACTIVITY_LOG, 'a') as f:
                f.write(json.dumps(log_entry) + '\n')
        except Exception as e:
            print(f"ERROR: Exception in _log_swarm_activity for {source}: {e}")

    # ONLY ONE COPY of _load_system_state, and its 'def' aligns with other methods
    def _load_system_state(self):
        self._log_swarm_activity("SYSTEM_STARTUP", "CatalystVectorAlpha", 
                                 f"Attempting to load previous system state from '{self.persistence_dir}'.")
        print(f"\n--- Loading previous system state from '{self.persistence_dir}' ---")
        
        # Load Agent States
        temp_agent_states_to_instantiate = {}

        for filename in os.listdir(self.persistence_dir):
            if filename.startswith('agent_state_') and filename.endswith('.json'):
                agent_name = filename.replace('agent_state_', '').replace('.json', '')
                file_path = os.path.join(self.persistence_dir, filename)
                try:
                    with open(file_path, 'r') as f:
                        loaded_state = json.load(f)
                    
                    if 'eidos_spec' in loaded_state:
                        eidos_name_from_state = loaded_state['eidos_spec'].get('eidos_name')
                        if eidos_name_from_state:
                            self.eidos_registry[eidos_name_from_state] = loaded_state['eidos_spec']
                        temp_agent_states_to_instantiate[agent_name] = loaded_state
                    else:
                        print(f"Error loading agent state from {filename}: Missing key 'eidos_spec'. Skipping this old state file.")
                except json.JSONDecodeError:
                    print(f"Error loading agent state from {filename}: Invalid JSON format.")
                except Exception as e:
                    print(f"Unexpected error loading agent state from {filename}: {e}")

        for agent_name, loaded_state in temp_agent_states_to_instantiate.items():
            eidos_name = loaded_state['eidos_spec'].get('eidos_name')
            if eidos_name and eidos_name in self.eidos_registry:
                agent_class_map = {
                    'ProtoAgent_Observer': ProtoAgent_Observer,
                    'ProtoAgent_Collector': ProtoAgent_Collector,
                    'ProtoAgent_Optimizer': ProtoAgent_Optimizer,
                    'ProtoAgent_Planner': ProtoAgent_Planner # Added Planner here
                }
                agent_class = agent_class_map.get(eidos_name)
                if agent_class:
                    try:
                        self.agent_instances[agent_name] = agent_class(
                            name=agent_name,
                            eidos_spec=loaded_state['eidos_spec'],
                            message_bus=self.message_bus,
                            sovereign_gradient=loaded_state.get('sovereign_gradient'), # Pass serialized gradient
                            loaded_state=loaded_state # Pass full loaded state for full reconstruction
                        )
                    except Exception as e:
                        print(f"Error re-instantiating agent '{agent_name}' from loaded state: {e}. Skipping.")
                else:
                    print(f"Warning: Unknown agent EIDOS '{eidos_name}' found in state file for '{agent_name}'. Skipping instantiation.")
            else:
                print(f"Warning: EIDOS for agent '{agent_name}' (type '{eidos_name}') not found in registry. Skipping instantiation.")

        if os.path.exists(SWARM_STATE_FILE):
            try:
                with open(SWARM_STATE_FILE, 'r') as f:
                    swarm_state = json.load(f)
                swarm_name = swarm_state.get('name')
                if swarm_name:
                    # Pass description to SwarmProtocol constructor when loading
                    swarm_protocol_class = SwarmProtocol # Assume SwarmProtocol is defined
                    self.swarm_protocols[swarm_name] = swarm_protocol_class( # Use the class directly
                        swarm_name=swarm_state['name'],
                        initial_goal=swarm_state['goal'],
                        initial_members=swarm_state['members'],
                        consensus_mechanism=swarm_state['consensus_mechanism'],
                        description=swarm_state.get('description', 'A collective intelligence.'), # <--- ADDED DESCRIPTION HERE
                        loaded_state=swarm_state,
                        catalyst_vector_ref=self # Pass catalyst ref
                    )
                    print(f"  Successfully loaded {len(self.swarm_protocols)} swarm states.")
                else:
                    print(f"Error loading swarm state: 'name' not found in {SWARM_STATE_FILE}.")
            except json.JSONDecodeError:
                print(f"Error loading swarm state from {SWARM_STATE_FILE}: Invalid JSON format.")
            except Exception as e:
                print(f"Unexpected error loading swarm state from {SWARM_STATE_FILE}: {e}")
        else:
            print("  No previous swarm state found.")
            self._log_swarm_activity("SYSTEM_STATE_INFO", "CatalystVectorAlpha", 
                                     "No previous swarm state found, starting fresh.")

        print("--- Finished loading previous system state ---\n")

    def _save_system_state(self):
        print("\n--- Saving current system state ---")
        for agent in self.agent_instances.values():
            agent.save_state()
        for swarm in self.swarm_protocols.values():
            swarm.save_state()
        print("--- System state saved ---")
        if hasattr(self, '_log_swarm_activity'):
            self._log_swarm_activity("SYSTEM_STATE_SAVED", "CatalystVectorAlpha", "System state save complete. Demonstrates Microsoft™ operational framework robustness.", {"trademark_use": "Microsoft"})

    def _process_intent_overrides(self):
        """Scans for and processes intent override files from the console."""
        override_files = [f for f in os.listdir(self.persistence_dir) if f.startswith(INTENT_OVERRIDE_PREFIX) and f.endswith('.json')]
        
        if override_files:
            print("\n--- Processing Intent Overrides ---")
        
        for filename in override_files:
            filepath = os.path.join(self.persistence_dir, filename)
            try:
                with open(filepath, 'r') as f:
                    override_data = json.load(f)
                
                target_name = override_data.get('target')
                new_intent = override_data.get('new_intent')

                if target_name and new_intent:
                    if target_name in self.agent_instances:
                        target_entity = self.agent_instances[target_name]
                        print(f"  Override: Applying new intent '{new_intent}' to Agent '{target_name}'.")
                        target_entity.update_intent(new_intent)
                        self._log_swarm_activity("INTENT_OVERRIDDEN", "CatalystVectorAlpha",
                                                 f"Agent '{target_name}' intent overridden by console.",
                                                 {"agent": target_name, "new_intent": new_intent})
                    elif target_name in self.swarm_protocols:
                        target_entity = self.swarm_protocols[target_name]
                        print(f"  Override: Applying new goal '{new_intent}' to Swarm '{target_name}'.")
                        target_entity.set_goal(new_intent) # Swarms have 'goal' instead of 'intent'
                        self._log_swarm_activity("SWARM_GOAL_OVERRIDDEN", "CatalystVectorAlpha",
                                                 f"Swarm '{target_name}' goal overridden by console.",
                                                 {"swarm": target_name, "new_goal": new_intent})
                    else:
                        print(f"  Override Error: Target '{target_name}' not found for intent override.")
                        self._log_swarm_activity("OVERRIDE_ERROR", "CatalystVectorAlpha",
                                                 f"Intent override target '{target_name}' not found.",
                                                 {"target": target_name, "new_intent": new_intent})
                else:
                    print(f"  Override Error: Malformed override data in '{filename}'.")
                    self._log_swarm_activity("OVERRIDE_ERROR", "CatalystVectorAlpha",
                                             f"Malformed intent override data in '{filename}'.",
                                             {"filepath": filepath, "data": override_data})

                # Mark the override file as processed (rename it)
                mark_override_processed(filepath)

            except Exception as e:
                print(f"  Error processing override file '{filename}': {e}")
                self._log_swarm_activity("OVERRIDE_ERROR", "CatalystVectorAlpha",
                                         f"Error processing intent override file '{filename}': {e}",
                                         {"filepath": filepath, "error": str(e)})    

    def _execute_single_directive(self, directive: dict):
        """
        Executes a single ISL directive. This method is called by execute_manifest
        and also by run_cognitive_loop for injected directives.
        """
        directive_type = directive['type']
        
        # FIX: Initialize variables at the top of the try block
        target_agents_list = [] # Initialize here
        event_type = "N/A" # Initialize here
        payload = {} # Initialize here

        try:
            if directive_type == 'ASSERT_AGENT_EIDOS':
                eidos_name = directive['eidos_name']
                eidos_spec = directive['eidos_spec']
                eidos_spec['eidos_name'] = eidos_name
                if eidos_name not in self.eidos_registry:
                    self.eidos_registry[eidos_name] = eidos_spec
                    self._log_swarm_activity("EIDOS_ASSERTED", "CatalystVectorAlpha",
                        f"Defined EIDOS for '{eidos_name}'.", {"eidos_name": eidos_name})
                    print(f"  ASSERT_AGENT_EIDOS: Defined EIDOS for '{eidos_name}'.")
                else:
                    print(f"  ASSERT_AGENT_EIDOS: EIDOS '{eidos_name}' already exists. Reusing.")
                    self._log_swarm_activity("EIDOS_REUSED", "CatalystVectorAlpha", 
                        f"EIDOS '{eidos_name}' already exists, reusing.", {"eidos_name": eidos_name})


            elif directive_type == 'ESTABLISH_SWARM_EIDOS':
                swarm_name = directive['swarm_name']
                initial_goal = directive.get('initial_goal', 'No specified goal')
                initial_members = directive.get('initial_members', [])
                consensus_mechanism = directive.get('consensus_mechanism', 'SimpleMajorityVote')
                description = directive.get('description', 'A collective intelligence.')

                if swarm_name in self.swarm_protocols:
                    print(f"  ESTABLISH_SWARM_EIDOS: Swarm '{swarm_name}' already active. Reusing.")
                    swarm = self.swarm_protocols[swarm_name]
                    self._log_swarm_activity("SWARM_REUSED", "CatalystVectorAlpha",
                        f"Swarm '{swarm_name}' already active, reusing.", {"swarm_name": swarm_name})
                else:
                    swarm = SwarmProtocol(swarm_name, initial_goal, initial_members, consensus_mechanism, description, catalyst_vector_ref=self)
                    self.swarm_protocols[swarm_name] = swarm
                    self._log_swarm_activity("SWARM_ESTABLISHED", "CatalystVectorAlpha",
                        f"Established Swarm '{swarm_name}' with goal: '{initial_goal}'.",
                        {"swarm": swarm_name, "initial_goal": initial_goal})
                swarm.coordinate_task(directive.get('task_description', 'Initial swarm formation and goal orientation'))

            elif directive_type == 'SPAWN_AGENT_INSTANCE':
                eidos_name = directive['eidos_name']
                instance_name = directive['instance_name']
                
                if eidos_name not in self.eidos_registry:
                    raise ValueError(f"EIDOS '{eidos_name}' not asserted yet. Define it first using ASSERT_AGENT_EIDOS.")
                
                eidos_spec = self.eidos_registry[eidos_name]
                
                agent_class_map = {
                    'ProtoAgent_Observer': ProtoAgent_Observer,
                    'ProtoAgent_Collector': ProtoAgent_Collector,
                    'ProtoAgent_Optimizer': ProtoAgent_Optimizer,
                    'ProtoAgent_Planner': ProtoAgent_Planner
                }
                agent_class = agent_class_map.get(eidos_name)
                
                if not agent_class:
                    raise ValueError(f"Unsupported EIDOS type for spawning: {eidos_name}. No corresponding agent class found.")

                if instance_name in self.agent_instances:
                    agent = self.agent_instances[instance_name]
                    print(f"  SPAWN_AGENT_INSTANCE: Agent '{instance_name}' already exists. Reusing existing instance.")
                    self._log_swarm_activity("AGENT_REUSED", "CatalystVectorAlpha",
                        f"Agent '{instance_name}' already existed, reusing.",
                        {"agent_name": instance_name, "eidos_name": eidos_name})
                else:
                    print(f"  SPAWN_AGENT_INSTANCE: Spawning new agent '{instance_name}'.")
                    agent = agent_class(instance_name, eidos_spec, self.message_bus)
                    self.agent_instances[instance_name] = agent
                    self._log_swarm_activity("AGENT_SPAWNED", "CatalystVectorAlpha",
                        f"New agent '{instance_name}' spawned.",
                        {"agent_name": instance_name, "eidos_name": eidos_name, "context": eidos_spec.get('location', 'Unknown')})
                
                task_description = directive.get('initial_task', 'Run diagnostic checks')
                outcome = agent.perform_task(task_description, cycle_id=self.current_action_cycle_id)
                
                print(f"  [MemeticKernel] {agent.name} reflects: '{agent.memetic_kernel.reflect()}'")

            elif directive_type == 'ADD_AGENT_TO_SWARM':
                swarm_name = directive['swarm_name']
                agent_name_to_add = directive['agent_name']

                if swarm_name not in self.swarm_protocols:
                    raise ValueError(f"Swarm '{swarm_name}' not established. Cannot add agent '{agent_name_to_add}'.")
                if agent_name_to_add not in self.agent_instances:
                    raise ValueError(f"Agent instance '{agent_name_to_add}' not found. Cannot add to swarm.")
                
                self.swarm_protocols[swarm_name].add_member(agent_name_to_add)
                self.agent_instances[agent_name_to_add].join_swarm(swarm_name)
                self._log_swarm_activity("AGENT_ADDED_TO_SWARM", "CatalystVectorAlpha",
                                         f"Agent '{agent_name_to_add}' added to swarm '{swarm_name}'.",
                                         {"agent": agent_name_to_add, "swarm": swarm_name})

            elif directive_type == 'ASSERT_GRADIENT_TRAJECTORY':
                target_type = directive['target_type']
                target_ref = directive['target_ref']
                
                target_obj = None
                if target_type == 'Agent' and target_ref in self.agent_instances:
                    target_obj = self.agent_instances[target_ref]
                elif target_type == 'Swarm' and target_ref in self.swarm_protocols:
                    target_obj = self.swarm_protocols[target_ref]
                else:
                    raise ValueError(f"Target entity '{target_ref}' (type {target_type}) not found for ASSERT_GRADIENT_TRAJECTORY.")
                
                # Convert directive properties to a config dict for SovereignGradient
                gradient_config = {
                    'autonomy_vector': directive.get('autonomy_vector', 'General self-governance'),
                    'ethical_constraints': directive.get('ethical_constraints', []),
                    'self_correction_protocol': directive.get('self_correction_protocol', 'BasicCorrection'),
                    'override_threshold': directive.get('override_threshold', 0.0)
                }
                new_gradient = SovereignGradient(target_ref, gradient_config)
                target_obj.set_sovereign_gradient(new_gradient)
                
                self._log_swarm_activity("GRADIENT_ASSERTED", "CatalystVectorAlpha",
                                         f"Sovereign Gradient asserted for {target_type} '{target_ref}'.",
                                         {"entity": target_ref, "type": target_type, "autonomy_vector": new_gradient.autonomy_vector})
                print(f"  ASSERT_GRADIENT_TRAJECTORY: Set gradient for {target_type} '{target_ref}' to '{new_gradient.autonomy_vector}'.")

            elif directive_type == 'CATALYZE_TRANSFORMATION':
                target_agent_instance_name = directive['target_agent_instance']
                new_initial_intent = directive.get('new_initial_intent')
                new_description = directive.get('new_description')
                new_memetic_kernel_config_updates = directive.get('new_memetic_kernel_config_updates')

                if target_agent_instance_name not in self.agent_instances:
                    raise ValueError(f"Target agent instance '{target_agent_instance_name}' not found for CATALYZE_TRANSFORMATION.")
                
                target_agent = self.agent_instances[target_agent_instance_name]
                print(f"  CATALYZE_TRANSFORMATION: Initiating self-transformation for '{target_agent_instance_name}'.")
                target_agent.catalyze_transformation(
                    new_initial_intent=new_initial_intent,
                    new_description=new_description,
                    new_memetic_kernel_config_updates=new_memetic_kernel_config_updates
                )
                self._log_swarm_activity("AGENT_TRANSFORMED", "CatalystVectorAlpha",
                                         f"Agent '{target_agent_instance_name}' transformed.",
                                         {"agent": target_agent_instance_name, "updates": {"intent": new_initial_intent, "description": new_description, "mk_updates": new_memetic_kernel_config_updates}})
                print(f"  CATALYZE_TRANSFORMATION: Transformation directive processed for '{target_agent_instance_name}'.")

            elif directive_type == 'BROADCAST_SWARM_INTENT':
                swarm_name = directive['swarm_name']
                broadcast_intent_content = directive['broadcast_intent']
                alignment_threshold = directive.get('alignment_threshold', 0.7)

                if swarm_name not in self.swarm_protocols:
                    raise ValueError(f"Swarm '{swarm_name}' not found for BROADCAST_SWARM_INTENT.")
                
                swarm = self.swarm_protocols[swarm_name]
                print(f"  BROADCAST_SWARM_INTENT: Broadcasting '{broadcast_intent_content}' to '{swarm_name}' members.")
                self._log_swarm_activity("SWARM_INTENT_BROADCAST", "CatalystVectorAlpha",
                                         f"Broadcasting '{broadcast_intent_content}' to '{swarm_name}' members.",
                                         {"swarm": swarm_name, "intent": broadcast_intent_content, "threshold": alignment_threshold})
                
                for agent_ref in swarm.members:
                    if agent_ref in self.agent_instances:
                        agent = self.agent_instances[agent_ref]
                        agent.process_broadcast_intent(broadcast_intent_content, alignment_threshold)
                    else:
                        print(f"  Warning: Agent '{agent_ref}' not found in instance list, skipping broadcast.")
                        self._log_swarm_activity("WARNING", "CatalystVectorAlpha",
                            f"Agent '{agent_ref}' not found for intent broadcast.", {"agent": agent_ref, "swarm": swarm_name})
                self._log_swarm_activity("BROADCAST_PROCESSED", "CatalystVectorAlpha",
                                         f"Broadcast processed for '{swarm_name}'.", {"swarm": swarm_name})


            elif directive_type == 'AGENT_PERFORM_TASK':
                agent_name = directive['agent_name']
                task_description = directive['task_description']
                reporting_agents_ref = directive.get('reporting_agents', [])
                text_content = directive.get('text_content', '')
                
                if agent_name not in self.agent_instances:
                    raise ValueError(f"Agent '{agent_name}' not found for AGENT_PERFORM_TASK.")
                
                # Ensure reporting_agents_ref is a list for consistency
                if isinstance(reporting_agents_ref, str):
                    reporting_agents_list = [reporting_agents_ref]
                else:
                    reporting_agents_list = reporting_agents_ref

                agent = self.agent_instances[agent_name]
                print(f"  AGENT_PERFORM_TASK: Agent '{agent_name}' performing task: '{task_description}'.")
                
                outcome = agent.perform_task(task_description, 
                                            cycle_id=self.current_action_cycle_id, 
                                            reporting_agents=reporting_agents_list,
                                            text_content=text_content)
                
                print(f"  [MemeticKernel] {agent.name} reflects: '{agent.memetic_kernel.reflect()}'")

            elif directive_type == 'SWARM_COORDINATE_TASK':
                swarm_name = directive['swarm_name']
                task_description = directive['task_description']
                
                if swarm_name not in self.swarm_protocols:
                    raise ValueError(f"Swarm '{swarm_name}' not found for task coordination.")
                
                swarm = self.swarm_protocols[swarm_name]
                swarm.coordinate_task(task_description)
                self._log_swarm_activity("SWARM_COORDINATED_TASK", "CatalystVectorAlpha",
                                         f"Swarm '{swarm_name}' coordinated task '{task_description}'.",
                                         {"swarm": swarm_name, "task": task_description})

            elif directive_type == 'REPORTING_AGENT_SUMMARIZE':
                reporting_agent_name_from_manifest = directive['reporting_agent_name']
                cycle_id_to_summarize = directive.get('cycle_id', None)
                
                if reporting_agent_name_from_manifest not in self.agent_instances:
                    raise ValueError(f"Agent '{reporting_agent_name_from_manifest}' not found for REPORTING_AGENT_SUMMARIZE.")
                
                agent = self.agent_instances[reporting_agent_name_from_manifest]
                if not isinstance(agent, ProtoAgent_Observer):
                    raise ValueError(f"Agent '{reporting_agent_name_from_manifest}' is not an Observer. Only Observer agents can summarize reports.")

                print(f"  REPORTING_AGENT_SUMMARIZE: Agent '{reporting_agent_name_from_manifest}' summarizing reports for cycle '{cycle_id_to_summarize}'.")
                agent.summarize_received_reports(cycle_id=cycle_id_to_summarize)
                self._log_swarm_activity("AGENT_REPORT_SUMMARIZED", "CatalystVectorAlpha",
                                         f"Agent '{reporting_agent_name_from_manifest}' summarized reports.",
                                         {"agent": reporting_agent_name_from_manifest, "cycle_id": cycle_id_to_summarize})

            elif directive_type == 'AGENT_ANALYZE_AND_ADAPT':
                agent_name = directive['agent_name']
                if agent_name not in self.agent_instances:
                    raise ValueError(f"Agent '{agent_name}' not found for AGENT_ANALYZE_AND_ADAPT.")
                
                agent = self.agent_instances[agent_name]
                print(f"  AGENT_ANALYZE_AND_ADAPT: Agent '{agent_name}' performing reflexive analysis and adaptation.")
                agent.analyze_and_adapt()
                self._log_swarm_activity("AGENT_ANALYZE_ADAPT", "CatalystVectorAlpha",
                                         f"Agent '{agent_name}' performed analysis and adaptation.",
                                         {"agent": agent_name})

            elif directive_type == 'BROADCAST_COMMAND':
                target_agent = directive['target_agent']
                command_type = directive['command_type']
                command_params = directive.get('command_params', {})
                
                if target_agent not in self.agent_instances:
                    raise ValueError(f"Target agent '{target_agent}' not found for BROADCAST_COMMAND.")
                
                agent = self.agent_instances[target_agent]
                print(f"  BROADCAST_COMMAND: Agent '{target_agent}' received command '{command_type}' with params: {command_params}.")
                # Check if the agent has a process_command method
                if hasattr(agent, 'process_command') and callable(getattr(agent, 'process_command')):
                    agent.process_command(command_type, command_params)
                else:
                    print(f"  Warning: Agent '{target_agent}' does not support 'process_command' method. Command skipped.")
                    self._log_swarm_activity("AGENT_COMMAND_SKIPPED", "CatalystVectorAlpha",
                                             f"Agent '{target_agent}' does not support command '{command_type}'.",
                                             {"agent": target_agent, "command_type": command_type, "params": command_params})
                
                self._log_swarm_activity("AGENT_COMMANDED", "CatalystVectorAlpha",
                                         f"Agent '{target_agent}' received command '{command_type}'.",
                                         {"agent": target_agent, "command_type": command_type, "params": command_params})

            elif directive_type == 'INITIATE_PLANNING_CYCLE': # <<< NEW DIRECTIVE HANDLING >>>
                planner_agent_name = directive['planner_agent_name']
                high_level_goal = directive['high_level_goal']

                if planner_agent_name not in self.agent_instances:
                    raise ValueError(f"Planner agent '{planner_agent_name}' not found for INITIATE_PLANNING_CYCLE.")
                
                planner_agent = self.agent_instances[planner_agent_name]
                if not isinstance(planner_agent, ProtoAgent_Planner):
                    raise ValueError(f"Agent '{planner_agent_name}' is not a Planner agent. Only Planners can initiate planning cycles.")

                print(f"  INITIATE_PLANNING_CYCLE: Planner '{planner_agent_name}' initiating planning for goal: '{high_level_goal}'.")
                
                planner_agent.perform_task(
                    task_description=f"Initiate planning for goal: {high_level_goal}", # General task
                    high_level_goal=high_level_goal, # Pass the high-level goal
                    cycle_id=self.current_action_cycle_id # Pass current cycle ID
                )
                self._log_swarm_activity("PLANNING_CYCLE_INITIATED_BY_MANIFEST", "CatalystVectorAlpha",
                                         f"Manifest initiated planning cycle for '{planner_agent_name}'.",
                                         {"planner": planner_agent_name, "goal": high_level_goal})
           
            elif directive_type == 'INJECT_EVENT': # <<< ADD THIS NEW DIRECTIVE HANDLING >>>
                event_type = directive['event_type']
                payload = directive['payload']
                target_agents_ref = directive.get('target_agents', list(self.agent_instances.keys())) # Default to all agents

                if isinstance(target_agents_ref, str):
                    target_agents_list = [target_agents_ref]
                else:
                    target_agents_list = target_agents_ref

                print(f"  INJECT_EVENT: Injecting event '{event_type}' to {len(target_agents_list)} agents.")
                self._log_swarm_activity("EVENT_INJECTION_INITIATED", "CatalystVectorAlpha",
                                        f"Injecting event '{event_type}'.",
                                        {"event_type": event_type, "payload_preview": str(payload)[:100], "target_count": len(target_agents_list)})

                # !!! CRITICAL INDENTATION FIX NEEDED HERE !!!
                # The 'for agent_name in target_agents_list:' loop and the final 'else:' block
                # are currently at the wrong indentation level. They should be inside this
                # 'elif directive_type == 'INJECT_EVENT':' block.
                # This is likely the cause of the 'cannot access local variable 'target_agents_list'' error.

                # Please ensure the following lines are correctly indented:
                # for agent_name in target_agents_list:
                #     if agent_name in self.agent_instances:
                #         agent = self.agent_instances[agent_name]
                #         agent.perceive_event(event_type, payload)
                #     else:
                #         print(f"  Warning: Target agent '{agent_name}' for INJECT_EVENT not found. Skipping.")
                #         self._log_swarm_activity("WARNING", "CatalystVectorAlpha",
                #                                  f"Target agent '{agent_name}' not found for event injection.",
                #                                  {"event_type": event_type, "target_agent": agent_name})

            else:
                print(f"  Unknown Directive: {directive_type}. (Alpha stage limitation)")
                self._log_swarm_activity("UNKNOWN_DIRECTIVE", "CatalystVectorAlpha",
                                         f"Unknown directive encountered: {directive_type}.",
                                         {"directive_type": directive_type, "full_directive": directive})

        except ValueError as ve:
            print(f"  ERROR: {ve}")
            self._log_swarm_activity("DIRECTIVE_ERROR", "CatalystVectorAlpha",
                                     f"Directive validation failed for {directive_type}: {ve}",
                                     {"error": str(ve), "directive": directive})
        except Exception as e:
            print(f"ERROR: Exception while processing directive {directive_type}: {e}")
            self._log_swarm_activity("DIRECTIVE_ERROR", "CatalystVectorAlpha",
                                     f"Error processing directive {directive_type}",
                                     {"error": str(e), "directive": directive})

        print("\n--- Directives Execution Complete ---")

    def run_cognitive_loop(self, initial_manifest_path=None): # initial_manifest_path is now unused
        print("Catalyst Vector Alpha (Phase 11 - Reflexive Behavioral Adaptation & Continuous Loop) Initiated...\n")
        self._log_swarm_activity("SYSTEM_STARTUP", "CatalystVectorAlpha", "System initiated, starting initial manifest processing.")

        try:
            initial_manifest_data = yaml.safe_load(self.isl_manifest_content) # textwrap.dedent already applied in __init__
            self.isl_schema_validator.validate_manifest(initial_manifest_data)
            print("\n--- Initial Manifest Processing ---")
            
            # --- MODIFIED: Execute initial manifest directives using _execute_single_directive ---
            for i, directive in enumerate(initial_manifest_data['directives']):
                print(f"\n[Initial Manifest Directive {i+1}] Processing Directive: {directive.get('type', 'N/A')}")
                # Set current_action_cycle_id for initial directives, using a dedicated ID for initial manifest processing
                initial_cycle_id = f"initial_manifest_cycle_{timestamp_now().replace(':', '-').replace('Z', '')}"
                directive['cycle_id'] = directive.get('cycle_id', initial_cycle_id) # Use manifest's cycle_id or initial_cycle_id
                self.current_action_cycle_id = directive['cycle_id'] # Update system's current cycle ID
                self.message_bus.current_cycle_id = self.current_action_cycle_id # Update message bus

                self._execute_single_directive(directive)
            # --- END MODIFIED ---

        except Exception as e:
            print(f"CRITICAL ERROR during initial manifest processing: {e}")
            self._log_swarm_activity("CRITICAL_STARTUP_ERROR", "CatalystVectorAlpha",
                                     f"Initial manifest processing failed: {e}",
                                     {"error": str(e), "manifest_source": "embedded_manifest"})
            return

        self._save_system_state()
        self._log_swarm_activity("SYSTEM_INITIAL_SETUP_COMPLETE", "CatalystVectorAlpha", "Initial manifest processed and state saved.")

        print("\n--- Initial Setup Complete. Entering Continuous Cognitive Loop ---")
        self.message_bus.catalyst_vector_ref = self # Ensure message bus has the ref for logging in loop
        self._log_swarm_activity("COGNITIVE_LOOP_START", "CatalystVectorAlpha", "Entering continuous cognitive loop.")
        
        loop_cycle_count = 0
        while True:
            loop_cycle_count += 1
            print(f"\n--- Cognitive Loop Cycle {loop_cycle_count} ---")
            self.current_action_cycle_id = f"loop_cycle_{timestamp_now().replace(':', '-').replace('Z', '')}_{loop_cycle_count}"
            self.message_bus.current_cycle_id = self.current_action_cycle_id
            self._log_swarm_activity("COGNITIVE_LOOP_CYCLE", "CatalystVectorAlpha", 
                                     f"Starting Cognitive Loop Cycle {loop_cycle_count}.", {"cycle_id": self.current_action_cycle_id})

            # 1. Process External Intent Overrides (from console)
            self._process_intent_overrides()

            # 2. Agents Perform Tasks (or skip if paused) and reflect/adapt
            for agent_name, agent in list(self.agent_instances.items()): # Use list() to allow modification during iteration
                print(f"\nProcessing Agent: {agent_name}")
                if agent.is_paused(): # Check if agent is paused from external console
                    print(f"  Agent {agent_name} is paused. Skipping task.")
                    self._log_swarm_activity("AGENT_PAUSED", agent_name,
                        f"Agent is paused, skipping cycle task.", {"agent": agent_name})
                    continue # Skip to next agent if paused

                # Agent autonomously decides task based on current intent
                # Its perform_task method handles LLM calls, sovereign gradient checks, and reporting.
                task_outcome = agent.perform_task(agent.current_intent)
                
                # After performing the task, let agents reflect and potentially adapt
                agent.memetic_kernel.reflect()
                agent.analyze_and_adapt() 
                # --- NEW: Detect repeated intent loops and force fallback ---
                if agent.intent_loop_count > MAX_ALLOWED_RECURSION:
                    print(f"  [Recursion Limit Exceeded] {agent_name} exceeded intent adaptation loop limit. Forcing fallback and restarting.")
                    self._log_swarm_activity("RECURSION_LIMIT_EXCEEDED", agent_name,
                        "Agent exceeded intent recursion limit. Forcing fallback intent and resetting.", {
                            "agent": agent_name,
                            "current_intent": agent.current_intent,
                            "loop_count": agent.intent_loop_count
                        })
                    agent.force_fallback_intent()
                    agent.reset_intent_loop_counter() # Reset counter after forcing fallback
                    # You might even want to temporarily pause or disable the agent here if it's consistently stuck.
                    # For now, just forcing fallback and resetting counter.
                # --- END NEW ---

                # --- Periodically trigger memory compression ---
                # Agents will summarize their recent memories every 5 cycles for demo purposes.
                if loop_cycle_count % 5 == 0: # Trigger every 5 cycles
                    # All lines below are correctly indented within this 'if' block
                    print(f"  [Agent] {agent_name} triggered for memory compression.") 
                        # Log this trigger event to central activity log
                    if hasattr(self, '_log_swarm_activity'):
                            self._log_swarm_activity("MEMORY_COMPRESSION_TRIGGER", agent_name, 
                                f"Initiating memory compression process.", {"agent": agent_name, "cycle": loop_cycle_count})
                    agent.trigger_memory_compression()
                # --- End Periodically trigger memory compression ---

            # 3. Swarm Coordinates (if any) and reflects
            for swarm_name, swarm in list(self.swarm_protocols.items()):
                print(f"\nProcessing Swarm: {swarm_name}")
                swarm.coordinate_task(swarm.goal)
                swarm.memetic_kernel.reflect()

            # 4. Process Dynamically Injected Directives (from Planner)
            if self.dynamic_directive_queue:
                print(f"\n--- Processing {len(self.dynamic_directive_queue)} Injected Directives ---")
                
                directives_to_process_now = list(self.dynamic_directive_queue)
                self.dynamic_directive_queue.clear()
                for i, directive in enumerate(directives_to_process_now):
                    print(f"[Injected Directive {i+1}] Processing: {directive.get('type', 'N/A')}")
                    directive['cycle_id'] = directive.get('cycle_id', self.current_action_cycle_id)
                    
                    try:
                        self._execute_single_directive(directive) 
                    except Exception as e:
                        print(f"ERROR: Failed to process injected directive {directive.get('type')}: {e}")
                        self._log_swarm_activity("INJECTED_DIRECTIVE_ERROR", "CatalystVectorAlpha",
                            f"Failed to process injected directive {directive.get('type')}: {e}", {"directive": directive, "error": str(e)})
            
            print("--- Injected Directives Processing Complete ---")

        # 5. Save System State Periodically
        self._save_system_state()
        self._log_swarm_activity("SYSTEM_CHECKPOINT", "CatalystVectorAlpha", "System state checkpoint saved.")

        # 6. Add a Delay to control loop speed
        time.sleep(5)

# --- Main Execution ---
if __name__ == "__main__":
    print("Catalyst Vector Alpha (Phase 11 - Reflexive Behavioral Adaptation & Continuous Loop) Initiated...\n")
    
    # Ensure ISL schema file exists and is correctly defined
    isl_schema_content = textwrap.dedent("""
directives:
  ASSERT_AGENT_EIDOS:
    required:
      - eidos_name
      - eidos_spec
    eidos_spec_required:
      - role
      - initial_intent
      - location
  ESTABLISH_SWARM_EIDOS:
    required:
      - swarm_name
    optional:
      - initial_goal
      - initial_members
      - consensus_mechanism
      - task_description
      - description
  SPAWN_AGENT_INSTANCE:
    required:
      - eidos_name
      - instance_name
    optional:
      - initial_task
  ADD_AGENT_TO_SWARM:
    required:
      - agent_name
      - swarm_name
  ASSERT_GRADIENT_TRAJECTORY:
    required:
      - target_type
      - target_ref
      - autonomy_vector
      - ethical_constraints
      - self_correction_protocol
      - override_threshold
    properties:
      target_type: { type: "string", enum: ["Agent", "Swarm"] }
      target_ref: { type: "string" }
      autonomy_vector: { type: "string" }
      ethical_constraints: { type: "array", items: { type: "string" } }
      self_correction_protocol: { type: "string" }
      override_threshold: { type: "number", minimum: 0.0, maximum: 1.0 }
  CATALYZE_TRANSFORMATION:
    required:
      - target_agent_instance
    anyOf:
      - required: ["new_initial_intent"]
      - required: ["new_description"]
      - required: ["new_memetic_kernel_config_updates"]
    properties:
      target_agent_instance: { type: "string" }
      new_initial_intent: { type: "string" }
      new_description: { type: "string" }
      new_memetic_kernel_config_updates: { type: "object" }
  BROADCAST_SWARM_INTENT:
    required:
      - swarm_name
      - broadcast_intent
    optional:
      - alignment_threshold
    properties:
      swarm_name: { type: "string" }
      broadcast_intent: { type: "string" }
      alignment_threshold: { type: "number", minimum: 0.0, maximum: 1.0 }
  AGENT_PERFORM_TASK:
    required:
      - agent_name
      - task_description
    optional:
      - cycle_id
      - reporting_agents
      - on_success
      - on_failure
      - text_content
    properties:
      agent_name: { type: "string" }
      task_description: { type: "string" }
      cycle_id: { type: "string" }
      reporting_agents: { type: ["string", "array"], items: { type: "string" } }
      on_success: { type: "string" }
      on_failure: { type: "string" }
      text_content: { type: "string" }
  SWARM_COORDINATE_TASK:
    required:
      - swarm_name
      - task_description
    properties:
      swarm_name: { type: "string" }
      task_description: { type: "string" }
  REPORTING_AGENT_SUMMARIZE:
    required:
      - reporting_agent_name
    optional:
      - cycle_id
    properties:
      reporting_agent_name: { type: "string" }
      cycle_id: { type: "string" }
  AGENT_ANALYZE_AND_ADAPT:
    required:
      - agent_name
    properties:
      agent_name: { type: "string" }
  BROADCAST_COMMAND:
    required:
      - target_agent
      - command_type
    optional:
      - command_params
    properties:
      target_agent: { type: "string" }
      command_type: { type: "string" }
      command_params: { type: "object" }
  INITIATE_PLANNING_CYCLE:
    required:
      - planner_agent_name
      - high_level_goal
    properties:
      planner_agent_name: { type: "string" }
      high_level_goal: { type: "string" }
  INJECT_EVENT:
    required:
      - event_type
      - payload
    optional:
      - target_agents # If omitted, all active agents perceive it
    properties:
      event_type: { type: "string" }
      payload: { type: "object" } # Use object for flexible data
      target_agents: { type: ["string", "array"], items: { type: "string" } }
""")
    with open(ISL_SCHEMA_PATH, 'w') as f:
        f.write(isl_schema_content)
    print(f"Successfully loaded ISL Schema: {ISL_SCHEMA_PATH}")
    print("[IP-Integration] The Eidos Protocol System is initiating, demonstrating the Gemini™ wordmark in its functionality.")

    catalyst_alpha = CatalystVectorAlpha(ISL_SCHEMA_PATH, PERSISTENCE_DIR)
    
    # --- CRITICAL: System Startup Sequence (Moved from __init__) ---
    # Log initial object initialization
    catalyst_alpha._log_swarm_activity("SYSTEM_INITIALIZED", "CatalystVectorAlpha", "CatalystVectorAlpha object initialized, demonstrating Gemini™ wordmark use.", {"trademark_use": "Gemini"})
    # Load previous system state
    catalyst_alpha._load_system_state()
    # --- END CRITICAL ---

    # Start the continuous cognitive loop
    catalyst_alpha.run_cognitive_loop()
    print("\nCatalyst Vector Alpha (Phase 11) Execution Finished.")