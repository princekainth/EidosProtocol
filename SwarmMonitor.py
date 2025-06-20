import time
import json
import os
import sys

# --- Constants and Configurations ---
PERSISTENCE_DIR = 'persistence_data' # Ensure this matches your main script
SWARM_ACTIVITY_LOG = os.path.join(PERSISTENCE_DIR, 'swarm_activity.jsonl')

# Ensure persistence directory exists (useful if running monitor first)
os.makedirs(PERSISTENCE_DIR, exist_ok=True)

def tail_log(filepath):
    print(f"--- Swarm Monitor (Tailing: {filepath}) ---")
    
    # Wait for the log file to appear if it doesn't exist yet
    if not os.path.exists(filepath):
        print(f"Waiting for log file to appear: {filepath}")
        while not os.path.exists(filepath):
            time.sleep(1)
        print("Log file found. Starting to tail...")

    with open(filepath, 'r') as f:
        # Seek to the end of the file initially to only read new lines
        f.seek(0, os.SEEK_END) 
        
        while True:
            line = f.readline()
            if not line:
                # No new lines, wait a bit before checking again
                time.sleep(0.5) 
                continue
            
            try:
                log_entry = json.loads(line.strip())
                # Customize this print statement for your desired cockpit view
                timestamp = log_entry.get('timestamp', 'N/A')
                event_type = log_entry.get('event_type', 'N/A')
                source = log_entry.get('source', 'N/A')
                description = log_entry.get('description', 'N/A')
                details = log_entry.get('details', {})

                # Basic pretty print of details if they exist
                details_str = json.dumps(details, indent=2) if details else "{}"

                print(f"[{timestamp}] <{event_type}> Source: {source}\n  Description: {description}\n  Details: {details_str}\n---")
            except json.JSONDecodeError:
                print(f"Error decoding JSON from log line: {line.strip()}")
            except KeyError as e:
                print(f"Missing expected key '{e}' in log entry: {line.strip()}")

if __name__ == "__main__":
    tail_log(SWARM_ACTIVITY_LOG)