from langchain.tools import BaseTool
from action_flow import ActionFlow
from typing import Union
import re
import time, random, threading

# Module-level thread management
action_thread = None
action_state = 'standby'  # 'standby', 'think', 'actions', 'actions_done'
actions_to_be_done = []
action_lock = threading.Lock()

# Global camera handler (will be set from main module)
camera_handler = None

def set_action_state(state: str):
    """Set the action state thread-safely."""
    global action_state
    with action_lock:
        action_state = state

def set_camera_handler(handler):
    """Set the global camera handler instance."""
    global camera_handler
    camera_handler = handler

def start_action_handler(action_flow: ActionFlow):
    """Start the background action handler thread once per process."""
    global action_thread
    
    if action_thread:
        return
    
    def action_handler():
        global action_state, actions_to_be_done
        
        standby_actions = ['waiting', 'feet_left_right']
        standby_weights = [1, 0.3]
        
        action_interval = 5  # seconds
        last_action_time = time.time()
        
        while True:
            with action_lock:
                _state = action_state
            if _state == 'standby':
                if time.time() - last_action_time > action_interval:
                    choice = random.choices(standby_actions, standby_weights)[0]
                    try:
                        action_flow.run(choice)
                    except Exception as e:
                        print(f'standby action error: {e}')
                    last_action_time = time.time()
                    action_interval = random.randint(2, 6)
            elif _state == 'think':
                # action_flow.run('think')
                # last_action_time = time.time()
                pass
            elif _state == 'actions':
                with action_lock:
                    _actions = actions_to_be_done
                for _action in _actions:
                    try:
                        action_flow.run(_action)
                    except Exception as e:
                        print(f'action error: {e}')
                    time.sleep(0.5)
                
                with action_lock:
                    action_state = 'actions_done'
                last_action_time = time.time()
            
            time.sleep(0.01)
    
    action_thread = threading.Thread(target=action_handler)
    action_thread.daemon = True
    action_thread.start()
    
    set_action_state('standby')


class ActionFlowTool(BaseTool):
    """Wrap a single ActionFlow operation as a LangChain tool."""

    name: str = "ActionFlowTool"
    description: str = None
    action_flow: ActionFlow = None
    action: str = None

    def __init__(self, action_flow: ActionFlow, action: str):
        super().__init__()
        self.action_flow = action_flow
        self.action = action  # original action name (may contain spaces)
        # Sanitize name for model/tool declaration: letters, digits, _ . - only
        safe_name = re.sub(r"[^A-Za-z0-9_.-]", "_", action)
        if not re.match(r"^[A-Za-z_]", safe_name):
            safe_name = f"_{safe_name}"
        self.name = safe_name[:64]
        self.description = f"Execute the dog action: {action} (tool id: {self.name})"
        
        # Start the background action handler thread
        start_action_handler(self.action_flow)

    def _run(self, **kwargs) -> tuple[str, str]:
        """Execute the underlying dog action and return a confirmation string."""
        global actions_to_be_done, action_lock, camera_handler
        with action_lock:
            actions_to_be_done = [self.action]
        set_action_state('actions')
        
        # Capture image using camera handler if available
        msg = f"Executed dog action {self.action}"
        if camera_handler and camera_handler.is_started:
            try:
                img_path = camera_handler.capture_image()
                image_data = camera_handler.get_image_base64(img_path)
                return msg, {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{image_data}"},
                    }
            except Exception as e:
                print(f"Camera capture error: {e}")
                return f"{msg} (camera capture failed)"
        else:
            return f"{msg} (no camera available)"