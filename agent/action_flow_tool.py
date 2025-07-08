from langchain.tools import BaseTool
from action_flow import ActionFlow
from typing import Union
import re
import time, random, threading
from logging_config import setup_logging, get_logger, truncate_at_base64

# Configure logging
setup_logging()
logger = get_logger(__name__)

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
                    _actions = actions_to_be_done.copy()
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

    def _run(self, query: str = "") -> list:
        """Use the tool."""
        logger.info(f"=== Tool Execution Started ===")
        logger.info(f"Tool: {self.name}")
        logger.info(f"Action: {self.action}")
        logger.info(f"Query: {query}")
        
        start_time = time.time()
        
        try:
            # Change action state to 'think'
            logger.info("Setting action state to 'think'")
            set_action_state('think')
            
            # Execute the action
            logger.info(f"Executing action: {self.action}")
            result = self.action_flow.run(self.action)
            
            execution_time = time.time() - start_time
            logger.info(f"Action completed in {execution_time:.2f}s")
            
            # Capture image if camera is available
            if camera_handler:
                logger.info("Capturing image after action...")
                try:
                    image_data = camera_handler.get_image_base64()
                    logger.info(f"Image captured successfully")

                    msg = f"Action '{self.action}' completed successfully"
                    # Get image description or status
                    result_msg = [
                        {"type": "text", "text": f"{msg}"},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{image_data}"},
                        }
                    ]
                    
                except Exception as e:
                    logger.error(f"Error capturing image: {e}")
                    result_msg = [f"Action '{self.action}' completed successfully, but failed to capture image: {e}"]
            else:
                logger.info("No camera handler available")
                result_msg = [f"Action '{self.action}' completed successfully (no camera available)"]
            
            logger.info(truncate_at_base64(f"Tool result: {result_msg}"))
            
            # Reset action state
            logger.info("Setting action state to 'standby'")
            set_action_state('standby')
            
            total_time = time.time() - start_time
            logger.info(f"=== Tool Execution Completed in {total_time:.2f}s ===")
            
            return result_msg
            
        except Exception as e:
            error_time = time.time() - start_time
            logger.error(f"=== Tool Execution Failed after {error_time:.2f}s ===")
            logger.error(f"Error in tool {self.name}: {e}")
            logger.error(f"Error type: {type(e).__name__}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            
            # Reset action state on error
            set_action_state('standby')
            
            return [f"Error executing action '{self.action}': {e}"]