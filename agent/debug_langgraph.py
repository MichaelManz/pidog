#!/usr/bin/env python3
"""
LangGraph Debugging Script

This script helps debug LangGraph execution by monitoring:
1. LLM API calls and responses
2. Tool execution timing
3. Threading and action state
4. Memory usage
5. Network connectivity

Usage:
    python debug_langgraph.py
"""

import time
import threading
from langchain_openai import ChatOpenAI
import psutil
import os
import sys
import logging
from datetime import datetime
import signal
import json
from logging_config import setup_logging, get_logger

from keys import OPENAI_API_KEY, OPENAI_ASSISTANT_ID

# Setup logging
setup_logging(log_level=logging.DEBUG, log_file='debug_langgraph.log')
logger = get_logger(__name__)

class LangGraphDebugger:
    """Comprehensive debugging tool for LangGraph execution"""
    
    def __init__(self):
        self.start_time = time.time()
        self.monitoring = False
        self.stats = {
            'api_calls': 0,
            'tool_calls': 0,
            'errors': 0,
            'memory_usage': [],
            'timing': []
        }
        
    def start_monitoring(self):
        """Start background monitoring"""
        self.monitoring = True
        
        # Start resource monitoring thread
        monitor_thread = threading.Thread(target=self._monitor_resources)
        monitor_thread.daemon = True
        monitor_thread.start()
        
        # Start action state monitoring
        action_monitor_thread = threading.Thread(target=self._monitor_action_state)
        action_monitor_thread.daemon = True
        action_monitor_thread.start()
        
        logger.info("LangGraph monitoring started")
    
    def stop_monitoring(self):
        """Stop monitoring"""
        self.monitoring = False
        logger.info("LangGraph monitoring stopped")
        self._print_summary()
    
    def _monitor_resources(self):
        """Monitor system resources"""
        while self.monitoring:
            try:
                # Get memory usage
                process = psutil.Process(os.getpid())
                memory_mb = process.memory_info().rss / 1024 / 1024
                cpu_percent = process.cpu_percent()
                
                self.stats['memory_usage'].append({
                    'timestamp': time.time(),
                    'memory_mb': memory_mb,
                    'cpu_percent': cpu_percent
                })
                
                # Log if memory usage is high
                if memory_mb > 500:  # 500MB threshold
                    logger.warning(f"High memory usage: {memory_mb:.1f}MB")
                
                # Check for hanging threads
                thread_count = threading.active_count()
                if thread_count > 10:
                    logger.warning(f"High thread count: {thread_count}")
                    
            except Exception as e:
                logger.error(f"Error monitoring resources: {e}")
            
            time.sleep(5)  # Check every 5 seconds
    
    def _monitor_action_state(self):
        """Monitor action flow state"""
        try:
            from action_flow_tool import action_state, action_lock
            
            last_state = None
            state_start_time = time.time()
            
            while self.monitoring:
                with action_lock:
                    current_state = action_state
                
                if current_state != last_state:
                    if last_state is not None:
                        duration = time.time() - state_start_time
                        logger.info(f"Action state changed: {last_state} -> {current_state} (duration: {duration:.2f}s)")
                    
                    last_state = current_state
                    state_start_time = time.time()
                else:
                    # Check if we've been in the same state too long
                    duration = time.time() - state_start_time
                    if duration > 30:  # 30 second threshold
                        logger.warning(f"Action state stuck in '{current_state}' for {duration:.2f}s")
                
                time.sleep(1)
                
        except ImportError:
            logger.warning("Cannot monitor action state - action_flow_tool not available")
        except Exception as e:
            logger.error(f"Error monitoring action state: {e}")
    
    def log_api_call(self, api_type, duration, success=True):
        """Log API call"""
        self.stats['api_calls'] += 1
        self.stats['timing'].append({
            'type': 'api_call',
            'api_type': api_type,
            'duration': duration,
            'success': success,
            'timestamp': time.time()
        })
        
        if not success:
            self.stats['errors'] += 1
            logger.error(f"API call failed: {api_type} took {duration:.2f}s")
        else:
            logger.info(f"API call: {api_type} took {duration:.2f}s")
    
    def log_tool_call(self, tool_name, duration, success=True):
        """Log tool call"""
        self.stats['tool_calls'] += 1
        self.stats['timing'].append({
            'type': 'tool_call',
            'tool_name': tool_name,
            'duration': duration,
            'success': success,
            'timestamp': time.time()
        })
        
        if not success:
            self.stats['errors'] += 1
            logger.error(f"Tool call failed: {tool_name} took {duration:.2f}s")
        else:
            logger.info(f"Tool call: {tool_name} took {duration:.2f}s")
      
    def _print_summary(self):
        """Print debugging summary"""
        total_time = time.time() - self.start_time
        
        print("\n" + "="*60)
        print("LANGGRAPH DEBUGGING SUMMARY")
        print("="*60)
        print(f"Total execution time: {total_time:.2f}s")
        print(f"API calls: {self.stats['api_calls']}")
        print(f"Tool calls: {self.stats['tool_calls']}")
        print(f"Errors: {self.stats['errors']}")
        
        if self.stats['memory_usage']:
            max_memory = max(m['memory_mb'] for m in self.stats['memory_usage'])
            print(f"Peak memory usage: {max_memory:.1f}MB")
        
        # Show slowest operations
        if self.stats['timing']:
            slowest = sorted(self.stats['timing'], key=lambda x: x['duration'], reverse=True)[:5]
            print("\nSlowest operations:")
            for op in slowest:
                print(f"  {op['type']}: {op.get('tool_name', op.get('api_type', 'unknown'))} - {op['duration']:.2f}s")
        
        print("="*60)

# Global debugger instance
debugger = LangGraphDebugger()

def signal_handler(signum, frame):
    """Handle CTRL+C gracefully"""
    logger.info("Received interrupt signal, stopping monitoring...")
    debugger.stop_monitoring()
    sys.exit(0)

def main():
    """Main debugging function"""
    signal.signal(signal.SIGINT, signal_handler)
    
    print("LangGraph Debugger Started")
    print("This will monitor your LangGraph execution for issues.")
    print("Press Ctrl+C to stop and see summary.")
    
    debugger.start_monitoring()
   
    # Import and run your main script
    try:
        print("\nStarting main application...")
        from model_helper import ModelHelper
        from action_flow_tool import ActionFlowTool, set_camera_handler
        from camera_handler import CameraHandler
        from action_flow import ActionFlow
        from keys import GEMINI_API_KEY
        from langchain_google_genai import ChatGoogleGenerativeAI
        
        # Initialize components
        camera_handler = CameraHandler()
        set_camera_handler(camera_handler)
        camera_handler.start()
        
        action_flow = ActionFlow(None)  # No physical dog for debugging
        tools = [ActionFlowTool(action_flow, action) for action in action_flow.OPERATIONS]
        
        llm = ChatGoogleGenerativeAI(
             model="gemini-2.5-pro",
             temperature=0,
             max_tokens=None,
             timeout=None,
             max_retries=2,
             google_api_key=GEMINI_API_KEY
         )
        #llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY, model="gpt-4o")
        
        model_helper = ModelHelper(llm, tools)
        
        # Test execution
        img_path = camera_handler.capture_image()
        response = model_helper.dialogue_with_img(
            "Invoke the action 'think' and describe what you see in the tool call result's output image",
            img_path,
            timeout=120  # 2 minute timeout
        )
        
        print(f"\nResponse received: {response}")
        
    except Exception as e:
        logger.error(f"Main application error: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
    
    finally:
        debugger.stop_monitoring()

if __name__ == "__main__":
    main() 