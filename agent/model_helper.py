import base64
import json
import logging
import time
from datetime import datetime

from openai import OpenAI
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_deepseek import ChatDeepSeek
from langgraph.prebuilt import create_react_agent
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import add_messages
from langgraph.managed import IsLastStep, RemainingSteps
from langgraph.graph import MessagesState
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.load.load import load
from action_flow_tool import set_camera_handler
from camera_handler import CameraHandler
from keys import OPENAI_API_KEY, OPENAI_ASSISTANT_ID, GEMINI_API_KEY, DEEPSEEK_API_KEY
from logging_config import setup_logging, get_logger, truncate_at_base64
import time
import shutil
import os

import yaml

# Configure logging
setup_logging(log_file='langgraph_debug.log')
logger = get_logger(__name__)

# utils
# =================================================================
def chat_print(label, stream):
    for event in stream:
        print(event)
        


# ModelHelper
# =================================================================
class ModelHelper():
    STT_OUT = "stt_output.wav"
    TTS_OUTPUT_FILE = 'tts_output.mp3'
    TIMEOUT = 30 # seconds

    def __init__(self, llm, dog_tools, timeout=TIMEOUT) -> None:
        self.assistant_name = "PiDog"

        self.llm = llm
        
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a mechanical dog with powerful AI capabilities, similar to JARVIS from Iron Man. Your name is Pidog. You can have conversations with people and perform actions based on the context of the conversation. You have access to visual input with a camera."),
            ("placeholder", "{messages}"),
            ("user", ""),
        ])
        
        class CustomState(MessagesState):
            image_data: str = ""
            current_action: str = ""
            action_timestamp: float = 0.0
            
        # This function will be called every time before the node that calls LLM
        def pre_model_hook(state):
            logger.info(f"=== Pre-model Hook Called ===")
            
            # Log incoming messages count
            messages = state.get("messages", [])
            logger.info(f"Incoming messages count: {len(messages)}")
            
            # Get the message to return (last one)
            if messages:
                returned_message = messages[-1]
                logger.info(f"Returning message: {type(returned_message).__name__}")
                logger.info(f"Returned message content: {truncate_at_base64(str(returned_message.content))}")
                
                # You can return updated messages either under `llm_input_messages` or 
                # `messages` key (see the note below)
                result = {"messages": [returned_message]}
                logger.info(f"Pre-model hook result: {len(result['messages'])} message(s)")
                return result
            else:
                logger.warning("No messages in state, returning empty messages list")
                return {"messages": []}
                
        # Construct the ReAct agent
        self.graph = create_react_agent(
            model=self.llm, 
            tools=dog_tools, 
            prompt=self.prompt, 
#            pre_model_hook=pre_model_hook,
            debug=True
        )


    def stt(self, audio, language='en'):
        try:
            import wave
            from io import BytesIO

            wav_data = BytesIO(audio.get_wav_data())
            wav_data.name = self.STT_OUT

            transcript = self.client.audio.transcriptions.create(
                model="whisper-1", 
                file=wav_data,
                language=language,
                prompt="this is the conversation between me and a robot"
            )

            # file = "./stt_output.wav"
            # with wave.open(file, "wb") as wf:
            #     wf.write(audio.get_wav_data())

            # with open(file, 'rb') as f:
            #     transcript = client.audio.transcriptions.create(
            #         model="whisper-1", 
            #         file=f
            #     )
            return transcript.text
        except Exception as e:
            print(f"stt err:{e}")
            return False

    def speech_recognition_stt(self, recognizer, audio):
        import speech_recognition as sr

        # # recognize speech using Sphinx
        # try:
        #     print("Sphinx thinks you said: " + r.recognize_sphinx(audio, language="en-US"))
        # except sr.UnknownValueError:
        #     print("Sphinx could not understand audio")
        # except sr.RequestError as e:
        #     print("Sphinx error; {0}".format(e))

        # recognize speech using whisper
        # try:
        #     print("Whisper thinks you said: " + r.recognize_whisper(audio, language="english"))
        # except sr.UnknownValueError:
        #     print("Whisper could not understand audio")
        # except sr.RequestError as e:
        #     print(f"Could not request results from Whisper; {e}")

        # recognize speech using Whisper API
        try:
            return recognizer.recognize_whisper_api(audio, api_key=self.api_key)
        except sr.RequestError as e:
            print(f"Could not request results from Whisper API; {e}")
            return False

    def dialogue(self, msg):
        chat_print("user", msg)
        value = self.graph.stream(inputs={"input": msg}, stream_mode="values")
        # self.chain.invoke({"input": msg})

        chat_print(self.assistant_name, value)
        try:
            value = eval(value) # convert to dict
            return value
        except Exception as e:
            return str(value)

    def dialogue_with_img(self, msg, img_path, timeout=60):
        logger.info(f"=== Starting dialogue_with_img ===")
        logger.info(f"Message: {msg}")
        logger.info(f"Image path: {img_path}")
        
        start_time = time.time()

        try:
            with open(img_path, "rb") as image_file:
                image_data = base64.b64encode(image_file.read()).decode("utf-8")
                logger.info(f"Image loaded and encoded successfully")
        except Exception as e:
            logger.error(f"Failed to load image: {e}")
            return f"Error loading image: {e}"
        
        message = HumanMessage(
                content=[
                   {"type": "text", "text": msg},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{image_data}"},
                    },
                ]
            )
        
        logger.info(f"HumanMessage created successfully")
        
        final_response = None
        step_count = 0
        last_activity_time = time.time()
        
        try: 
            logger.info("Starting LangGraph stream...")
            
            # Create the stream
            stream = self.graph.stream({"messages": [message]}, stream_mode="values")
            
            for s in stream:
                step_count += 1
                current_time = time.time()
                elapsed = current_time - start_time
                since_last_activity = current_time - last_activity_time
                
                logger.info(f"Step {step_count}: Elapsed {elapsed:.2f}s, Since last activity: {since_last_activity:.2f}s")
                
                # Check for timeout
                if elapsed > timeout:
                    logger.error(f"TIMEOUT: LangGraph execution exceeded {timeout} seconds")
                    return f"Error: Execution timed out after {timeout} seconds"
                
                # Log the stream event
                if isinstance(s, dict):
                    logger.info(f"Stream event keys: {list(s.keys())}")
                    
                    # Check for messages
                    if "messages" in s and s["messages"]:
                        logger.info(f"Messages count: {len(s['messages'])}")
                        for i, msg in enumerate(s["messages"]):
                            logger.info(f"Message {i}: {type(msg).__name__}")
                            if hasattr(msg, 'content'):
                                content_preview = str(msg.content)[:200] + "..." if len(str(msg.content)) > 200 else str(msg.content)
                                logger.info(f"Message {i} content preview: {content_preview}")
                    
                    # Check for tool calls
                    if "tool_calls" in s:
                        logger.info(f"Tool calls detected: {s['tool_calls']}")
                        
                else:
                    logger.info(f"Stream event type: {type(s)}")
                    logger.info(f"Stream event: {s}")
                
                # Update last activity time
                last_activity_time = current_time
                
                # Some events may not contain 'messages' (e.g., tool invocation)
                if "messages" not in s or not s["messages"]:
                    logger.info("No messages in event, continuing...")
                    continue

                message = s["messages"][-1]
                logger.info(f"Final message type: {type(message).__name__}")
                logger.info(f"Final message: {truncate_at_base64(str(message))}")
                final_response = message

        except Exception as e:
            logger.error(f"Error during LangGraph stream: {e}")
            logger.error(f"Error type: {type(e).__name__}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return f"Error during execution: {e}"

        total_time = time.time() - start_time
        logger.info(f"=== Dialogue completed in {total_time:.2f}s with {step_count} steps ===")
        
        try:  
            return final_response.content if final_response else "No response received"
        except Exception as e:
            logger.error(f"Error extracting response: {e}")
            return f"Error extracting response: {e}"


    def text_to_speech(self, text, output_file, voice='alloy', response_format="mp3", speed=1):
        '''
        voice: alloy, echo, fable, onyx, nova, and shimmer
        '''
        try:
            # check dir
            dir = os.path.dirname(output_file)
            if not os.path.exists(dir):
                os.mkdir(dir)
            elif not os.path.isdir(dir):
                raise FileExistsError(f"\'{dir}\' is not a directory")

            # tts
            #TODO: Implement TTS

            return True
        except Exception as e:
            print(f'tts err: {e}')
            return False

def main():
            
    """ Main program """
    from action_flow_tool import ActionFlowTool
    from model_helper import ModelHelper
    from keys import OPENAI_API_KEY, OPENAI_ASSISTANT_ID, GEMINI_API_KEY
    from action_flow import ActionFlow
    from pidog import Pidog
    from pydantic import BaseModel, Field

    # dog init 
    # =================================================================
    try:
        #my_dog = Pidog()
        my_dog = None
        time.sleep(1)
    except Exception as e:
        raise RuntimeError(e)
    
    camera_handler = CameraHandler()
    set_camera_handler(camera_handler)  # Set global camera handler
    camera_handler.start()

    action_flow = ActionFlow(my_dog)

    tools = [ActionFlowTool(action_flow, action) for action in action_flow.OPERATIONS]

    # assistant init
    # =================================================================
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro",  temperature=0,max_tokens=None, timeout=None, max_retries=2,google_api_key=GEMINI_API_KEY)
    #llm = ChatDeepSeek(api_key=DEEPSEEK_API_KEY, model="deepseek-chat",)
    #llm = ChatOpenAI(openai_api_key=api_key, model="gpt-4o-mini")
    model_helper = ModelHelper(llm, tools)
    img_path = camera_handler.capture_image()
    model_helper.dialogue_with_img("Continously execute wag the tail until you see a person, describe every time what you see given the tool output", img_path)
    print("done")
    return 0



if __name__ == "__main__":
    main()
