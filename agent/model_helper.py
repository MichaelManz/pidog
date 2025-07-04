import base64
import json

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

import time
import shutil
import os

import yaml

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
            ("user", "Remember, always be polite!"),
        ])
        
        class CustomState(MessagesState):
            image_data: str = ""
            current_action: str = ""
            action_timestamp: float = 0.0
            robot_status: str = "standby"  # standby, thinking, performing_action

        # Construct the ReAct agent
        self.graph = create_react_agent(model=self.llm, tools=dog_tools, prompt=self.prompt, debug=True)


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

    def dialogue_with_img(self, msg, img_path):

        with open(img_path, "rb") as image_file:
            image_data = base64.b64encode(image_file.read()).decode("utf-8")
        
        message = HumanMessage(
                content=[
                    {"type": "text", "text": msg},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{image_data}"},
                    },
                ]
            )
        
        for s in self.graph.stream({"messages": [message]}, stream_mode="values"):
            # Some events may not contain 'messages' (e.g., tool invocation)
            if "messages" not in s or not s["messages"]:
                print(s)  # debug fallback
                continue

            message = s["messages"][-1]
            if isinstance(message, tuple):
                print(message)
            else:
                value = message.pretty_print()

        try:
            value = s["messages"][-1].content # convert to dict
            return value
        except Exception as e:
            return str(value)


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
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash",  temperature=0,max_tokens=None, timeout=None, max_retries=2,google_api_key=GEMINI_API_KEY)
    #llm = ChatDeepSeek(api_key=DEEPSEEK_API_KEY, model="deepseek-chat",)
    #llm = ChatOpenAI(openai_api_key=api_key, model="gpt-4o-mini")
    model_helper = ModelHelper(llm, tools)
    img_path = camera_handler.capture_image()
    model_helper.dialogue_with_img("Bark until you see a person in the camera", img_path)
    return 0



if __name__ == "__main__":
    main()
