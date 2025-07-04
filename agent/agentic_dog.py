# Standard library imports
import os
import sys
import time
import threading
import random
import readline  # optimize keyboard input

# Third-party imports
import speech_recognition as sr
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_deepseek import ChatDeepSeek

# Local imports
from action_flow_tool import ActionFlowTool, set_action_state, action_lock, action_state, actions_to_be_done, set_camera_handler
from camera_handler import CameraHandler
from model_helper import ModelHelper
from keys import OPENAI_API_KEY, OPENAI_ASSISTANT_ID, GEMINI_API_KEY
from action_flow import ActionFlow
from pidog import Pidog
from utils import *

os.popen("pinctrl set 20 op dh") # enable robot_hat speake switch
current_path = os.path.dirname(os.path.abspath(__file__))
os.chdir(current_path) # ch

input_mode = None
with_img = True
args = sys.argv[1:]
if '--keyboard' in args:
    input_mode = 'keyboard'
elif '--agent' in args:
    input_mode = 'agent'
else:
    input_mode = 'voice'

if '--no-img' in args:
    with_img = False
else:
    with_img = True




LANGUAGE = ['en']
# LANGUAGE = ['zh', 'en'] # config stt language code, https://en.wikipedia.org/wiki/List_of_ISO_639_language_codes

# VOLUME_DB = 5
VOLUME_DB = 3

# select tts voice role, counld be "alloy, echo, fable, onyx, nova, and shimmer"
# https://platform.openai.com/docs/guides/text-to-speech/supported-languages
TTS_VOICE = 'shimmer'

VOICE_ACTIONS = ["bark", "bark harder", "pant",  "howling"]

# dog init 
# =================================================================
try:
    my_dog = Pidog()
    time.sleep(1)
except Exception as e:
    raise RuntimeError(e)

action_flow = ActionFlow(my_dog)
tools = [ActionFlowTool(action_flow, action) for action in action_flow.OPERATIONS]

# camera init
# =================================================================
camera_handler = CameraHandler()
set_camera_handler(camera_handler)  # Set global camera handler

# assistant init
# =================================================================
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash",  temperature=0,max_tokens=None, timeout=None, max_retries=2,google_api_key=GEMINI_API_KEY)
#llm = ChatDeepSeek(api_key=DEEPSEEK_API_KEY, model="deepseek-chat",)
#llm = ChatOpenAI(openai_api_key=api_key, model="gpt-4o-mini")
model_helper = ModelHelper(llm, tools)

# Camera start
# =================================================================
if with_img:
    camera_handler.start()

# speech_recognition init
# =================================================================
'''
self.energy_threshold = 300  # minimum audio energy to consider for recording
self.dynamic_energy_threshold = True
self.dynamic_energy_adjustment_damping = 0.15
self.dynamic_energy_ratio = 1.5
self.pause_threshold = 0.8  # seconds of non-speaking audio before a phrase is considered complete
self.operation_timeout = None  # seconds after an internal operation (e.g., an API request) starts before it times out, or ``None`` for no timeout

self.phrase_threshold = 0.3  # minimum seconds of speaking audio before we consider the speaking audio a phrase - values below this are ignored (for filtering out clicks and pops)
self.non_speaking_duration = 0.5  # seconds of non-speaking audio to keep on both sides of the recording

'''
recognizer = sr.Recognizer()
recognizer.dynamic_energy_adjustment_damping = 0.16
recognizer.dynamic_energy_ratio = 1.6
recognizer.pause_threshold = 1.0

# speak_hanlder
# =================================================================
speech_loaded = False
speech_lock = threading.Lock()
tts_file = None

def speak_hanlder():
    global speech_loaded, tts_file
    while True:
        with speech_lock:
            _isloaded = speech_loaded
        if _isloaded:
            gray_print('speak start')
            my_dog.speak_block(tts_file)
            gray_print('speak done')
            with speech_lock:
                speech_loaded = False
        time.sleep(0.05)

speak_thread = threading.Thread(target=speak_hanlder)
speak_thread.daemon = True




# main
# =================================================================
def main():
    global current_feeling, last_feeling
    global speech_loaded
    global action_state, actions_to_be_done
    global tts_file

    my_dog.rgb_strip.close()
    action_flow.change_status(action_flow.STATUS_SIT)

    speak_thread.start()

    while True:
        if input_mode == 'voice':
            # listen
            # ----------------------------------------------------------------
            gray_print("listening ...")

            set_action_state('standby')
            my_dog.rgb_strip.set_mode('listen', 'cyan', 1)

            _stderr_back = redirect_error_2_null() # ignore error print to ignore ALSA errors
            # If the chunk_size is set too small (default_size=1024), it may cause the program to freeze
            with sr.Microphone(chunk_size=8192) as source:
                cancel_redirect_error(_stderr_back) # restore error print
                recognizer.adjust_for_ambient_noise(source)
                audio = recognizer.listen(source)

            # stt
            # ----------------------------------------------------------------
            my_dog.rgb_strip.set_mode('boom', 'yellow', 0.5)

            st = time.time()
            _result = model_helper.stt(audio, language=LANGUAGE)
            gray_print(f"stt takes: {time.time() - st:.3f} s")

            if _result == False or _result == "":
                print() # new line
                continue

        elif input_mode == 'keyboard':
            set_action_state('standby')
            my_dog.rgb_strip.set_mode('listen', 'cyan', 1)

            _result = input(f'\033[1;30m{"intput: "}\033[0m').encode(sys.stdin.encoding).decode('utf-8')

            if _result == False or _result == "":
                print() # new line
                continue

            my_dog.rgb_strip.set_mode('boom', 'yellow', 0.5)

        elif input_mode == 'agent':
            set_action_state('standby')
            my_dog.rgb_strip.set_mode('listen', 'cyan', 1)

            time.sleep(0.1)
            _result = 'What do you see? Do react accordingly'

            my_dog.rgb_strip.set_mode('boom', 'yellow', 0.5)

        else:
            raise ValueError("Invalid input mode")

        # ---------------------------------------------------------------- 
        response = {}
        

        set_action_state('think')

        st = time.time()

        if with_img:
            img_path = './img_input.jpg'
            camera_handler.capture_image(img_path)
            response = model_helper.dialogue_with_img(_result, img_path)
        else:
            response = model_helper.dialogue(_result)
        
        gray_print(f'chat takes: {time.time() - st:.3f} s')

        # Extract answer from response
        if isinstance(response, dict) and 'answer' in response:
            answer = response['answer']
        elif isinstance(response, str):
            answer = response
        else:
            answer = ''

        try:
            # ---- tts ----
            _status = False
            if answer != '':
                st = time.time()
                _time = time.strftime("%y-%m-%d_%H-%M-%S", time.localtime())
                _tts_f = f"./tts/{_time}_raw.wav"
                _status = model_helper.text_to_speech(answer, _tts_f, TTS_VOICE, response_format='wav') # onyx
                if _status:
                    tts_file = f"./tts/{_time}_{VOLUME_DB}dB.wav"
                    _status = sox_volume(_tts_f, tts_file, VOLUME_DB)
                gray_print(f'tts takes: {time.time() - st:.3f} s')

                if _status:
                    with speech_lock:
                        speech_loaded = True
                    my_dog.rgb_strip.set_mode('speak', 'pink', 1)
            else:
                my_dog.rgb_strip.set_mode('breath', 'blue', 1)

            # ---- wait speak done ----
            if _status:
                while True:
                    with speech_lock:
                        if not speech_loaded:
                            break
                    time.sleep(.01)


            # ---- wait actions done ----
            while True:
                with action_lock:
                    if action_state != 'actions':
                        break
                time.sleep(.01)

            ##
            print() # new line

        except Exception as e:
            print(f'actions or TTS error: {e}')


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"\033[31mERROR: {e}\033[m")
    finally:
        if with_img:
            camera_handler.close()
        my_dog.close()
