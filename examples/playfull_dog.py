#!/usr/bin/env python3
from pidog import Pidog
from time import sleep
from math import sin

from vilib import Vilib
from preset_actions import bark
from preset_actions import pant
from preset_actions import bark_action
from preset_actions import howling



my_dog = Pidog()

sleep(0.1)

STEP = 0.5

class BallTrack():
       
     # init
    def __init__(self):
        self.yaw = 0
        self.roll = 0
        self.pitch = 0
        self.flag = False
        self.direction = 0
        self.ball_x = 0
        self.ball_y = 0
        self.width = 0

    def ball_tracking_inner_loop(self):
        self.ball_x = Vilib.detect_obj_parameter['color_x'] - 320
        self.ball_y = Vilib.detect_obj_parameter['color_y'] - 240
        self.width = Vilib.detect_obj_parameter['color_w']

        if self.ball_x > 15 and self.yaw > -80:
            self.yaw -= STEP

        elif self.ball_x < -15 and self.yaw < 80:
            self.yaw += STEP

        if self.ball_y > 25:
            self.pitch -= STEP
            if self.pitch < - 40:
                self.pitch = -40
        elif self.ball_y < -25:
            self.pitch += STEP
            if self.pitch > 20:
                self.pitch = 20

        
        #print(f"yaw: {self.yaw}, pitch: {self.pitch}, width: {self.width}")

        my_dog.head_move([[self.yaw, 0, self.pitch]], immediately=True, speed=100)
        if self.width == 0:
            self.pitch = 0
            self.yaw = 0
        elif self.width < 300:
            if my_dog.is_legs_done():
                if self.yaw < -30:
                    print("turn right")
                    my_dog.do_action('turn_right', speed=98)
                elif self.yaw > 30:
                    print("turn left")
                    my_dog.do_action('turn_left', speed=98)
                else:
                    my_dog.do_action('forward', speed=98)

        sleep(0.02)


def delay(time):
    my_dog.wait_legs_done()
    my_dog.wait_head_done()
    sleep(time)

def exection_loop():
    Vilib.camera_start(vflip=False, hflip=False)
    Vilib.display(local=True, web=True)
    Vilib.color_detect(color="red")
    sleep(0.2)
    print('start')

    wake_up()

    ball_track = BallTrack()

    #my_dog.do_action('stand', speed=50)
    my_dog.head_move([[ball_track.yaw, 0, ball_track.pitch]], immediately=True, speed=80)
    delay(0.5)

    while True:
        ball_track.ball_tracking_inner_loop()

        # alert
        response_inner_loop()


def response_inner_loop():

    if my_dog.dual_touch.read() != 'N':
        if len(my_dog.head_action_buffer) < 2:
            head_nod(1)
            my_dog.wait_legs_done()
            my_dog.do_action('wag_tail', step_count=5, speed=80)
            howling(my_dog)


def wake_up():
    # stretch
    #my_dog.rgb_strip.set_mode('listen', color='yellow', bps=0.6, brightness=0.8)
    my_dog.do_action('stretch', speed=50)
    my_dog.head_move([[0, 0, 30]]*2, immediately=True)
    my_dog.wait_all_done()
    sleep(0.5)
    my_dog.head_move([[0, 0, -30]], immediately=True, speed=90)
    # sit and wag_tail
    my_dog.do_action('sit', speed=25)
    my_dog.wait_legs_done()
    my_dog.do_action('wag_tail', step_count=10, speed=100)
    #my_dog.rgb_strip.set_mode('breath', color=[245, 10, 10], bps=2.5, brightness=0.8)
    pant(my_dog, pitch_comp=-30, volume=80)
    my_dog.wait_all_done()
    my_dog.rgb_strip.set_mode('breath', 'pink', bps=0.5)
   

def lean_forward():
    my_dog.speak('angry', volume=80)
    bark_action(my_dog)
    sleep(0.2)
    bark_action(my_dog)
    sleep(0.4)
    bark_action(my_dog)

def head_nod(step):
    y = 0
    r = 0
    p = 30
    angs = []
    for i in range(20):
        r = round(10*sin(i*0.314), 2)
        p = round(20*sin(i*0.314) + 10, 2)
        angs.append([y, r, p])

    my_dog.head_move(angs*step, immediately=False, speed=80)


if __name__ == "__main__":
    try:
        exection_loop()
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"\033[31mERROR: {e}\033[m")
    finally:
        Vilib.camera_close()
        my_dog.close()

