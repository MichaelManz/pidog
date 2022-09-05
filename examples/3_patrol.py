#!/usr/bin/env python3
import time
from pidog import Pidog
from time import sleep
from preset_actions import bark

t = time.time()
my_dog = Pidog()
my_dog.do_action('stand', speed=80)
my_dog.wait_all_done()
sleep(0.1) 

stand = my_dog.feet_angle_calculation([[0,80],[0,80],[30,75],[30,75]])

def patrol():
    my_dog.rgb_strip.set_mode('breath', 'white', delay=0.1)
    my_dog.do_action('forward', step_count=2, wait=False, speed=98)
    my_dog.do_action('shake_head', step_count=1, wait=False, speed=80)
    my_dog.do_action('wag_tail', step_count=5, wait=False, speed=99)
    print(f"distance: {round(my_dog.distance.value, 2)} cm")
    if my_dog.distance.value < 15:
        print(f"distance: {round(my_dog.distance.value, 2)} cm. DANGER!")
        my_dog.body_stop()
        head_yaw = my_dog.head_current_angle[0]
        my_dog.rgb_strip.set_mode('boom', 'red', delay=0.01)
        my_dog.tail_move([[0]], speed=80)
        my_dog.feet_move([stand], speed=70)
        my_dog.wait_all_done()
        sleep(0.1)
        bark(my_dog, [head_yaw,0,0])
    my_dog.rgb_strip.set_mode('breath', 'white', delay=0.1)

if __name__ == "__main__":
    try:
        while True:  
            patrol()
    finally:
        my_dog.close()



