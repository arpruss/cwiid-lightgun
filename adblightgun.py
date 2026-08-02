#!/usr/bin/python3
import subprocess
import math
import sys
import uinput
import re

cmd = ["adb", "logcat", "godot:I", "*:S", "-e", "LightgunData"]

header = "LightgunData R "

map = ((1, uinput.BTN_MOUSE),
        (2, uinput.BTN_RIGHT),
        (4, uinput.KEY_Z),
        (8, uinput.KEY_X),
        (32, uinput.KEY_S))

def isMouse(u):
    return u == uinput.BTN_MOUSE or u == uinput.BTN_RIGHT

def emulateMouse(mouseName="LightgunMouse",controllerName="WiimoteButtons",map=map):
    global running
    
    size = (1920,1080)
    events = [
        uinput.ABS_X + (0,size[0],0,0),
        uinput.ABS_Y + (0,size[1],0,0),
        uinput.BTN_LEFT,
        uinput.BTN_RIGHT
        ]
        
    events2 = [(uinput.KEY_ESC[0],i) for i in range(uinput.KEY_ESC[1], uinput.KEY_MICMUTE[1]+1)]

    with uinput.Device(events,name=mouseName) as device:
        with uinput.Device(events2,name=controllerName) as device2:
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, bufsize=1)
            prevButtons = 0
            uinputPressed = set()
            prevX1 = -1
            prevY1 = -1

            for line in iter(process.stdout.readline, ''):
                line = line.strip()
                if not line:
                    continue
                try:
                    start = line.index(header)
                    data = re.split(r'[,\s]+',line[start+len(header):])
                    x = float(data[0])
                    y = float(data[1])
                    buttons = int(data[2])
                except ValueError:
                    continue

                def press(dev, u):
                    if u not in uinputPressed:
                        dev.emit(u, 1)
                        uinputPressed.add(u)
                        
                def release(dev, u):
                    if u in uinputPressed:
                        dev.emit(u, 0)
                        uinputPressed.remove(u)
                
                pressed = buttons &~ prevButtons
                released = ~buttons & prevButtons
                prevButtons = buttons
                    
                for cb,u in map:
                    if isMouse(u):
                        dev = device
                    else:
                        dev = device2
                    if pressed & cb:
                        press(dev, u)
                    elif released & cb:
                        release(dev, u)

                    if math.isnan(x) or math.isnan(y):
                        x1 = 0
                        y1 = 0
                    else:
                        x1 = int((1-x) * size[0]+.5)
                        y1 = int((1-y) * size[0]+.5)
                        if x1 < 0:
                            x1 = 0
                        elif x1 >= size[0]:
                            x1 = size[0]-1
                        if y1 < 0:
                            y1 = 0
                        elif y1 >= size[1]:
                            y1 = size[1]

                    if x1 != prevX1 or y1 != prevY1:
                        prevX1 = x1
                        prevY1 = y1
                        print(x1,y1)
                        device.emit(uinput.ABS_X,x1,syn=False)
                        device.emit(uinput.ABS_Y,y1)

emulateMouse()
