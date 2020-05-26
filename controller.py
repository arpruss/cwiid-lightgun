import cwiid
import uinput
import time
import math
import os
import sys
import atexit
import threading
import argparse
import subprocess

abortConnect = False

CONFIG_DIR = os.sep.join((os.path.expanduser("~"),".wiilightgun"))
LED_FILE = os.sep.join((os.path.expanduser("~"),".wiilightgun","irledcoordinates"))
SCREENSHOT_FILE = os.sep.join((os.path.expanduser("~"),".wiilightgun","screenshot"))
CALIBRATION_FILE = os.sep.join((os.path.expanduser("~"),".wiilightgun","wiimotecalibration"))
BLACK = (0,0,0)
WHITE = (255,255,255)
RED = (255,0,0)
GRAY = (64,64,64)
DARK_GREEN = (0,64,0)
VERY_DARK_GREEN = (0,32,0)
WINDOW_SIZE = None
wm = None
running = True
WIIMOTE_EVENT = threading.Event()
CONNECTED_EVENT = threading.Event()
DISCONNECT_DETECT_TIME = 4
lastMessage = 0
LONG_PRESS_TIME = 0.75
FONT_SIZE = 0.05
REPEAT_DELAY = 0.75
REPEAT_TIME = 0.05
CENTER_X = 1024/2
CENTER_Y = 768/2
NUNCHUK_SHIFT = 16
NUNCHUK_C = cwiid.NUNCHUK_BTN_C << NUNCHUK_SHIFT
NUNCHUK_Z = cwiid.NUNCHUK_BTN_Z << NUNCHUK_SHIFT
NUNCHUK_DEADZONE = 40
NUNCHUK_HYSTERESIS = 10

verticalMap = ((cwiid.BTN_B, uinput.BTN_MOUSE),
        (cwiid.BTN_A, uinput.BTN_RIGHT),
        (cwiid.BTN_1, uinput.KEY_Z),
        (cwiid.BTN_2, uinput.KEY_X),
        (NUNCHUK_Z, uinput.KEY_S),
        (NUNCHUK_C, uinput.KEY_A),
        (cwiid.BTN_PLUS, uinput.KEY_Q),
        (cwiid.BTN_HOME, uinput.KEY_ENTER),
        (cwiid.BTN_DOWN, uinput.KEY_DOWN),
        (cwiid.BTN_UP, uinput.KEY_UP),
        (cwiid.BTN_LEFT, uinput.KEY_LEFT),
        (cwiid.BTN_RIGHT, uinput.KEY_RIGHT))

minusVerticalMap = ((cwiid.BTN_DOWN, uinput.KEY_F6),
        (cwiid.BTN_UP, uinput.KEY_F7),
        (cwiid.BTN_LEFT, uinput.KEY_F4),
        (cwiid.BTN_RIGHT, uinput.KEY_F2),
        (cwiid.BTN_A, uinput.KEY_F1),
        (cwiid.BTN_B, uinput.KEY_TAB),
        (cwiid.BTN_HOME, uinput.KEY_F2),
        (cwiid.BTN_1, uinput.KEY_LEFTBRACE),
        (cwiid.BTN_2, uinput.KEY_RIGHTBRACE))
       
horizontalMap = (
        (cwiid.BTN_B, uinput.KEY_S),
        (cwiid.BTN_A, uinput.KEY_A),
        (cwiid.BTN_1, uinput.KEY_Z),
        (cwiid.BTN_2, uinput.KEY_X),
        (NUNCHUK_Z, uinput.KEY_S),
        (NUNCHUK_C, uinput.KEY_A),
        (cwiid.BTN_HOME, uinput.KEY_ENTER),
        (cwiid.BTN_PLUS, uinput.KEY_Q),
        (cwiid.BTN_DOWN, uinput.KEY_RIGHT),
        (cwiid.BTN_UP, uinput.KEY_LEFT),
        (cwiid.BTN_LEFT, uinput.KEY_DOWN),
        (cwiid.BTN_RIGHT, uinput.KEY_UP))

minusHorizontalMap = (
        (cwiid.BTN_DOWN, uinput.KEY_F2),
        (cwiid.BTN_UP, uinput.KEY_F4),
        (cwiid.BTN_LEFT, uinput.KEY_F6),
        (cwiid.BTN_RIGHT, uinput.KEY_F7),
        (cwiid.BTN_A, uinput.KEY_F1),
        (cwiid.BTN_B, uinput.KEY_TAB),
        (cwiid.BTN_HOME, uinput.KEY_F2),
        (cwiid.BTN_1, uinput.KEY_LEFTBRACE),
        (cwiid.BTN_2, uinput.KEY_RIGHTBRACE))
       
class FakeWiimote():
    def __init__(self):
        self.state = { "acc":(128,128,128), "buttons":0, "ir_src":[], "fake":True }

def getAddress(wm):        
    try:
        return wm.address
    except:
        return "unknown"
        
def getButtons(state):
    b = state['buttons']
    if 'nunchuk' in state:
        return b | state['nunchuk']['buttons'] << NUNCHUK_SHIFT
    else:
        return b

def wiimoteWait(timeout=None):
    if isinstance(wm, FakeWiimote):
        time.sleep(1)
        return

    WIIMOTE_EVENT.wait(timeout if timeout is not None and timeout < DISCONNECT_DETECT_TIME else DISCONNECT_DETECT_TIME)
    if time.time() > lastMessage + DISCONNECT_DETECT_TIME:
        print("Disconnect detected")
        connect()
        CONNECTED_EVENT.wait()
        print("Reconnected")
    WIIMOTE_EVENT.clear()
    
def wiimoteCallback(list,t):
    global lastMessage
    lastMessage = time.time()
    WIIMOTE_EVENT.set()

def emulate(controllerName="WiimoteButtons", horizontal=False):
    global running
    
    events = [(uinput.KEY_ESC[0],i) for i in range(uinput.KEY_ESC[1], uinput.KEY_MICMUTE[1]+1)]

    def updateLEDs():
        if horizontal:
            wm.led = cwiid.LED2_ON | cwiid.LED3_ON
        else:
            wm.led = cwiid.LED1_ON | cwiid.LED4_ON

    with uinput.Device(events,name=controllerName) as device:
            try:
                prevButtons = 0
                prevNunchukX = 128
                prevNunchukY = 128
                uinputPressed = set()
                CONNECTED_EVENT.wait()
                updateLEDs()
                
                def press(dev, u):
                    if u not in uinputPressed:
                        dev.emit(u, 1)
                        uinputPressed.add(u)
                        
                def release(dev, u):
                    if u in uinputPressed:
                        dev.emit(u, 0)
                        uinputPressed.remove(u)
                
                while running:
                    wiimoteWait()
                    buttons = getButtons(wm.state)
                    pressed = buttons &~ prevButtons
                    released = ~buttons & prevButtons
                    prevButtons = buttons
                    
                    if buttons & cwiid.BTN_MINUS:
                        if pressed & cwiid.BTN_PLUS:
                            horizontal = not horizontal
                            updateLEDs()
                        map = minusVerticalMap if not horizontal else minusHorizontalMap
                        for wii,u in map:
                            if pressed & wii:
                                press(device, u)
                            elif released & wii:
                                release(devic2, u)
                    elif pressed or released:
                        map = verticalMap if not horizontal else horizontalMap

                        for wii,u in map:
                            if pressed & wii:
                                press(device, u)
                            elif released & wii:
                                release(device, u)
                                
                    if 'nunchuk' in wm.state:
                        def stick(offset,prevOffset,key):
                            if offset < NUNCHUK_DEADZONE-NUNCHUK_HYSTERESIS and prevOffset >= NUNCHUK_DEADZONE-NUNCHUK_HYSTERESIS:
                                release(device, key)
                            elif offset >= NUNCHUK_DEADZONE:
                                press(device, key)

                        x,y = wm.state['nunchuk']['stick']

                        stick(x-128,prevNunchukX-128,uinput.KEY_RIGHT)
                        stick(128-x,128-prevNunchukX,uinput.KEY_LEFT)
                        stick(y-128,prevNunchukY-128,uinput.KEY_UP)
                        stick(128-y,128-prevNunchukY,uinput.KEY_DOWN)

                        prevNunchukX, prevNunchukY = x,y

            except KeyboardInterrupt:
                pass
            finally:
                for u in uinputPressed:
                    device.emit(u, 0)


def connect(backgroundTimeout=0):
    global wm, lastMessage, CENTER_X, CENTER_Y
    wm = None
    t0 = time.time()
    CONNECTED_EVENT.clear()
    while True:
        try:
            wm = cwiid.Wiimote()
            wm.mesg_callback = wiimoteCallback
            wm.enable(cwiid.FLAG_MESG_IFC)
            wm.rpt_mode = cwiid.RPT_BTN | cwiid.RPT_EXT
            wm.led = cwiid.LED1_ON | cwiid.LED4_ON
            # give it a bit of extra time for messages to start flowing
            CONNECTED_EVENT.set()
            lastMessage = time.time()+5
            return
        except RuntimeError:
            if (backgroundTimeout and time.time() > t0 + backgroundTimeout) or abortConnect:
                print("Giving up, connecting a fake wiimote")
                wm = FakeWiimote()
                CONNECTED_EVENT.set()
                return

def run(command):
    global running, args, abortConnect
    subprocess.run(command, shell=True)
    running = False
    abortConnect = True
                
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Calibrate and use Wiimote with four IR LEDs around screen.")
    parser.add_argument("-o", "--horizontal", action="store_true", help="Horizontal mode (without lightgun)")
    #parser.add_argument("-m", "--mouse-name", help="Set name of mouse device", default="LightgunMouse")
    parser.add_argument("-b", "--buttons-name", help="Set name of buttons device", default="WiimoteButtons")
    parser.add_argument("-B", "--background-connect", type=float, default=0, help="Connect in background for this many seconds")
    parser.add_argument("command", help="Run this command while simulating a mouse", nargs="?")
    args = parser.parse_args()

    try:
        os.mkdir(CONFIG_DIR)
    except:
        pass
    
    thread = threading.Thread(target=connect, args=(args.background_connect,))
    thread.daemon = True
    thread.start()
    
    if not args.background_connect:
        print("Press 1+2 on Wii Remote, making sure Wii is turned off.")
        CONNECTED_EVENT.wait()
        print("Ready.")
        
    if args.command:
        thread = threading.Thread(target=run, args=(args.command,))
        thread.daemon = True
        thread.start()
    emulate(controllerName=args.buttons_name,horizontal=args.horizontal)
