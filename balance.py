import cwiid
import time
import uinput
import threading
import subprocess
import sys

FRACTION = 0.75

zeroValues = None
lastValues = None
calibrate = False

pressed = set()

def press(key,state):
    if bool(state) != (key in pressed):
        controller.emit(key, state)
        if state:
            pressed.add(key)
        else:
            pressed.remove(key)

def update(values):
    right = values['right_top']+values['right_bottom']
    #left = values['left_top']+values['left_bottom']
    top = values['left_top']+values['right_top']
    bottom = values['left_bottom']+values['right_bottom']
    total = top+bottom
    if total < 20:
        press(uinput.KEY_UP, 0)
        press(uinput.KEY_DOWN, 0)
        press(uinput.KEY_RIGHT, 0)
        press(uinput.KEY_LEFT, 0)
    else:
        print(top/total,right/total)
        if top > FRACTION * total:
            press(uinput.KEY_UP, 1)
            press(uinput.KEY_DOWN, 0)
        elif top < (1-FRACTION) * total:
            press(uinput.KEY_UP, 0)
            press(uinput.KEY_DOWN, 1)
        else:
            press(uinput.KEY_UP, 0)
            press(uinput.KEY_DOWN, 0)
        if right > FRACTION * total:
            press(uinput.KEY_RIGHT, 1)
            press(uinput.KEY_LEFT, 0)
        elif right < (1-FRACTION) * total:
            press(uinput.KEY_RIGHT, 0)
            press(uinput.KEY_LEFT, 1)
        else:
            press(uinput.KEY_RIGHT, 0)
            press(uinput.KEY_LEFT, 0)

def wiimoteCallback(list,t):
    global zeroValues,calibrate
    for item in list:
        print(item)
        if item[0] == 1:
            if (item[1] & 8) and zeroValues is None:
                calibrate = True
                print("calibrating")
            else:
                press(uinput.KEY_SPACE, item[1] & 8)
        elif item[0] == 6:
            values = {}
            for pos in item[1]:
                c = calibration[pos]
                v = item[1][pos]
                if v < c[1]:
                    values[pos] = 17. * (v - c[0]) / (c[1] - c[0])
                else:
                    values[pos] = 17. * (v - c[1]) / (c[2] - c[1]) + 17.
            if calibrate:
                zeroValues = values.copy()
                calibrate = False
            if zeroValues is not None:
                for pos in values:
                    values[pos] -= zeroValues[pos]
                lastValues = values.copy()
            update(values)

def go(wm):
    global calibration,controller
    print("connected")
    c = wm.get_balance_cal()
    calibration = { 'right_top': c[0], 'right_bottom': c[1], 'left_top': c[2], 'left_bottom': c[3] }


    kbdEvents = [(uinput.KEY_ESC[0],i) for i in range(uinput.KEY_ESC[1], uinput.KEY_MICMUTE[1]+1)]

    with uinput.Device(kbdEvents,name="BalanceController") as c:
        controller = c

        wm.mesg_callback = wiimoteCallback

        while running:
            time.sleep(0.25)

def connect():
    print("connecting")
    while True and running:
        try:
            wm = cwiid.Wiimote()
            wm.enable(cwiid.FLAG_MESG_IFC)
            wm.rpt_mode = cwiid.RPT_BTN | cwiid.RPT_BALANCE
            break        
        except RuntimeError:
            time.sleep(0.5)
    if running:
        go(wm)

def run(command):
    global running, args, abortConnect
    subprocess.run(command, shell=True)
    running = False

thread2 = threading.Thread(target=run, args=(sys.argv[1],))
thread2.daemon = True
thread2.start()

running = True

connect()
