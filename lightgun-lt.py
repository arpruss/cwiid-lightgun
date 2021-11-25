
import cwiid
import uinput
import time
import math
import os
import pygame
import sys
import numpy as np
import atexit
import threading
import argparse
import subprocess

USE_P3P = True # use P3P if only three points are visible at a given time

if USE_P3P:
    import p3p

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
PXSCALE = 1
ACCEL_FILTER_TIME = 0.25
PYGAME_MODE = True
RUMBLE_TIME = 0.06
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
ASPECT_RATIO = 1900./1080
#CAMERA_ASPECT_RATIO = 1024./768

# for moderate angles, setting this to False gets about half a pixel more
# precision, which probably isn't worth it
FAST_CORRECTION = False
CALIBRATION_CORNERS = ((0.125,0.05), (0.875,0.05), (0.875,0.95), (0.125,0.95))
UNIT_SQUARE = ((0,0), (1,0), (1,1), (0,1))


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
       
class Config():
    def __init__(self):
        self.center = {}
        try:
            with open(CALIBRATION_FILE) as f:
                for line in f:
                    try:
                        a,d = line.strip().split(maxsplit=2)
                        self.center[a] = tuple(map(float,d.split(",")))
                    except:
                        pass
        except:
            pass
            
        self.aspect = 1900./1080.
        self.ledLocations = None
        self.yCorrection = 0
        try:
            with open(LED_FILE) as f:
                s = tuple(map(float,f.readline().strip().split(",")))
                ledLocations = [ [0,0] for i in range(4) ]
                for i in range(4):
                    line = f.readline().strip().split(",")
                    for j in range(2):
                        ledLocations[i][j]=float(line[j])/s[j]
                self.ledLocations = ledLocations
                try:
                    for line in f:
                        l = line.strip().split()
                        if l[0].lower() == "ycorrection":
                            self.yCorrection = float(l[1])/s[1]
                        elif l[0].lower() == "aspect":
                            self.aspect = float(l[1])
                except:
                    pass
        except Exception as e:
            pass

    def haveCenter(self,wm):
        return getAddress(wm) in self.center

    def setLEDLocations(self,loc,size=(1.,1.)):
        self.ledLocations = [[loc[i][0]/size[0],loc[i][1]/size[1]] for i in range(4)]
            
    def getCenter(self,wm):
        return self.center.get(getAddress(wm), (1024/2.,768/2.))
        
    def setCenter(self,wm,c):
        self.center[getAddress(wm)] = c
        
    def saveCalibration(self):
        with open(CALIBRATION_FILE, "w") as f:
            for a in self.center:
                f.write("%s %g,%g\n" % (a,self.center[a][0],self.center[a][1]))
                
    def saveLEDs(self):
        if self.ledLocations:
            with open(LED_FILE, "w") as f:
                f.write("1,1\n")
                for l in self.ledLocations:
                    f.write("%g,%g\n" % tuple(l))
                f.write("ycorrection %g\n" % self.yCorrection)
                f.write("aspect %g\n" % self.aspect)
            
    def pointerPosition(self,irQuad):
        h = Homography(irQuad,self.ledLocations)
        if self.yCorrection:
            if FAST_CORRECTION:
                xy = h.apply((0,0))
                xy2 = h.apply((0.01,0.01))
                dx,dy = (xy2[0]-xy[0])*self.aspect,xy2[1]-xy[1]
                d = math.hypot(dx,dy)
                return xy[0]+self.yCorrection*dx/d/self.aspect,xy[1]+self.yCorrection*dy/d
            else:
                return h.apply((0,self.yCorrection/h.minimumScalingAtOrigin(self.aspect))) 
        else:
            return h.apply((0,0))

CONFIG = Config()
            
class FakeWiimote():
    def __init__(self):
        self.state = { "acc":(128,128,128), "buttons":0, "ir_src":[], "fake":True }

# find a local minimum        
def minimize(f,a,b,n=4):
    fa = f(a)
    fb = f(b)
    fab = fa
    while n>0:
        ab = 0.5*(a+b)
        fab = f(ab)
        if fa<fb:
            fb=fab
            b=ab
        else:
            fa=fab
            a=ab
        n-=1
    if fa<fb:
        return fa,a
    else:
        return fb,b

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

# 1280
INTRINSIC = np.array( ( [1280/768.,0,0.5],
    [0,1280/768.,0.5],
    [0,0,1] ), dtype=np.float32 )

class Homography:
    def __init__(self,*args):
        if len(args) == 2:
            input = args[0]
            output = args[1]
            #http://www.corrmap.com/features/homography_transformation.php
            x1,x2,x3,x4=tuple(input[i][0] for i in range(4))
            y1,y2,y3,y4=tuple(input[i][1] for i in range(4))
            X1,X2,X3,X4=tuple(output[i][0] for i in range(4))
            Y1,Y2,Y3,Y4=tuple(output[i][1] for i in range(4))
            matrix = np.array( ( [x1,y1,1,0,0,0,-x1*X1,-y1*X1],
                                 [x2,y2,1,0,0,0,-x2*X2,-y2*X2],
                                 [x3,y3,1,0,0,0,-x3*X3,-y3*X3],
                                 [x4,y4,1,0,0,0,-x4*X4,-y4*X4],
                                 [0,0,0,x1,y1,1,-x1*Y1,-y1*Y1],
                                 [0,0,0,x2,y2,1,-x2*Y2,-y2*Y2],
                                 [0,0,0,x3,y3,1,-x3*Y3,-y3*Y3],
                                 [0,0,0,x4,y4,1,-x4*Y4,-y4*Y4] ) )
            inv = np.linalg.inv(matrix)
            self.a,self.b,self.c,self.d,self.e,self.f,self.g,self.h = inv.dot(np.array([X1,X2,X3,X4,Y1,Y2,Y3,Y4]))
        elif len(args) == 1:
            if len(args[0]) == 8:
                self.a,self.b,self.c,self.d,self.e,self.f,self.g,self.h = args[0]
            elif len(args[0]) == 3:
                m = np.array(args[0]) / args[0][2][2]
                self.a,self.b,self.c = m[0]
                self.d,self.e,self.f = m[1]
                self.g,self.h = m[2][:2]

    @property
    def matrix(self):
        return np.array( ( (self.a,self.b,self.c),(self.d,self.e,self.f),(self.g,self.h,1) ) )

    #def jacobianAtOrigin(self):
    #    return np.array( (self.a-self.c*self.g, self.b-self.c*self.h),
    #                     (self.d-self.f*self.g, self.e-self.f*self.h) )

    def minimumScalingAtOrigin(self, aspect):
        # This measures the lowest scaling between camera and screen coordinates.
        # Geometric intuition (I haven't proved it) suggests this corresponds to
        # the correct conversion between camera and screen coordinates for y adjustment.
        # This is the smallest singular value of the ^acobian at the origin
        # ( https://lucidar.me/en/mathematics/singular-value-decomposition-of-a-2x2-matrix/ )
        A = aspect*(self.a-self.c*self.g)
        B = aspect*(self.b-self.c*self.h)
        C = self.d-self.f*self.g
        D = self.e-self.f*self.h
        
        S1 = A*A+B*B+C*C+D*D
        u = A*A+B*B-C*C-D*D
        v = A*C+B*D
        S2 = math.sqrt(u*u+v*v)
        return math.sqrt((S1-S2)/2.)

    def apply(self,xy):
        x,y=xy
        den = self.g*x+self.h*y+1.
        return (self.a*x+self.b*y+self.c)/den,(self.d*x+self.e*y+self.f)/den

    def __repr__(self):
        return repr((self.a,self.b,self.c,self.d,self.e,self.f,self.g,self.h))

def drawText(s,x=0.5,y=0.5,color=WHITE):
    text = MYFONT.render(s, True, color)
    textRect = text.get_rect()
    textRect.center = (WINDOW_SIZE[0]*x,WINDOW_SIZE[1]*y)
    surface.blit(text, textRect)

def drawBlob(xy,size=3,color=WHITE):
    pygame.draw.rect(surface, color, (xy[0]*WINDOW_SIZE[0]-size*PXSCALE/2, (1.-xy[1])*WINDOW_SIZE[1]-size*PXSCALE/2, size*PXSCALE, size*PXSCALE))

def drawCross(xy,thickness=3,size=0.25,color=WHITE):
    l = size*WINDOW_SIZE[1]/2.
    x = xy[0]*WINDOW_SIZE[0]
    y = (1-xy[1])*WINDOW_SIZE[1]
    t = thickness*PXSCALE
    pygame.draw.rect(surface, color, (x-l/2.,y-t/2.,l,t))
    pygame.draw.rect(surface, color, (x-t/2.,y-l/2.,t,l))

def showPoints(ir,irQuad):
    cy = int(WINDOW_SIZE[1] * 0.25)
    cx = WINDOW_SIZE[0] // 2
    height = int(WINDOW_SIZE[1] * 0.4)
    width = height * 4 // 3
    
    pygame.draw.rect(surface, VERY_DARK_GREEN, (cx-width//2, cy-height//2, width, height))

    rawPoints = [getPoint(p) for p in ir if p is not None]

    if irQuad:
        for i in range(4):
            xy = irQuad[i]
            x = int(cx + xy[0] * height)
            y = int(cy + (-xy[1]) * height)
            text = MYFONT.render(str(i+1), True, RED if (tuple(xy) in rawPoints) else GRAY)
            textRect = text.get_rect()
            textRect.center = (x,y)
            surface.blit(text, textRect)
    for point in ir:
        if point is not None:
            xy = getPoint(point)
            size = point.get('size',1)
            x = int(cx + xy[0] * height)
            y = int(cy + (-xy[1]) * height)
            pygame.draw.rect(surface, WHITE, (x-size*PXSCALE/2, y-size*PXSCALE/2, size*PXSCALE, size*PXSCALE))
    
def getPoint(p):
    return (p['pos'][0]-CENTER_X) / 768., (p['pos'][1]-CENTER_Y) / 768.
    
accelHistory = []    
lastAngle = math.pi / 2
    
def updateAcceleration(accel):
    global lastAngle
    
    a = accel[0]-128.,accel[1]-128.,accel[2]-128.
    t = time.time()
    accelHistory.append((t,a))
    while accelHistory[0][0] < t-ACCEL_FILTER_TIME:
        del accelHistory[0]
    s = [0,0,0]
    for _,a in accelHistory:
        s[0] += a[0]
        s[1] += a[1]
        s[2] += a[2]
    try:
        lastAngle = math.atan2(s[2],s[0])
    except:
        pass

lastQuad = None

def identifyPoints(points):
    rot = (lastAngle-math.pi/2)
    c = math.cos(rot)
    s = math.sin(rot)

    n = len(points)

    cx = sum(p[0] for p in points)/float(n)
    cy = sum(p[1] for p in points)/float(n)

    identified = [None for i in range(n)]

    def rotate(p):
        return (p[0]*c-p[1]*s,p[0]*s+p[1]*c)

    for i in range(n):
        p = rotate((points[i][0]-cx, points[i][1]-cy))
        if p[0] < 0 and p[1] < 0 and 0 not in identified:
            identified[i] = 0
        elif p[0] > 0 and p[1] < 0 and 1 not in identified:
            identified[i] = 1
        elif p[0] > 0 and p[1] > 0 and 2 not in identified:
            identified[i] = 2
        elif p[0] < 0 and p[1] > 0 and 3 not in identified:
            identified[i] = 3

    if None in identified:
        unidentified = list(set(range(n))-set(identified))
        if len(unidentified) == 1:
            identified[identified.index(None)] = unidentified[0]

    return identified

def points3To4(points):
    if CONFIG.ledLocations is None:
        return None

    identified = identifyPoints(points)
    if None in identified:
        return None

    missing = tuple(set((0,1,2,3)) - set(identified))[0]

    def fix(p):
        return (p[0]*CONFIG.aspect,p[1],0)

    source = (fix(CONFIG.ledLocations[identified[i]]) for i in range(3))
    hs = p3p.homographies(*points,*source)
    if not hs:
        return None
    return None

    bestR2 = float("inf")
    missingLED = fix(CONFIG.ledLocations[missing])

    if lastQuad is None:
        return None
    lastPoint = lastQuad[missing]
    bestD = float("inf")
    bestP = None
    for h in hs:
        p = p3p.applyHomography(h, missingLED)
        d = math.hypot(p[0]-lastPoint[0],p[1]-lastPoint[1])
        if d < bestD:
            bestD = d
            bestP = p
    if bestP is None:
        return None

    out = [None,None,None,None]
    for i in range(4):
        if i == missing:
            out[i] = bestP
        else:
            out[i] = points[identified.index(i)]

    return out
    
def getIRQuad(ir):
    global lastQuad

    if ir is None:
        return None

    # get the IR LED quad, normalized and arranged counterclockwise from lower left
    count = sum((1 for p in ir if p is not None))

    if count !=3 and count != 4:
        return None

    if count == 3 and not USE_P3P:
        return None

    points = [getPoint(p) for p in ir if p is not None]

    if count == 3:
        points = points3To4(points)
        if points is None:
            return None
        lastQuad = points
        return points
    else:
        identified = identifyPoints(points)
        if None in identified:
            return None
        lastQuad =[points[identified.index(i)] for i in range(4)]
        return lastQuad
    
def getDisplaySize():
    info = pygame.display.Info()
    return info.current_w, info.current_h    

def screenshot():
    size = getDisplaySize()
    img = pygame.Surface(size)
    img.blit(surface,(0,0),((0,0),size))
    pygame.image.save(img,SCREENSHOT_FILE+str(time.time())+".png")
    
def checkQuitAndKeys():
    global running
    pygame.event.pump()
    keys = set()
    for event in pygame.event.get():
        if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
            running = False
            sys.exit(0)
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_F8:
                screenshot()
            keys.add(event.key)
    if wm and 'buttons' in wm.state:
        if wm.state['buttons'] & cwiid.BTN_HOME:
            running = False
            sys.exit(0)
    return keys
    
def verticalArrow(xy,length,color=WHITE):
    x,y = xy
    pygame.draw.line(surface,color,[x,y],[x,y+length],2)
    pygame.draw.line(surface,color,[x,y],[x-length//2,y+length//2],2)
    pygame.draw.line(surface,color,[x,y],[x+length//2,y+length//2],2)

def drawArrow(xy,bottom,color=WHITE):
    x,y = xy
    if bottom:
        length = -y
        edgeY = WINDOW_SIZE[1]-1
        signY = -1
    else:
        length = y-WINDOW_SIZE[1]
        edgeY = 0
        signY = 1
    verticalArrow((x,edgeY),length*signY,color=color)
    
def measure(flexible=False,screenWidth=1.):
    global running,surface
    
    size = WINDOW_SIZE
    scale = float(screenWidth)/WINDOW_SIZE[0]
    corner = 0

    ledPixel = [[int(size[0]*1./3),int(-0.1*size[1])],[int(size[0]*2./3),int(-0.1*size[1])],[int(size[0]*2./3),int(1.1*size[1])],[int(size[0]*1./3),int(1.1*size[1])]]

    if CONFIG.ledLocations:
        for i in range(4):
            for j in range(2):
                ledPixel[i][j] = int(math.floor(0.5+CONFIG.ledLocations[i][j]*size[j]))
    
    buttonMap = ((cwiid.BTN_LEFT,(-1,0)),(cwiid.BTN_RIGHT,(1,0)),(cwiid.BTN_UP,(0,1)),(cwiid.BTN_DOWN,(0,-1)))

    prevButtons = 0
    prevTime = time.time()
    CONNECTED_EVENT.wait()
    
    if not CONFIG.haveCenter(wm):
        center()    
    
    nextRepeat = 0
    done = False
    yCorrection = int(math.floor(CONFIG.yCorrection*size[1] + 0.5))
    
    while running:
        time.sleep(0.005)
        checkQuitAndKeys()
        surface.fill(DARK_GREEN)
        updateAcceleration(wm.state['acc'])
        ir = wm.state['ir_src']
        irQuad = getIRQuad(ir)
        
        showPoints(ir,irQuad)

        if irQuad:
            CONFIG.setLEDLocations(ledPixel,size)
            CONFIG.yCorrection = yCorrection / size[1]
            s = CONFIG.pointerPosition(irQuad)
            drawCross(s,color=RED)

        drawText("HOME: quit without saving", y=0.5+0.075*2)
        drawText("A: done", y=0.5+0.075*3)

        buttons = getButtons(wm.state)
        pressed = buttons &~ prevButtons
        released = ~buttons & prevButtons
        prevButtons = buttons

        move = (0,0)

        for wii,dir in buttonMap:
            if pressed & wii:
                nextRepeat = time.time()+REPEAT_DELAY
                move = dir
                break
            elif buttons & wii and time.time()>=nextRepeat:
                nextRepeat = time.time()+REPEAT_TIME
                move = dir
                break

        if corner==4:
            yCorrection += move[1]

        for i in range(4):
            xy = ledPixel[i]
            bottom = UNIT_SQUARE[i][1] < 0.5
            if xy[1] > 0 and bottom:
                xy[1] = 0
            if xy[1] < WINDOW_SIZE[1]-1 and not bottom:
                xy[1] = WINDOW_SIZE[1]-1
            if xy[0] < 0:
                xy[0] = 0
            if xy[0] >= WINDOW_SIZE[0]:
                xy[0] = WINDOW_SIZE[0]-1
            if not flexible:
                ledPixel[i^1][1] = xy[1]
            if i == corner:
                xy[0] += move[0]
                xy[1] += move[1]
            x,y = xy
            if bottom:
                length = -y
                signY = 1
            else:
                length = y-size[1]
                signY = -1
            drawArrow(xy,bottom,color=WHITE if i==corner else GRAY)
        
        if corner<4:
            drawText("DPad: move LED location",y=0.5)
            drawText("-/+: next/previous setting",y=0.5+0.075)
            drawText("LED is %.4g units (%.1f px) off-screen" % (length*scale, length),y=0.5+0.075*4)
        else:
            drawText("Up/Down: adjust Y correction",y=0.5)
            drawText("-/+: next/previous setting",y=0.6)               
            drawText("Y correction is %.4g units (%.1f px)" % (yCorrection*scale, yCorrection),y=0.5+0.075*4)
            ax = int(size[0]//4)
            ay = int(size[1]*0.5-yCorrection/2)            
            b = min(size[1] * .2, yCorrection + size[1] * .1)
            pygame.draw.rect(surface, VERY_DARK_GREEN, ( [ax-b//2,ay+yCorrection//2-b//2,b,b] ))
            verticalArrow((ax,ay),yCorrection,color=WHITE)

        if pressed & cwiid.BTN_PLUS:
            corner = (corner+1) % 5
        elif pressed & cwiid.BTN_MINUS:
            corner = (corner-1) % 5
        elif ( pressed & cwiid.BTN_A ):
            done = True
            break                        
            
        pygame.display.flip()

    if not done:
        return False
        
    CONFIG.setLEDLocations(ledPixel,size)
    CONFIG.yCorrection = yCorrection / size[1]
    CONFIG.saveLEDs()
        
    return True
                
# compute location of LEDs in screen coordinates                
def computeLEDs(calibrationData,flexible):
    avg = [[0,0] for i in range(len(CALIBRATION_CORNERS))]
    
    for i in range(len(CALIBRATION_CORNERS)):
        for j in range(2):
            avg[i][j] = sum((c[j] for c in calibrationData[i])) / len(calibrationData[i])

    fromIRToScreen = Homography(avg,CALIBRATION_CORNERS)
    
    leds = list(list(fromIRToScreen.apply(p)) for p in UNIT_SQUARE)
    
    if not flexible:
        y = (leds[0][1]+leds[1][1])/2.
        leds[0][1] = y
        leds[1][1] = y
        y = (leds[2][1]+leds[3][1])/2.
        leds[2][1] = y
        leds[3][1] = y
        
    return leds

def calibrate(flexible=False):
    global CALIBRATION_CORNERS,running,surface

    if args.terminal:
        print("Calibration cannot work with terminal mode.")

    corner = 0
    calibrationData = tuple([] for i in range(len(CALIBRATION_CORNERS)))

    prevButtons = 0
        
    # coordinate systems: 
    #  IR: (0,0) is lower-left LED and (1,1) is upper-right LED
    #  screen: (0,0) is lower-left corner of screen and (1,1) is upper-right corner of screen
    #  calibrationData is in IR coordinates
    #  CALIBRATION_CORNERS is in screen coordinates
    
    CONNECTED_EVENT.wait()
    
    if not CONFIG.haveCenter(wm):
        center()

    lastCalibrated = time.time()

    while running:
        wiimoteWait(0.25)
        ir = wm.state['ir_src']
        buttons = getButtons(wm.state)
        newButtons = buttons & ~prevButtons
        prevButtons = buttons
        checkQuitAndKeys()
        surface.fill(BLACK)
        updateAcceleration(wm.state['acc'])
        irQuad = getIRQuad(ir)
        showPoints(ir,irQuad)
        debounced = 0.5 + lastCalibrated < time.time()
        valid = irQuad and debounced
        drawCross(CALIBRATION_CORNERS[corner],color=RED if valid else GRAY)
        if debounced:
            drawText("Press trigger (B"+(" or C" if 'nunchuk' in wm.state else "")+") while pointing at red calibration mark" if irQuad else "Point Wiimote at calibration mark from far enough away")
        if newButtons & cwiid.BTN_MINUS and len(calibrationData[0]):
            if corner == 0:
                corner = len(CALIBRATION_CORNERS)-1
            else:
                corner -= 1
            if len(calibrationData[corner]):
                del calibrationData[corner][-1]
        elif newButtons & (cwiid.BTN_B | NUNCHUK_C) and valid:
            lastCalibrated = time.time()
            z = irQuad.toUnitSquare((0.5,0.5))
            calibrationData[corner].append(z)
            corner = (corner + 1) % len(CALIBRATION_CORNERS)
        n = len(calibrationData[-1])
        if n:
            drawText("Each mark has been calibrated "+("once" if n==1 else "%d times" % n),y=0.7)
            drawText("Press A "+("or C " if 'nunchuk' in wm.state else "")+"button if that's enough",y=0.8)
            if newButtons & cwiid.BTN_A:
                break
            if not flexible:
                leds = computeLEDs(calibrationData,flexible)
                for i in range(4):
                   x = int(leds[i][0] * WINDOW_SIZE[0])
                   y = int((1-leds[i][1]) * WINDOW_SIZE[1])
        pygame.display.flip()
            
    if not running or not len(calibrationData[-1]):
        return False

    ledLocations = computeLEDs(calibrationData,flexible)
    
    CONFIG.ledLocations = ledLocations
    CONFIG.ycorrection = 0
    CONFIG.saveLEDs()
    
    return True


def center():
    CONNECTED_EVENT.wait()

    running = True

    quads = [None,None]

    while running:
        keys = checkQuitAndKeys()
        updateAcceleration(wm.state['acc'])
        surface.fill(BLACK)
        wiimoteWait(0.25)
        if quads[0] is None:
            drawText("Put Wiimote right-side-up pointing at LEDs")
            drawText("Ensure repeatable alignment", y=0.6)
            index = 0
        elif quads[1] is None:
            drawText("Put Wiimote upside-down pointing at LEDs")
            drawText("Ensure same alignment as before ", y=0.6)
            index = 1
        else:
            break
        ir = wm.state['ir_src']
        irQuad = getIRQuad(ir)
        showPoints(ir,irQuad)
        #print(lastAngle, (1-index*2)*math.pi/2)
        if irQuad and abs(lastAngle - (1-index*2)*math.pi/2) < math.pi/4:
            drawText("Press C on Nunchuk or SPACE on keyboard", y=0.7)
            buttons = getButtons(wm.state)
            if buttons & NUNCHUK_C or pygame.K_SPACE in keys:
                quads[index] = irQuad
        pygame.display.flip()

    if not running:
        CENTER_X = 1024/2
        CENTER_Y = 768/2
        return

    sx = 0
    sy = 0

    for i in range(2): 
        for p in quads[0]:
            sx += p[0]
            sy += p[1]
            
    CENTER_X = sx / 8. * 1024.
    CENTER_Y = sy / 8. * 768.
    
    CONFIG.setCenter(wm, (CENTER_X, CENTER_Y))
    CONFIG.saveCalibration()

def demo():
    CONNECTED_EVENT.wait()

    running = True

    while running:
        wiimoteWait(0.25)
        drawText("Press HOME to exit")
        buttons = getButtons(wm.state)
        ir = wm.state['ir_src']
        checkQuitAndKeys()
        surface.fill(BLACK)
        updateAcceleration(wm.state['acc'])
        irQuad = getIRQuad(ir)
        showPoints(ir,irQuad)
        if irQuad:
            screenXY = CONFIG.pointerPosition(irQuad)
            drawCross(screenXY,color=RED)
        pygame.display.flip()

    pygame.quit()

def emulateMouse(mouseName="LightgunMouse",controllerName="WiimoteButtons", horizontal=False,rumble=False):
    global running
    
    size = WINDOW_SIZE or (1920,int(0.5+1920/CONFIG.aspect))
    events = [
        uinput.ABS_X + (0,size[0],0,0),
        uinput.ABS_Y + (0,size[1],0,0),
        uinput.BTN_LEFT,
        uinput.BTN_RIGHT
        ]
        
    events2 = [(uinput.KEY_ESC[0],i) for i in range(uinput.KEY_ESC[1], uinput.KEY_MICMUTE[1]+1)]

    def updateLEDs():
        if horizontal:
            wm.led = cwiid.LED2_ON | cwiid.LED3_ON
        else:
            wm.led = cwiid.LED1_ON | cwiid.LED4_ON

    rumbleStarted = None

    with uinput.Device(events,name=mouseName) as device:
        with uinput.Device(events2,name=controllerName) as device2:
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
                    updateAcceleration(wm.state['acc'])
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
                                press(device2, u)
                            elif released & wii:
                                release(device2, u)
                    elif pressed or released:
                        map = verticalMap if not horizontal else horizontalMap

                        for wii,u in map:
                            dev = device if (u == uinput.BTN_LEFT or u == uinput.BTN_RIGHT) else device2
                            if pressed & wii:
                                press(dev, u)
                                if rumble:
                                    wm.rumble = True
                                    rumbleStarted = time.time()
                            elif released & wii:
                                release(dev, u)

                    if rumble and rumbleStarted and rumbleStarted + RUMBLE_TIME <= time.time():
                        wm.rumble = False
                                
                    if 'nunchuk' in wm.state:
                        def stick(offset,prevOffset,key):
                            if offset < NUNCHUK_DEADZONE-NUNCHUK_HYSTERESIS and prevOffset >= NUNCHUK_DEADZONE-NUNCHUK_HYSTERESIS:
                                release(device2, key)
                            elif offset >= NUNCHUK_DEADZONE:
                                press(device2, key)

                        x,y = wm.state['nunchuk']['stick']

                        stick(x-128,prevNunchukX-128,uinput.KEY_RIGHT)
                        stick(128-x,128-prevNunchukX,uinput.KEY_LEFT)
                        stick(y-128,prevNunchukY-128,uinput.KEY_UP)
                        stick(128-y,128-prevNunchukY,uinput.KEY_DOWN)

                        prevNunchukX, prevNunchukY = x,y

                    if not horizontal:
                        ir = wm.state['ir_src']
                        irQuad = getIRQuad(ir)
                        if irQuad:
                            x,y = CONFIG.pointerPosition(irQuad)
                            x1 = int(size[0]*x)
                            y1 = int(size[1]*(1-y))
                            device.emit(uinput.ABS_X,x1,syn=False)
                            device.emit(uinput.ABS_Y,y1)
                            
            except KeyboardInterrupt:
                pass
            finally:
                for u in uinputPressed:
                    (device if u == uinput.BTN_LEFT or u == uinput.BTN_RIGHT else device2).emit(u, 0)


def connect(backgroundTimeout=0):
    global wm, lastMessage, CENTER_X, CENTER_Y
    wm = None
    t0 = time.time()
    CONNECTED_EVENT.clear()
    while True:
        try:
            wm = cwiid.Wiimote()
            print(getAddress(wm))
            wm.mesg_callback = wiimoteCallback
            wm.enable(cwiid.FLAG_MESG_IFC)
            wm.rpt_mode = cwiid.RPT_IR | cwiid.RPT_BTN | cwiid.RPT_ACC | cwiid.RPT_EXT
            wm.led = cwiid.LED1_ON | cwiid.LED4_ON
            CENTER_X,CENTER_Y = CONFIG.getCenter(wm)
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
    parser.add_argument("-c", "--calibrate", action="store_true", help="Force calibration")
    parser.add_argument("-C", "--center", action="store_true", help="Center calibration for individual Wiimote")
    parser.add_argument("-M", "--measure", action="store_true", help="Calibrate by manual measurement of IR LED positions.")
    parser.add_argument("-w", "--width", type=float, default=1, help="Screen width for measurement calibration in preferred units.")
    parser.add_argument("-d", "--demo", action="store_true", help="Demo")
    parser.add_argument("-f", "--flexible-led-placement", action="store_true", help="Do not assume the top and bottom LED pairs are horizontal")
    parser.add_argument("-o", "--horizontal", action="store_true", help="Horizontal mode (without lightgun)")
    parser.add_argument("-t", "--terminal", action="store_true", help="Use terminal rather than pygame (doesn't work for calibration)")
    parser.add_argument("-m", "--mouse-name", help="Set name of mouse device", default="LightgunMouse")
    parser.add_argument("-b", "--buttons-name", help="Set name of buttons device", default="WiimoteButtons")
    parser.add_argument("-B", "--background-connect", type=float, default=0, help="Connect in background for this many seconds")
    parser.add_argument("command", help="Run this command while simulating a mouse", nargs="?")
    parser.add_argument("-r", "--rumble", action="store_true", help="Rumble on fire")
    args = parser.parse_args()

    try:
        os.mkdir(CONFIG_DIR)
    except:
        pass
    
    thread = threading.Thread(target=connect, args=(args.background_connect,))
    thread.daemon = True
    thread.start()
    
    if args.calibrate or args.measure:
        ledLocations = None
    else:
        ledLocations = CONFIG.ledLocations

    if not args.terminal and (not args.background_connect or not ledLocations or args.center):
        pygame.init()
        atexit.register(pygame.quit)
        WINDOW_SIZE = getDisplaySize()
        CONFIG.aspect = float(WINDOW_SIZE[0])/WINDOW_SIZE[1]
        MYFONT = pygame.font.SysFont(pygame.font.get_default_font(),int(FONT_SIZE*WINDOW_SIZE[1]))                
        surface = pygame.display.set_mode(WINDOW_SIZE, pygame.FULLSCREEN)
        pygame.mouse.set_visible(False)
        
        running = True
        if not args.background_connect:
            while running and wm is None:
                checkQuitAndKeys()
                surface.fill(BLACK)
                drawText("Press 1+2 on Wii Remote")
                drawText("Make sure Wii is turned off", y=0.7)
                drawText("Press ESC to exit", y=0.8)
                pygame.display.flip()
                CONNECTED_EVENT.wait(0.5)
            if not running:
                sys.exit(0)
    elif not args.background_connect:
        print("Press 1+2 on Wii Remote, making sure Wii is turned off.")
        CONNECTED_EVENT.wait()
        print("Ready.")
        
    def cal():
        CONNECTED_EVENT.wait()
        if args.center:
            try:
                del calibrationFileData[getAddress(wm)]
            except KeyError:
                pass
        if not args.calibrate:
            return measure(flexible=args.flexible_led_placement,screenWidth=args.width)
        else:
            return calibrate(flexible=args.flexible_led_placement)

    if args.calibrate or args.measure:
        if cal():
            demo()
    elif args.center:
        center()
    else:
        if ledLocations is None:
            if not cal():
                print("Missing calibration map")
                sys.exit(1)
        running = True
        if args.demo:
            demo()
        else:
            if not args.terminal:
                pygame.quit()
            if args.command:
                thread = threading.Thread(target=run, args=(args.command,))
                thread.daemon = True
                thread.start()
            emulateMouse(mouseName=args.mouse_name,controllerName=args.buttons_name,horizontal=args.horizontal,rumble=args.rumble)
