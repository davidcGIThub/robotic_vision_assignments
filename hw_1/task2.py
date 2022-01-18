from operator import mod
import threading
import numpy as np
import cv2 as cv
from PIL import Image, ImageTk
from tkinter import Tk, Frame, Button, BOTH, Label, Scale, Radiobutton       # Graphical User Inetrface Stuff
from tkinter import font as tkFont
import tkinter as tk
import copy

camera = cv.VideoCapture(0)
width = int(camera.get(cv.CAP_PROP_FRAME_WIDTH))
height = int(camera.get(cv.CAP_PROP_FRAME_HEIGHT))
videoout = cv.VideoWriter('./task2.avi', cv.VideoWriter_fourcc(*'XVID'), 25, (width, height))   # Video format

# Button Definitions
ORIGINAL = 0
BINARY = 1
CANNY = 2
CORNER = 3
LINE = 4
ABSDIFF = 5

def cvMat2tkImg(arr):           # Convert OpenCV image Mat to image for display
    rgb = cv.cvtColor(arr, cv.COLOR_BGR2RGB)
    img = Image.fromarray(rgb)
    return ImageTk.PhotoImage(img)

class App(Frame):
    def __init__(self, winname='OpenCV'):       # GUI Design

        self.root = Tk()
        self.stopflag = True
        self.buffer = np.zeros((height, width, 3), dtype=np.uint8)

        global helv18
        helv18 = tkFont.Font(family='Helvetica', size=18, weight='bold')
        # print("Width",windowWidth,"Height",windowHeight)
        self.root.wm_title(winname)
        positionRight = int(self.root.winfo_screenwidth() / 2 - width / 2)
        positionDown = int(self.root.winfo_screenheight() / 2 - height / 2)
        # Positions the window in the center of the page.
        self.root.geometry("+{}+{}".format(positionRight, positionDown))
        self.root.wm_protocol("WM_DELETE_WINDOW", self.exitApp)
        Frame.__init__(self, self.root)
        self.pack(fill=BOTH, expand=1)
        # capture and display the first frame
        ret0, frame = camera.read()
        image = cvMat2tkImg(frame)
        self.panel = Label(image=image)
        self.panel.image = image
        self.panel.pack(side="top")
        # buttons
        global btnStart
        btnStart = Button(text="Start", command=self.startstop)
        btnStart['font'] = helv18
        btnStart.pack(side='right', pady = 2)
        # sliders
        global Slider1, Slider2
        Slider2 = Scale(self.root, from_=0, to=255, length= 255, orient='horizontal')
        Slider2.pack(side='right')
        Slider2.set(255)
        Slider1 = Scale(self.root, from_=0, to=255, length= 255, orient='horizontal')
        Slider1.pack(side='right')
        Slider1.set(0)
        # radio buttons
        global mode
        mode = tk.IntVar()
        mode.set(ORIGINAL)
        Radiobutton(self.root, text="Original", variable=mode, value=ORIGINAL).pack(side = 'left', pady = 4)
        Radiobutton(self.root, text="Binary", variable=mode, value=BINARY).pack(side = 'left', pady = 4)
        Radiobutton(self.root, text="Edge", variable=mode, value=CANNY).pack(side = 'left', pady = 4)
        Radiobutton(self.root, text="Corner", variable=mode, value=CORNER).pack(side = 'left', pady = 4)
        Radiobutton(self.root, text="Line", variable=mode, value=LINE).pack(side='left', pady=4)
        Radiobutton(self.root, text="Abs Diff", variable=mode, value=ABSDIFF).pack(side='left', pady=4)
        # threading
        self.stopevent = threading.Event()
        self.thread = threading.Thread(target=self.capture, args=())
        self.thread.start()

    def capture(self):
        while not self.stopevent.is_set():
            if not self.stopflag:
                ret0, frame = camera.read()
                gray_image = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
                low_threshold = Slider1.get()
                high_threshold = Slider2.get()

                if mode.get() == BINARY:
                    gray_image[gray_image < low_threshold] = 0
                    gray_image[gray_image > high_threshold] = 0
                    gray_image[gray_image > 0] = 255
                    modified_image = cv.cvtColor(gray_image, cv.COLOR_GRAY2BGR) 

                elif mode.get() == CANNY:
                    edges = cv.Canny(frame,low_threshold, high_threshold)
                    modified_image = cv.cvtColor(edges, cv.COLOR_GRAY2BGR)

                elif mode.get() == CORNER:
                    max_corners = high_threshold
                    quality_level = low_threshold/510.0
                    if quality_level == 0:
                        quality_level = 1/510.0
                    if quality_level == 1:
                        quality_level = 244/510.0
                    min_distance = 10
                    corners = cv.goodFeaturesToTrack(gray_image,max_corners,quality_level,min_distance)
                    corners = np.int0(corners)
                    modified_image = copy.deepcopy(frame)
                    for i in corners:
                        x,y = i.ravel()
                        color = (0,0,255)
                        radius = 3
                        thickness = -1
                        cv.circle(modified_image,(x,y),radius,color,thickness)

                elif mode.get() == LINE:
                    edge_low_threshold = 60
                    edge_high_threshold = 130
                    edges = cv.Canny(frame, edge_low_threshold, edge_high_threshold)
                    rho = 1 # distance resolution of the accumulator in radians
                    alpha = np.pi/180 #angle resolution of the accumulator in radians
                    lines = cv.HoughLines(edges, rho, alpha, high_threshold)
                    if lines is None:
                        modified_image = frame
                    else:
                        modified_image = copy.deepcopy(frame)
                        for line in lines:
                            r, theta = line[0]
                            x_middle = np.cos(theta)*r
                            y_middle = np.sin(theta)*r
                            step_size = 1000
                            x_left = int(x_middle - step_size*np.sin(theta))
                            x_right = int(x_middle + step_size*np.sin(theta))
                            y_bottom = int(y_middle + step_size*np.cos(theta))
                            y_top = int(y_middle - step_size*np.cos(theta))
                            color = (0,0,255)
                            thickness = 2
                            cv.line(modified_image, (x_left,y_bottom), (x_right,y_top), color, thickness)
                        
                elif mode.get() == ABSDIFF:
                    difference = np.zeros((height, width, 3), dtype=np.uint8)
                    cv.absdiff(self.buffer, frame, difference)
                    modified_image = difference
                else:
                    modified_image = frame
                self.buffer = frame
                image = cvMat2tkImg(modified_image)
                self.panel.configure(image=image)
                self.panel.image = image
                videoout.write(modified_image)

    def startstop(self):        #toggle flag to start and stop
        if btnStart.config('text')[-1] == 'Start':
            btnStart.config(text='Stop')
        else:
            btnStart.config(text='Start')
        self.stopflag = not self.stopflag

    def run(self):              #run main loop
        self.root.mainloop()

    def exitApp(self):          #exit loop
        self.stopevent.set()
        self.root.quit()


app = App()
app.run()
#release the camera
camera.release()
cv.destroyAllWindows()