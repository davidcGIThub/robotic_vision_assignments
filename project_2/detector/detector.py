import numpy as np
import cv2
import sys
import os
import re
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from rotations import Euler2Rotation
# plt.ion()

import matplotlib
# matplotlib.rcParams['backend'] = 'Qt5Agg'
# print(plt.matplotlib.rcParams['figure.dpi'])
# if plt.get_backend() == 'Qt5Agg':
#     from matplotlib.backends.qt_compat import QtWidgets
#     qApp = QtWidgets.QApplication(sys.argv)
#     # print(qApp.desktop().physicalDpiX())
#     plt.matplotlib.rcParams['figure.dpi'] = qApp.desktop().physicalDpiX()

class BallDetector:
    def __init__(self):
        self.prev_circle = None
        self.prev_box = None
        self.border = 5
        self.border_prev = 40
        self.all_pts = []
        self.plot_pts = []
        self.R_cam_2_catch = Euler2Rotation(np.pi,0,0)
        self.T_cam_2_catch = np.array([[2-10, 30, 22]]).T
        self.H_cam_2_catch = np.vstack([np.hstack([self.R_cam_2_catch, self.T_cam_2_catch]),np.array([0,0,0,1])])

        self.left_start = (364,99)
        self.right_start = (278,104)
        self.first = True
        self.num_not_detect = 0

        with open('paramsL.npz','rb') as f:
            self.mtx_L = np.load(f)
            self.dist_L = np.load(f)

        with open('paramsR.npz','rb') as f:
            self.mtx_R = np.load(f)
            self.dist_R = np.load(f)

        with open('paramsST.npz','rb') as f:
            self.rot = np.load(f)
            self.tran = np.load(f)
            self.e_mtx = np.load(f)
            self.f_mtx = np.load(f)

        with open('params_rect.npz', 'rb') as f:
            self.R1 = np.load(f)
            self.R2 = np.load(f)
            self.P1 = np.load(f)
            self.P2 = np.load(f)
            self.Q = np.load(f)

        with open('params_map.npz', 'rb') as f:
            self.mapx1 = np.load(f)
            self.mapy1 = np.load(f)
            self.mapx2 = np.load(f)
            self.mapy2 = np.load(f)

        # self.video_out = cv2.VideoWriter('./baseball_detection.avi', cv2.VideoWriter_fourcc(*'XVID'), 10, (640, 480))
    
    def mouse_callback_L(self,event,x,y,flags,param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.left_start = (x,y)
            print("Left start: ", x,y)
            cv2.destroyAllWindows()

    def mouse_callback_R(self,event,x,y,flags,param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.right_start = (x,y)
            print("Right start: ", x,y)
            cv2.destroyAllWindows()

    def set_start_location(self,left, right):
        while True:
            try:
                cv2.imshow('left',left)
                cv2.setMouseCallback('left',self.mouse_callback_L)
                cv2.waitKey(1)
                cv2.getWindowProperty('left',0)
            except:
                break
        
        while True:
            try:
                cv2.imshow('right',right)
                cv2.setMouseCallback('right',self.mouse_callback_R)
                cv2.waitKey(1)
                cv2.getWindowProperty('right',0)
            except:
                break
        

    def detect_ball(self, img, old_img,side,x_prev,y_prev,w_prev,h_prev):
        
        X = x_prev
        Y = y_prev
        R = None
        W = w_prev
        H = h_prev

        gray_img = cv2.cvtColor(img.copy(),cv2.COLOR_BGR2GRAY)
        old_img = cv2.cvtColor(old_img.copy(),cv2.COLOR_BGR2GRAY)
        # display_img = img.copy()
        
        X_min = X - W - self.border_prev
        X_max = X + W + self.border_prev
        Y_min = Y - H - self.border_prev
        Y_max = Y + H + self.border_prev

        if X_min < 0:
            X_min = 0
        if Y_min < 0:
            Y_min = 0
        if X_max > 640:
            X_max = 640
        if Y_max > 480:
            Y_max = 480
        gray_img = gray_img[Y_min : Y_max, X_min : X_max]
        old_img = old_img[Y_min : Y_max, X_min : X_max]

        
        gray_img = cv2.absdiff(gray_img,old_img)
            
        # cv2.imshow('absdiff',gray_img)
        gray_img = cv2.blur(gray_img,(3,3))
        if np.linalg.norm([H,W]) > 60:
            gray_img = cv2.blur(gray_img,(15,15))
        # cv2.imshow('blurr1',gray_img)
        

        thresh, gray_img = cv2.threshold(gray_img,12,255,0)
        # cv2.imshow('thresh1',gray_img)
        

        # kernel = np.ones((3,3), np.uint8)
        # gray_img = cv2.dilate(gray_img,kernel)
        gray_img = cv2.blur(gray_img,(3,3))
        gray_img = cv2.blur(gray_img,(5,5))
        # cv2.imshow('blurr2',gray_img)

        thresh, gray_img = cv2.threshold(gray_img,100,255,0)
        # cv2.imshow('thresh2',gray_img)


        kernel = np.ones((10,10), np.uint8)
        gray_img = cv2.dilate(gray_img,kernel)
        
        nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(gray_img, connectivity=8)
        sizes = stats[1:, -1]; nb_components = nb_components - 1
        for i in range(0, nb_components):
                if sizes[i] <= 100:
                    gray_img[output == i + 1] = 0
                    
        # cv2.imshow("before",gray_img)
        if nb_components > 1:
            # if np.abs(max(sizes) - min(sizes)) > 200:
            for ii in range(0,nb_components):
                if np.abs(max(sizes)-sizes[ii]) > 200:
                    gray_img[output == ii + 1] = 0
        # cv2.imshow("after",gray_img)   
            
        contours, hierarchy= cv2.findContours(gray_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        
        if len(contours) > 0:
            min_y = np.inf
            min_x = np.inf
            max_y = 0
            max_x = 0

            # get combined bounding box
            for (i,c) in enumerate(contours):
                x,y,w,h= cv2.boundingRect(c)
                x = X_min + x
                y = Y_min + y
                if x < min_x:
                    min_x = x
                if y < min_y:
                    min_y = y
                if x+w > max_x:
                    max_x = x+w
                if y+h > max_y:
                    max_y = y+h
            
            if min_x - self.border < 0:
                min_x += self.border
            if min_y - self.border < 0:
                min_y += self.border
            if max_x + self.border> 640:
                max_x -= self.border
            if max_y + self.border > 480:
                max_y -= self.border
                
            cropped_contour= img[min_y-self.border : max_y+self.border, min_x-self.border:max_x+self.border]
            W = max_x - min_x
            H = max_y - min_y
        else:
            return X,Y,R,W,H
        # cv2.rectangle(display_img,(min_x,min_y),(max_x,max_y),(0,255,0),2)
        

        cropped_contour = cv2.cvtColor(cropped_contour,cv2.COLOR_BGR2GRAY)
        cropped_contour = cv2.blur(cropped_contour,(7,7))
        # cropped_contour = cv2.Canny(cropped_contour,70,150)
        # print(np.linalg.norm([H,W]))
        # if side == 'L':
        if np.linalg.norm([H,W]) < 60:
            cropped_contour = cv2.adaptiveThreshold(cropped_contour, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 31, 0) 
        # else:
            # cropped_contour = cv2.adaptiveThreshold(cropped_contour, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 101, 0) 
            cropped_contour = cv2.blur(cropped_contour,(7,7))
            thresh, cropped_contour = cv2.threshold(cropped_contour,150,255,0)
            cropped_contour = cv2.blur(cropped_contour,(7,7))
        # else:
        #     if np.linalg.norm([H,W]) < 60:
        #         cropped_contour = cv2.adaptiveThreshold(cropped_contour, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 31, 0) 
        #     # else:
        #         # cropped_contour = cv2.adaptiveThreshold(cropped_contour, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 31, 0) 
        #         cropped_contour = cv2.blur(cropped_contour,(7,7))
        #         thresh, cropped_contour = cv2.threshold(cropped_contour,150,255,0)
        #         cropped_contour = cv2.blur(cropped_contour,(7,7))

        
        # kernel = np.ones((3,3), np.uint8)
        # cropped_contour = cv2.dilate(cropped_contour,kernel)
        cv2.imshow('cropped',cropped_contour)

        i = 100
        len_circles = 0
        
        if side == 'L':
            while len_circles < 1 and i > 10:
                circles = cv2.HoughCircles(cropped_contour, cv2.HOUGH_GRADIENT, 1.5, 100, param1=100, param2=i, maxRadius=40,minRadius=4)
                if circles is not None:
                    len_circles = len(circles[0])
                i -= 1
        else:
            while len_circles < 1 and i > 10:
                circles = cv2.HoughCircles(cropped_contour, cv2.HOUGH_GRADIENT, 1.3, 100, param1=100, param2=i, maxRadius=40,minRadius=4)
                if circles is not None:
                    len_circles = len(circles[0])
                i -= 1

        # print(i)
      
        cropped_contour = cv2.cvtColor(cropped_contour,cv2.COLOR_GRAY2BGR)    
       
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            
            (xc,yc,r) = circles[np.argmax(circles[:,2])]
            # (xc,yc,r) = circles[0]

            # cv2.circle(cropped_contour, (xc, yc), r, (0, 255, 0), 4)
            # cv2.rectangle(cropped_contour, (xc - 5, yc - 5), (xc + 5, yc + 5), (0, 128, 255), -1)

            # cv2.circle(display_img, (x-10+xc, y-10+yc), r, (0, 255, 0), 4)
            # cv2.rectangle(display_img, (x-10+xc - 5, y-10+yc - 5), (x-10+xc + 5, y-10+yc + 5), (0, 128, 255), -1)

            self.prev_circle = (xc,yc,r)
            self.prev_box = (x,y)

            X = min_x-self.border + xc
            Y = min_y-self.border+yc
            R = r

        return X,Y,R,W,H
    
    def rectify_left_point(self,x,y):
        pt = np.array([[[x,y]]]).astype(np.float64)
        L_undist = cv2.undistortPoints(pt, self.mtx_L, self.dist_L, R=self.R1, P=self.P1)
        return L_undist.item(0), L_undist.item(1)

    def rectify_right_point(self,x,y):
        pt = np.array([[[x,y]]]).astype(np.float64)
        R_undist = cv2.undistortPoints(pt, self.mtx_R, self.dist_R, R=self.R2, P=self.P2)
        return R_undist.item(0), R_undist.item(1)

    def run(self,img_L,img_R):
    
        img_L_disp = img_L.copy()
        img_R_disp = img_R.copy()

        if self.first:
            self.old_img_L = img_L
            self.old_img_R = img_R

            self.X = self.left_start[0]
            self.Y = self.left_start[1]
            self.R = 10
            self.H = 20
            self.W = 20

            self.Xr = self.right_start[0]
            self.Yr = self.right_start[1]
            self.Rr = 10
            self.Hr = 20
            self.Wr = 20

            self.first = False
        
        self.X,self.Y,self.R,self.W,self.H = self.detect_ball(img_L, self.old_img_L,'L',self.X,self.Y,self.W,self.H)
        self.Xr,self.Yr,self.Rr,self.Wr,self.Hr = self.detect_ball(img_R, self.old_img_R,'R',self.Xr,self.Yr,self.Wr,self.Hr)

        if self.R is not None:
            cv2.circle(img_L_disp, (self.X, self.Y), self.R, (0, 255, 0), 4)
            cv2.rectangle(img_L_disp, (self.X - 5, self.Y - 5), (self.X + 5, self.Y + 5), (0, 128, 255), -1)
        if self.Rr is not None:
            cv2.circle(img_R_disp, (int(self.Xr), int(self.Yr)), self.Rr, (0, 255, 0), 4)
            cv2.rectangle(img_R_disp, (int(self.Xr) - 5, self.Yr - 5), (int(self.Xr) + 5, int(self.Yr) + 5), (0, 128, 255), -1)
        if self.R is not None and self.Rr is not None:
            xl_rect, yl_rect = self.rectify_left_point(self.X,self.Y)
            xr_rect, yr_rect = self.rectify_right_point(self.Xr,self.Yr)

            pt_disp =np.array([xl_rect,yl_rect, xl_rect - xr_rect])
            pt_transformed = cv2.perspectiveTransform(pt_disp.reshape(1,1,3).astype(np.float64),self.Q)
            self.all_pts.append(pt_transformed)
            
            self.num_not_detect = 0
        else:
            self.num_not_detect += 1
        
        # if self.X < 20 or self.X > 620 or self.Y < 20 or self.Y > 460:
        # if len(self.all_pts) > 40
        if self.num_not_detect > 7:
            self.all_pts = []
            
        cv2.imshow("left",img_L_disp)
        cv2.imshow("right",img_R_disp)
        cv2.waitKey(250)
        self.old_img_L = img_L
        self.old_img_R = img_R

if __name__ == "__main__":

    detector = BallDetector()
    file = "./baseball sequence/Sequence3/"
    # left_pic = cv2.imread(file + "L/0.png")
    # right_pic = cv2.imread(file+"R/0.png")
    # detector.set_start_location(left_pic,right_pic)

    directory_L = file+"L"
    directory_R = file+"R"
    files = os.listdir(directory_L)
    # files.sort(key=int)
    files.sort(key=lambda test_string : list(map(int, re.findall(r'\d+', test_string)))[0])
    first = True
    i = 1
    for file in files:
        if file.endswith(".jpg") or file.endswith(".png"):
            
            img_L = cv2.imread(os.path.join(directory_L,file))
            img_R = cv2.imread(os.path.join(directory_R,file))

            detector.run(img_L,img_R)
            print(len(detector.all_pts))

    fig = plt.figure(1)
    ax = fig.add_subplot(projection='3d')
    pts = np.array(detector.all_pts).reshape(-1,3)
    pts = np.hstack([pts,np.ones((len(pts),1))])
    pts = (detector.H_cam_2_catch @ pts.T).T
    pts = pts[:,0:3]
    # print(pts)
    ax.plot3D(pts[:,0],pts[:,1],pts[:,2])
    # R = 
    frames = [4,9,14,19,24,28]
    plt.figure(2)
    plt.scatter(pts[frames,2],pts[frames,1])
    coef = np.polyfit(pts[frames,2],pts[frames,1],2)
    x=np.linspace(pts[0,2],0,500)
    y = coef.item(0)*x**2 + coef.item(1)*x + coef.item(2)
    plt.plot(x,y)
    plt.ylim([-20,80])
    plt.xlim([min(x),max(x)])
    plt.grid(True)
    plt.xlabel("Z")
    plt.ylabel("Y")
    plt.title('Z-Y Plot')
    plt.savefig("task3A.jpg")
    print("final Y = ", y[-1])

    plt.figure(3)
    plt.scatter(pts[frames,2],pts[frames,0])
    coef = np.polyfit(pts[frames,2],pts[frames,0],1)
    x=np.linspace(pts[0,2],0,500)
    y = coef.item(0)*x + coef.item(1)
    plt.plot(x,y)
    plt.ylim([-50,50])
    plt.xlim([min(x),max(x)])
    plt.grid(True)
    plt.xlabel("Z")
    plt.ylabel("X")
    plt.title('Z-X Plot')
    plt.savefig("task3B.jpg")
    print("final X = ", y[-1])
    plt.show()
    # detector = BallDetector()
    # detector.run('R')