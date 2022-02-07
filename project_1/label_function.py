import cv2
import numpy as np
import matplotlib.pyplot as plt


def generate_empty_image():
    return np.ones(shape=(512,512,3), dtype=np.int16)

def add_label_and_rectangle(image,object_type,x,y,l,w):
    GOLDFISH = 0
    RABBIT = 1
    WHALE = 2
    if object_type == GOLDFISH:
        label = "Good"
        text_color = (0, 255, 0)
    elif object_type == RABBIT:
        label = "Ugly"
        text_color = (255, 0, 0)
    elif object_type == WHALE:
        label = "Bad"
        text_color = (165,100,42)
    else:
        label = " "
        text_color = (255,255,255)
    cv2.putText(img=image, text=label, org=(x, y), fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=1, color=text_color,thickness=1)
    image = cv2.rectangle(image, (x,y), (x+w,y+l), text_color, 2)






img = generate_empty_image()

add_label_and_rectangle(img,2,100,300,100,50)

plt.imshow(img)
plt.show()