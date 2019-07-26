
from matplotlib import pyplot as plt
import numpy as np
import cv2
import requests
import json

input_img_file = 'org.jpg'
org_img = cv2.imread(input_img_file)

org_img = cv2.resize(org_img, (1000, 850))#[:,:,::-1]
cv2.imwrite("org.jpg", org_img)
org_img = cv2.imread(input_img_file)[:,:,::-1]
#[:,:,::-1]

"""
fig = plt.figure()
fig.set_size_inches(10, 7.5)
plt.title("The original image")
plt.imshow(org_img)
plt.show()
"""



url = 'http://localhost:5000/model/predict'
from time import time

def get_pose(input_img):

    files = {'file': ('image.jpg',open(input_img,'rb'), 'images/jpeg')}
    result = requests.post(url, files=files).json()
   
    return result



preds = get_pose(input_img_file)
#print(json.dumps(preds, indent=2))



CocoColors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], 
              [170, 255, 0], [85, 255, 0], [0, 255, 0], [0, 255, 85], 
              [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], 
              [0, 0, 255], [85, 0, 255], [170, 0, 255], [255, 0, 255], 
              [255, 0, 170], [255, 0, 85]]


def draw_pose(preds, img):
    
    humans = preds['predictions']
    for human in humans:
        pose_lines = human['pose_lines']
        arr_x = []
        arr_y = []
        for i in range(len(pose_lines)):
            line = pose_lines[i]['line']
            cv2.line(img, (line[0], line[1]), (line[2], line[3]), CocoColors[i], 3)        
            arr_x.append(line[0])
            arr_y.append(line[1])
            arr_x.append(line[2])
            arr_y.append(line[3])
        arr_x.sort()
        arr_y.sort()
        x_min = arr_x[0]
        y_min = arr_y[0]
        x_max = arr_x[len(arr_x)-1]
        y_max = arr_y[len(arr_y)-1]
        cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (255, 100, 75), 2)

    

fig = plt.figure()
fig.set_size_inches(18.5, 10.5)

plt.subplot(1, 3, 1)
plt.imshow(org_img)
plt.title("The original image")

pose_img = np.zeros(org_img.shape, dtype=np.uint8)
draw_pose(preds, pose_img)
plt.subplot(1, 3, 2)
plt.imshow(pose_img)
plt.title("The detected poses")

overlaid_img = org_img.copy()
draw_pose(preds, overlaid_img)
plt.subplot(1, 3, 3)
plt.imshow(overlaid_img)
plt.title("Poses overlaid on original image")

plt.show()

