
from matplotlib import pyplot as plt
import numpy as np
import cv2
import requests
import json
from time import time

#input_img_file = 'assets/test1.jpg'
url = 'http://localhost:5001/model/predict'

cap = cv2.VideoCapture("video.mp4")
def get_pose(input_img):
    files = {'file': ('image.jpg',open(input_img,'rb'), 'images/jpeg')}
    #files = {'file': ('image.jpg',input_img, 'images/jpeg')}
    result = requests.post(url, files=files).json()
    return result
start = time()
proces_frame = True
while(cap.read()):
    try:
        _, org_img = cap.read()
        _, org_img = cap.read()
        org_img = cv2.resize(org_img, (150, 150))
        aa = time()
        cv2.imwrite("out.jpg", org_img)
        #org_img = cv2.imread(input_img_file)[:,:,::-1]

        #preds = get_pose(input_img_file)

        preds = get_pose('out.jpg')
        #preds = get_pose(org_img)
        print("tiempo guardado y leido:  ", time()-aa)
        #print(json.dumps(preds, indent=2))

        CocoColors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], 
                    [170, 255, 0], [85, 255, 0], [0, 255, 0], [0, 255, 85], 
                    [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], 
                    [0, 0, 255], [85, 0, 255], [170, 0, 255], [255, 0, 255], 
                    [255, 0, 170], [255, 0, 85]]

        humans = preds['predictions']
        #img = np.zeros(org_img.shape, dtype=np.uint8)
        img = org_img
        for human in humans:
            pose_lines = human['pose_lines']
            arr_x = []
            arr_y = []
            for i in range(len(pose_lines)):
                line = pose_lines[i]['line']
                cv2.line(img, (line[0], line[1]), (line[2], line[3]), CocoColors[i], 1)
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
        #img = added_image = cv2.addWeighted(org_img,0.4,img,0.1,0)
        img = cv2.resize(img, (600, 500))
        cv2.imshow("frame", img )
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    except:
        print("salida")
        print("tiempo total: ", time()-start)
        break

cap.release()
print("tiempo total sin mostrar imagenes (slo guardarlas y leerlas): ", time()-start)
cv2.destroyAllWindows()

        

   

