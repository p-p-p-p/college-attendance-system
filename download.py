import cv2
import numpy as np
from deepface import DeepFace
import os
def deep_data_extract(img):
    embedding=None
    faces=[]
    facial_data=[]
    try:
        embedding = DeepFace.represent(img, model_name='Facenet',detector_backend='ssd')
        if embedding:
          for  i in range(len(embedding)):
            x, y, w, h = embedding[i]['facial_area']['x'], embedding[i]['facial_area']['y'], embedding[i]['facial_area']['w'], embedding[i]['facial_area']['h']
            x1, y1, x2, y2 = x, y, x+w, y+h
            faces.append((x1, y1, x2, y2 ))
            facial_data.append(embedding[i]['embedding'])
    except:
        pass
    return faces,facial_data


img=cv2.imread("p.jpg")

faces,facial_data=deep_data_extract(img)
print(faces[0],facial_data[0])
cv2.imshow("img",img)
cv2.waitKey(0)
