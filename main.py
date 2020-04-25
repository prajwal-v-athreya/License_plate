import numpy as np
import tensorflow as tf
import cv2 as cv
from collections import OrderedDict
import math

x1 = (1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36)
x2 = (1,2,3,4,5,6,7,8,9,'A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z',0)
x_cords = np.array([])
y_cords = np.array([])
chars = np.array([])
right_cords = np.array([])
bottom_cords = np.array([])
dict = {}
licence_plate = np.array([])
for A, B in zip(x1, x2):
    dict[A] = B
font = cv.FONT_HERSHEY_SIMPLEX
# Read the graph.
with tf.gfile.FastGFile('frozen_inference_graph.pb', 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())

with tf.Session() as sess:
    # Restore session
    sess.graph.as_default()
    tf.import_graph_def(graph_def, name='')
    # Read and preprocess an image.
    img = cv.imread('sample/Sample_2.jpg') #Input image
    img = cv.resize(img, (int(img.shape[1]), int(img.shape[0])))
    rows = img.shape[0]
    cols = img.shape[1]

    inp = img[:, :, [2, 1, 0]]  # BGR2RGB

    # Run the model
    out = sess.run([sess.graph.get_tensor_by_name('num_detections:0'),
                    sess.graph.get_tensor_by_name('detection_scores:0'),
                    sess.graph.get_tensor_by_name('detection_boxes:0'),
                    sess.graph.get_tensor_by_name('detection_classes:0')],
                   feed_dict={'image_tensor:0': inp.reshape(1, inp.shape[0], inp.shape[1], 3)})

    # Visualize detected bounding boxes.
    num_detections = int(out[0][0])
    for i in range(num_detections):
        classId = int(out[3][0][i])
        score = float(out[1][0][i])
        bbox = [float(v) for v in out[2][0][i]]

        if score > 0.5:
            x = bbox[1] * cols
            y = bbox[0] * rows
            right = bbox[3] * cols
            bottom = bbox[2] * rows
            x_cords = np.append(x_cords,int(x))
            y_cords = np.append(y_cords,int(y))
            right_cords = np.append(right_cords,int(right))
            bottom_cords = np.append(bottom_cords,int(bottom))
            chars = np.append(chars,str(dict[classId]))
            cv.rectangle(img, (int(x), int(y)), (int(right), int(bottom)), (125, 255, 51), thickness=2)
            cv.putText(img, str(dict[classId]), (int(x), int(y) - 5), font, 0.8, (0,0,255), 2, cv.LINE_AA)



cords = []
cords_line_2 = []

j = 0
dtype =[('x',int),('y',int),('right',int),('bottom',int),('char','U10')]
for i in x_cords:
    cords.append((x_cords[j],y_cords[j],right_cords[j],bottom_cords[j],chars[j]))
    j += 1

cords = np.array(cords,dtype = dtype)
cords_y = np.sort(cords,order = ['y'])

dist = abs(int((cords_y[0][1]+cords_y[0][3])/2) - int((cords_y[len(cords_y) - 1][1]+cords_y[len(cords_y) - 1][3])/2))

line_1 = int((cords_y[0][1]+cords_y[0][3])/2)
line_2 = int((cords_y[len(cords_y) - 1][1]+cords_y[len(cords_y) - 1][3])/2)

cv.line(img,(0,line_1),(img.shape[1],line_1),(255,255,255),2)
cv.line(img,(0,line_2),(img.shape[1],line_2),(255,255,255),2)

licence_plate = []

cords_x = np.sort(cords, order = ['x'])

line_1_chars = []
line_2_chars = []
FINAL_no = np.array([])

if (dist < 10):
    no_lines = 1
    licence_plate = np.array([])
    for i in cords_x:
        licence_plate = np.append(licence_plate,i[4])
        FINAL_no = licence_plate

elif (dist >= 10):
    no_lines = 2
    j = 0
    new_dtype = [('x',int),('y',int),('char','U10')]
    for i in cords:
        print (i)
        distance_1 = abs((i[1]+i[3])/2 - line_1)
        distance_2 = abs((i[1]+i[3])/2 - line_2)
        if (distance_1 > distance_2):
            line_2_chars.append((i[0],i[1],i[4]))

        elif (distance_1 < distance_2):
            line_1_chars.append((i[0],i[1],i[4]))

    line_1_chars = np.array(line_1_chars,dtype=new_dtype)
    line_1_chars = np.sort(line_1_chars,order='x')
    line_1_chars = line_1_chars.tolist()
    line_2_chars = np.array(line_2_chars,dtype=new_dtype)
    line_2_chars = np.sort(line_2_chars,order='x')
    line_2_chars = line_2_chars.tolist()

    licence_plate = line_1_chars + line_2_chars

    for i in licence_plate:
        FINAL_no = np.append(FINAL_no,i[2])



string = ''
print ('######################################')
for i in FINAL_no:
    string += i
print ('License Plate Number: ',string)
print ('######################################')

cv.imshow('Results', img)
cv.waitKey()
