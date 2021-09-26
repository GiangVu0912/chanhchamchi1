import re
import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
import os
from tensorflow.keras.preprocessing import image as tfimage
import uuid

### GLOBAL Variable ###
HOME_FOLDER = os.getcwd()

#YOLO MODEL#
MODEL = 'yolo\yolov3-face.cfg'
WEIGHT = 'yolo\yolov3-wider_16000.weights'

net = cv2.dnn.readNetFromDarknet(MODEL, WEIGHT)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

#CLASSIFIED MODEL
MODEL_MEMBER_PREDICT_PATH = 'model\\facerec_checkpoint.h5'
model_member_predict = tf.keras.models.load_model(MODEL_MEMBER_PREDICT_PATH)
CLASS_NAMES = ['giang', 'hanh', 'loc', 'nam']

#EMOTIONAL MODEL 
MODEL_EMOTION_RECOGNIZE_PATH = 'model\\FE_ResNet.h5'
EMOTION_CLASS_NAMES = ['angry', 'happy', 'neutral', 'sad']
MODEL_EMOTION_RECOGNIZE = tf.keras.models.load_model(MODEL_EMOTION_RECOGNIZE_PATH)

def predict_emotion(image_path): 
    #INPUT: PATH OF AN FACE
    #OUTPUT: PREDICTION OF EMOTION   
    img_array = tfimage.load_img(image_path,target_size=(48,48))
    img_array = tfimage.img_to_array(img_array)
    img_array  = np.expand_dims(img_array, axis=0) #predict nhận theo batch (1,224,224,3)
    print(img_array.shape)
    prediction = MODEL_EMOTION_RECOGNIZE.predict(img_array)
    print('prediction:', prediction)
    index = prediction.argmax()   

    return prediction[0], index

def detect_bounding_box(input_img):
    #INPUT: RAW IMAGE CAPTURE FROM WEBCAM FRAME 
    #OUTPUT: Final boxs include all bounding box, once topleft_x, topleft_y, width, height

    IMG_WIDTH, IMG_HEIGHT = 416, 416
    blob = cv2.dnn.blobFromImage(input_img, 
                                1/255, (IMG_WIDTH, IMG_HEIGHT),
                                [0, 0, 0], 1, crop=False)
    # Set model input
    net.setInput(blob)

    # Define the layers that we want to get the outputs from
    output_layers = net.getUnconnectedOutLayersNames()

    # Run 'prediction'
    outs = net.forward(output_layers)

    # Define Boundbox
    frame = input_img.copy()
    frame_height = frame.shape[0] #480
    frame_width = frame.shape[1] #640

    # Scan through all the bounding boxes output from the network and keep only
    # the ones with high confidence scores. Assign the box's class label as the
    # class with the highest score.

    confidences = []
    boxes = []

    # Each frame produces 3 outs corresponding to 3 output layers
    for out in outs:
            # One out has multiple predictions for multiple captured objects.
        for detection in out:
            confidence = detection[-1]
            # Extract position data of face area (only area with high confidence)
            if confidence > 0.5:
                center_x = int(round(detection[0] * frame_width))
                center_y = int(round(detection[1] * frame_height))
                width    = int(round(detection[2] * frame_width))
                height   = int(round(detection[3] * frame_height))
                #print(center_x,center_y,width,height)
                # Find the top left point of the bounding box
                topleft_x = center_x - width//2
                topleft_y = center_y - height//2

                #print(topleft_x, topleft_y)
                confidences.append(float(confidence))
                boxes.append([topleft_x, topleft_y, width, height])


    # Perform non-maximum suppression to eliminate 
    # redundant overlapping boxes with lower confidences.
    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    # print(len(indices))

    result = frame.copy()
    final_boxes = []

    tmp_str = 'number of faces detected:' + str(len(indices))
    for i in indices:    
        i = i[0]
        box = boxes[i]
        final_boxes.append(box)
    return final_boxes

def predict_image(image_path):
    #INPUT: IMAGE PATH
    #OUTPUT: prediction, index, emotion
    print('Predict function')

    #PREDICT EMOTION
    
    prediction, index = predict_emotion(image_path)
    emotion = EMOTION_CLASS_NAMES[index]
    proba_emotion = prediction[index]

    #PREDICT 
    img_array, label = load_and_preprocess_image(image_path)
    img_array  = np.expand_dims(img_array, axis=0) #predict nhận theo batch (1,224,224,3)
    
    prediction = model_member_predict.predict(img_array)
    index = prediction.argmax()    
    
    return prediction[0], index, emotion, proba_emotion

def save_file_to_tmp(img):
    tmp_path = os.path.normpath('tmp')
    os.chdir(tmp_path)
    id = uuid.uuid1()
    fname_tmp = str(id) + '.jpg' 
    cv2.imwrite(fname_tmp,img)
    print("Image saved successful: ",fname_tmp)
    os.chdir(HOME_FOLDER) # Trả về thư mục gốc
    tmp_path = 'tmp\\'+ fname_tmp

    return str(tmp_path)

def preprocess_image(image):
    image = tf.image.decode_png(image, channels=3)
    image = tf.image.resize(image, [224, 224])
    image /= 255.0  # normalize to [0,1] range
    return image

def load_and_preprocess_image(path, label=None):
    image_raw = tf.io.read_file(path)
    image = preprocess_image(image_raw)
    return image, label

