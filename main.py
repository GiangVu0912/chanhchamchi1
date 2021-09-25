#### MAIN FILE FOR PROJECT ###

### IMPORT LIBRARY ###   
import cv2
import os 
import numpy as np
import uuid

### GLOBAL VARIABLE
MODEL = 'yolo\yolov3-face.cfg'
WEIGHT = 'yolo\yolov3-wider_16000.weights'

net = cv2.dnn.readNetFromDarknet(MODEL, WEIGHT)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
HOME_FOLDER = os.getcwd()
### 

# Main Function: Open Webcam And Crop Image 
# Press c to capture
# Press ESC to quit
# Remember to in put yourname line 50
def open_webcam():
    cap = cv2.VideoCapture(0)

    ### Check if the webcam is opened correctly
    if not cap.isOpened():
        raise IOError("Cannot open webcam")

    while True:
        ret, frame = cap.read()

        # frame is now the image capture by the webcam (one frame of the video)
        cv2.imshow('Input', frame)

        c = cv2.waitKey(1)
            # Break when pressing ESC
        #print('Button clicked:',c)
        if c == 27:
            break

        if c == ord('c'): #button 'c'
            # Capture picture and save
            print('Prepare for model')
            len(frame)
            captured_image = frame
            #captured_image = cv2.cvtColor(captured_image, cv2.COLOR_BGR2RGB )  # Conver image to RGB color
            print('Shape of captured image',captured_image.shape) # (y,x,3) (cao, dài, channels)
            try:            
                name = 'nam'    # <== Change here to run
                directory_path = 'member_photo\\' + name     ## Sẽ có 1 hàm input để thu thập sau           
                path = os.path.normpath(directory_path)
                print(path)
                detect_and_crop_faces(captured_image,str(path))
            except ValueError:
                cap.release()
                cv2.destroyAllWindows()
                print(ValueError)

    cap.release()
    cv2.destroyAllWindows()

def detect_and_crop_faces(input_img,directory_path): 
    #Input: image capture from webcam
    #Define Bounding Box
    #Crop The Face
    #Output: save this face
    #image_path = 'member_photo\\nam\Input_screenshot_23.09.202-2.png'
    #input_img = cv2.imread(image_path)

    IMG_WIDTH, IMG_HEIGHT = 416, 416

    # Making blob object from original image
    blob = cv2.dnn.blobFromImage(input_img, 
                                    1/255, (IMG_WIDTH, IMG_HEIGHT),
                                    [0, 0, 0], 1, crop=False)

    # Set model input
    net.setInput(blob)

    # Define the layers that we want to get the outputs from
    output_layers = net.getUnconnectedOutLayersNames()

    # Run 'prediction'
    outs = net.forward(output_layers)

    # print('Output length:',len(outs))

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

        # Extract position data
        x_left = box[0]
        y_top = box[1]
        width = box[2]
        height = box[3] 
        
        #Crop image and write it to folder
        crop_img = input_img[y_top:y_top+height, x_left:x_left+width]

        #Write To folder
        os.chdir(directory_path)
        #Write it
        id = uuid.uuid1()
        fname_tmp = str(id) + '.jpg' #Tên thành viên_id
        cv2.imwrite(fname_tmp,crop_img)
        print("Image saved successful: ",fname_tmp)
        os.chdir(HOME_FOLDER) # Trả về thư mục gốc

        # # Draw bouding box with the above measurements
        # ### YOUR CODE HERE
        # cv2.rectangle(result, (x_left,y_top), (x_left+width,y_top+height), (0, 0, 255), 1)		
    
        # # Display text about confidence rate above each box
        # text = f'{confidences[i]:.2f}'
        # ### YOUR CODE HERE
        # result = cv2.putText(result,"{:.2f}%".format(confidences[0] *100),(x_left,y_top), cv2.FONT_HERSHEY_SIMPLEX,  1, (255,0,0), 2,  cv2.LINE_AA)

        # Display text about number of detected faces on topleft corner
        # YOUR CODE HERE
        tmp_str = 'number of faces detected:' + str(len(indices))
        print(tmp_str)
        # result = cv2.putText(result,tmp_str,(0,20), cv2.FONT_HERSHEY_SIMPLEX,  0.5, (0,0,255), 1,  cv2.LINE_AA)


###### Running Function ####
open_webcam()


