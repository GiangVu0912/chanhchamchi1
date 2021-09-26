import cv2
from func import CLASS_NAMES, detect_bounding_box
from func import save_file_to_tmp
from func import predict_image
import os

cap = cv2.VideoCapture(0)


if not cap.isOpened():
    raise IOError("Cannot open webcam")

while True:
    ret, frame = cap.read()

    # frame is now the image capture by the webcam (one frame of the video)
    cv2.imshow('Input', frame)
    try:
        boxes = detect_bounding_box(frame)
        # print(boxes)
        x_left, y_top, width, height = boxes[0]

        #vẽ bounding box lên cái frame
        cv2.rectangle(frame, (x_left,y_top), (x_left+width,y_top+height), (0, 0, 255), 1) 
        
        crop_img = frame[y_top:y_top+height, x_left:x_left+width]
        img_path = save_file_to_tmp(crop_img)
        print(img_path)

        prediction, index, emotion, proba_emotion = predict_image(img_path)
        os.remove(img_path)

        #Result
        model_confidence = prediction[index]
        if model_confidence > 0.75: # Tỷ lệ quá thấp -> người lạ
            member_name = CLASS_NAMES[index]
        else:
            member_name = 'YOU ARE STRANGER'

        member_emotion = emotion


        print('reslt:',prediction, index, emotion)
        frame = cv2.putText(frame,"{:.2f}%".format(model_confidence *100)+ '-' + member_name,(x_left,y_top), cv2.FONT_HERSHEY_SIMPLEX,  0.8, (255,0,0), 1,  cv2.LINE_AA)
        frame = cv2.putText(frame,"{:.2f}%".format(proba_emotion *100)+ '-' +member_emotion,(x_left,y_top+20), cv2.FONT_HERSHEY_SIMPLEX,  0.8, (0,255,0), 1,  cv2.LINE_AA)
    
    except IndexError as ierr:
        frame = cv2.putText(frame,'Cannot found Bounding Box',(20,20), cv2.FONT_HERSHEY_SIMPLEX,  0.8, (255,0,0), 1,  cv2.LINE_AA)
        print(ierr)
        pass

    except Exception as e:
        frame = cv2.putText(frame,str(e),(10,10), cv2.FONT_HERSHEY_SIMPLEX,  0.8, (255,0,0), 1,  cv2.LINE_AA)
        print(e)   
        pass     

    cv2.imshow('Input', frame)
    
    c = cv2.waitKey(1)
    # Break when pressing ESC
    #print('Button clicked:',c)
    if c == 27:
        run = False
        break


cap.release()
cv2.destroyAllWindows()

