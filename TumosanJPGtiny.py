# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
#%%
import numpy as np
import cv2

img = cv2.imread("/home/fuchs/Desktop/yolo/custom_yolo_model/yolov4/darknet/spot_data/spot_images/6.jpg")

img_height = img.shape[0]
img_width = img.shape[1]

img_blob = cv2.dnn.blobFromImage(img, 1/255, (416,416), swapRB=True, crop=False)

labels = ["Tumosan"]

colors = ["0,0,255", "0,255,255", "255,0,0", "255,255,0", "0,255,0"]
colors = [np.array(color.split(",")).astype("int") for color in colors]
colors = np.array(colors)
colors = np.tile(colors,(18,1))
#%%

model = cv2.dnn.readNetFromDarknet("/home/fuchs/Desktop/yolo/custom_yolo_model/yolov4/darknet/yolov4-tiny.cfg", "/home/fuchs/Desktop/yolo/custom_yolo_model/yolov4/darknet/yolov4-tiny_best.weights")

layers = model.getLayerNames()

output_layer = [layers[layer-1] for layer in model.getUnconnectedOutLayers()]

model.setInput(img_blob)

detection_layers = model.forward(output_layer)

#Nonmaxium suppresion

ids_list = []
boxes_list = []
confidences_list = []



#%% 4.bölüm

for detection_layer in detection_layers:
    for object_detection in detection_layer:
        
        scores = object_detection[5:]
        predicted_id = np.argmax(scores)
        confidence = scores[predicted_id]
        
        if confidence > 0.3:
            
            label = labels[predicted_id]
            bounding_box = object_detection[0:4]*np.array([img_width,img_height,img_width,img_height])
            (box_center_x, box_center_y, box_width, box_height) = bounding_box.astype("int")
            
            start_x = int(box_center_x-(box_width/2))
            start_y = int(box_center_y-(box_height/2))
            
            ##NON MAXIMUM SUPPRESSION OPeration 2 start
            ids_list.append(predicted_id)
            confidences_list.append(float(confidence))
            boxes_list.append([start_x, start_y, int(box_width), int(box_height)])
            
             
            ###END SUPPRESSION  OPeration 2 end
            
##NON MAXIMUM SUPPRESSION  OPeration 3 start
            
max_ids = cv2.dnn.NMSBoxes(boxes_list, confidences_list, 0.5, 0.4)

for max_id in max_ids:
    max_class_id = max_id
    box = boxes_list[max_class_id]
    
    start_x = box[0]
    start_x = box[1]
    box_width = box[2]
    box_height = box[3]
    
    predicted_id = ids_list[max_class_id]
    label = labels[predicted_id]
    confidence = confidences_list[max_class_id]
    

 ###END SUPPRESSION  OPeration 3 end
            
            
            
    end_x = start_x + box_width
    end_y = start_y + box_height
    
    box_color = colors[predicted_id]
    box_color = [int(each) for each in box_color]
    
    cv2.rectangle(img, (start_x, start_y), (end_x, end_y), box_color, 1)
    cv2.putText(img, label, (start_x, start_y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 1)

while True:
    cv2.imshow("Detection Window", img)
    if cv2.waitKey(1) == ord("q"):
        break
cv2.destroyAllWindows()

            