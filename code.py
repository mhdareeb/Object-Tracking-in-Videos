# -*- coding: utf-8 -*-
"""
Created on Fri Apr  5 00:53:56 2019

@author: Areeb
"""

# import the necessary packages
import cv2
import argparse
import numpy as np
from imutils.video import FPS
import imutils
import dlib

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", required=True,
	help="path to input video file")
ap.add_argument("-o", "--output", type=str,
	help="path to optional output video file")
args = vars(ap.parse_args())

config='yolov3.cfg'
weights='yolov3.weights'
classes='yolov3.txt'

with open(classes, 'r') as f:
    classes = [line.strip() for line in f.readlines()]


COLORS = np.random.uniform(0, 255, size=(len(classes), 3))

net = cv2.dnn.readNet(weights, config)

def get_output_layers(net):
    
    layer_names = net.getLayerNames()
    
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    return output_layers


def draw_prediction(img, class_id, confidence, x, y, x_plus_w, y_plus_h):

    label = str(classes[class_id])

    color = COLORS[class_id]

    cv2.rectangle(img, (x,y), (x_plus_w,y_plus_h), color, 2)

    cv2.putText(img, label, (x-10,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


vs = cv2.VideoCapture(args["video"])
length = int(vs.get(cv2.CAP_PROP_FRAME_COUNT))
writer = None

# initialize the list of object trackers and corresponding class
# labels

fps = FPS().start()
count=0

while True:
	# grab the next frame from the video file
    (grabbed, frame) = vs.read()

	# check to see if we have reached the end of the video file
    if frame is None:
        break
    else:
        count+=1
    per = str(round(((count/length)*100),5))+" %"
    print(per,end='\r')

    

	# resize the frame for faster processing and then convert the
	# frame from BGR to RGB ordering (dlib needs RGB ordering)
    frame = imutils.resize(frame, width=600)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    Width = frame.shape[1]
    Height = frame.shape[0]
    scale = 0.00392

	# if we are supposed to be writing a video to disk, initialize
	# the writer
    if args["output"] is not None and writer is None:
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        writer = cv2.VideoWriter(args["output"], fourcc, 30,
			(frame.shape[1], frame.shape[0]), True)

	# if there are no object trackers we first need to detect objects
	# and then create a tracker for each object
    if count%15 == 0 or count==1:
        trackers = []
        labels = []
		# grab the frame dimensions and convert the frame to a blob
        (h, w) = frame.shape[:2]
        
        blob = cv2.dnn.blobFromImage(frame, scale, (416,416), (0,0,0), True, crop=False)

        net.setInput(blob)
        
        outs = net.forward(get_output_layers(net))
        
        class_ids = []
        confidences = []
        boxes = []
        conf_threshold = 0.5
        nms_threshold = 0.4
        
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:
                    center_x = int(detection[0] * Width)
                    center_y = int(detection[1] * Height)
                    w = int(detection[2] * Width)
                    h = int(detection[3] * Height)
                    x = center_x - w / 2
                    y = center_y - h / 2
                    class_ids.append(class_id)
                    confidences.append(float(confidence))
                    boxes.append([x, y, w, h])
                    
                    
        indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

        for i in indices:
            i = i[0]
            box = boxes[i]
            x = box[0]
            y = box[1]
            w = box[2]
            h = box[3]
            startX = round(x)
            startY = round(y)
            endX = startX+round(w)
            endY = startY+round(h)
            rect = dlib.rectangle(startX, startY, endX, endY)
            t = dlib.correlation_tracker()
            t.start_track(rgb, rect)
            idx = class_ids[i]
            label = classes[idx]
            labels.append(label)
            trackers.append(t)
            
                
    else:
		# loop over each of the trackers
        for (t, l) in zip(trackers, labels):
			# update the tracker and grab the position of the tracked
			# object
            t.update(rgb)
            pos = t.get_position()

			# unpack the position object
            startX = int(pos.left())
            startY = int(pos.top())
            endX = int(pos.right())
            endY = int(pos.bottom())

			# draw the bounding box from the correlation object tracker
            cv2.rectangle(frame, (startX, startY), (endX, endY),
				(0, 255, 0), 2)
            cv2.putText(frame, l, (startX, startY - 15),
				cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)

	# check to see if we should write the frame to disk
    if writer is not None:
        writer.write(frame)

	# update the FPS counter
    fps.update()

# stop the timer and display FPS information
fps.stop()
print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

# check to see if we need to release the video writer pointer
if writer is not None:
	writer.release()

# do a bit of cleanup
cv2.destroyAllWindows()
vs.release()