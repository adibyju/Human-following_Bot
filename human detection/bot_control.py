#!/usr/bin/env python
import numpy as np
import cv2 as cv
import imutils
import time
import rospy
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import argparse
import subprocess
import os

x2=1
y2=1
a=-1.0
FLAGS = []
count = 0
w=0.0
boxes = []
confidences = []
classids = []
idxs = ()










def draw_labels_and_boxes(img, boxes, confidences, classids, idxs, colors, labels):
    # If there are any detections
    if len(idxs) > 0:
        for i in idxs.flatten():
            # Get the bounding box coordinates
            global w
            x, y = boxes[i][0], boxes[i][1]
            w, h = boxes[i][2], boxes[i][3]
            
            # Get the unique color for this class
            color = [int(c) for c in colors[classids[i]]]

            # Draw the bounding box rectangle and label on the image
            cv.rectangle(img, (x, y), (x+w, y+h), color, 2)
            text = "{}: {:4f}".format(labels[classids[i]], confidences[i])
            cv.putText(img, text, (x, y-5), cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            global x2,y2
            x2 = x + int(w / 2)
            y2 = y + int(h / 2)
            cv.circle(img, (x2, y2), 4, (0, 255, 0), -1)
            text = "x: " + str(x2) + ", y: " + str(y2)
            cv.putText(img, text, (x2 - 10, y2 - 10),
            cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return img


def generate_boxes_confidences_classids(outs, height, width, tconf):
    boxes = []
    confidences = []
    classids = []

    for out in outs:
        for detection in out:
            #print (detection)
            #a = input('GO!')
            
            # Get the scores, classid, and the confidence of the prediction
            scores = detection[5:]
            classid = np.argmax(scores)
            confidence = scores[classid]
            
            # Consider only the predictions that are above a certain confidence level
            if confidence > tconf:
                # TODO Check detection
                box = detection[0:4] * np.array([width, height, width, height])
                centerX, centerY, bwidth, bheight = box.astype('int')

                # Using the center x, y coordinates to derive the top
                # and the left corner of the bounding box
                x = int(centerX - (bwidth / 2))
                y = int(centerY - (bheight / 2))

                # Append to list
                boxes.append([x, y, int(bwidth), int(bheight)])
                confidences.append(float(confidence))
                classids.append(classid)

    return boxes, confidences, classids

def infer_image(net, layer_names, height, width, img, colors, labels, FLAGS, 
            boxes=None, confidences=None, classids=None, idxs=None, infer=True):
    
    if infer:
        # Contructing a blob from the input image
        blob = cv.dnn.blobFromImage(img, 1 / 255.0, (416, 416), 
                        swapRB=True, crop=False)

        # Perform a forward pass of the YOLO object detector
        net.setInput(blob)

        # Getting the outputs from the output layers
        start = time.time()
        outs = net.forward(layer_names)
        end = time.time()

        if FLAGS.show_time:
            print ("[INFO] YOLOv3 took {:6f} seconds".format(end - start))

        
        # Generate the boxes, confidences, and classIDs
        boxes, confidences, classids = generate_boxes_confidences_classids(outs, height, width, FLAGS.confidence)
        
        # Apply Non-Maxima Suppression to suppress overlapping bounding boxes
        idxs = cv.dnn.NMSBoxes(boxes, confidences, FLAGS.confidence, FLAGS.threshold)

    if boxes is None or confidences is None or idxs is None or classids is None:
        raise '[ERROR] Required variables are set to None before drawing boxes on images.'
        
    # Draw labels and boxes on the image
    img = draw_labels_and_boxes(img, boxes, confidences, classids, idxs, colors, labels)

    return img, boxes, confidences, classids, idxs
















def drive_robot(lin,ang):
  pub = rospy.Publisher('cmd_vel', Twist, queue_size=1)
  move = Twist()
  move.linear.x = lin
  move.angular.z = ang
  pub.publish(move)

def subscriber():
  rospy.Subscriber('/camera/rgb/image_raw',Image,entecallbackfunction)
  rospy.spin()
  cv.destroyAllWindows()

def entecallbackfunction(entemsg):
  try:
	print("Received image")
	bridge = CvBridge()
	cap1 = bridge.imgmsg_to_cv2(entemsg, desired_encoding='passthrough')
	# grab the current frame
	frame = cap1
	frame = imutils.resize(frame, width=800)
	global count
	height, width = frame.shape[:2]
	global x2,y2
	y2 = -1
	global w
	w = 0.0
	global boxes, confidences, classids, idxs
	if count == 0:
		frame, boxes, confidences, classids, idxs = infer_image(net, layer_names, \
    						height, width, frame, colors, labels, FLAGS)
		count += 1
	else:
		frame, boxes, confidences, classids, idxs = infer_image(net, layer_names, \
    						height, width, frame, colors, labels, FLAGS, boxes, confidences, classids, idxs, infer=False)
		count = (count + 1) % 6
	cv.imshow('Frame', frame)
	cv.waitKey(1)

	#forward_dist = rospy.wait_for_message('/scan', LaserScan, timeout=None).ranges[0]
	global a
	if 0 < y2 < 60:
		drive_robot(-0.8,0.0)
	else:
		if w > 77.0:
			drive_robot(-0.8,0.0)
		elif w > 70.0:
			drive_robot(0.0,0.0)
		elif w == 0.0:
			drive_robot(0.0,a)
		else:
			if x2 > 420:
				drive_robot(0.8,-0.4)
				a=-1.0
			elif x2 < 380:
				drive_robot(0.8,0.4)
				a=1.0
			else:
				drive_robot(0.8,0.0)

  except CvBridgeError as e:
    print(e)

if __name__=='__main__':
	rospy.init_node("bot_control", anonymous=True)

	parser = argparse.ArgumentParser()

	parser.add_argument('-m', '--model-path',
		type=str,
		default='./yolov3-coco/',
		help='The directory where the model weights and \
			  configuration files are.')

	parser.add_argument('-w', '--weights',
		type=str,
		default='./yolov3-coco/yolov3.weights',
		help='Path to the file which contains the weights \
			 	for YOLOv3.')

	parser.add_argument('-cfg', '--config',
		type=str,
		default='./yolov3-coco/yolov3.cfg',
		help='Path to the configuration file for the YOLOv3 model.')

	parser.add_argument('-i', '--image-path',
		type=str,
		help='The path to the image file')

	parser.add_argument('-v', '--video-path',
		type=str,
		help='The path to the video file')


	parser.add_argument('-vo', '--video-output-path',
		type=str,
        default='./output.avi',
		help='The path of the output video file')

	parser.add_argument('-l', '--labels',
		type=str,
		default='./yolov3-coco/coco-labels',
		help='Path to the file having the \
					labels in a new-line seperated way.')

	parser.add_argument('-c', '--confidence',
		type=float,
		default=0.5,
		help='The model will reject boundaries which has a \
				probabiity less than the confidence value. \
				default: 0.5')

	parser.add_argument('-th', '--threshold',
		type=float,
		default=0.3,
		help='The threshold to use when applying the \
				Non-Max Suppresion')

	parser.add_argument('--download-model',
		type=bool,
		default=False,
		help='Set to True, if the model weights and configurations \
				are not present on your local machine.')

	parser.add_argument('-t', '--show-time',
		type=bool,
		default=False,
		help='Show the time taken to infer each image.')

	FLAGS, unparsed = parser.parse_known_args()

	# Download the YOLOv3 models if needed
	if FLAGS.download_model:
		subprocess.call(['./yolov3-coco/get_model.sh'])

	# Get the labels
	labels = open(FLAGS.labels).read().strip().split('\n')

	# Intializing colors to represent each label uniquely
	colors = np.random.randint(0, 255, size=(len(labels), 3), dtype='uint8')

	# Load the weights and configutation to form the pretrained YOLOv3 model
	net = cv.dnn.readNetFromDarknet(FLAGS.config, FLAGS.weights)

	# Get the output layer names of the model
	layer_names = net.getLayerNames()
	layer_names = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

	subscriber()
