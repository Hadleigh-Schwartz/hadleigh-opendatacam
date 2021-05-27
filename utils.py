"""
utils.py

Various utility functions to aid in analyzing and converting
annotations and videos. 
"""


import numpy as np
import argparse
import imutils
import time
import cv2
import os
import json
import coco_names 
import glob


"""
FORMAT SPECIFICATIONS:
The object detection evaluation module at https://github.com/rafaelpadilla/review_object_detection_metrics
(and most other detection evaluation modules) require very specific input. 

The code I have integrated into our opendatacam setup to extract YOLO detections from
the stream outputs annotations in "opendatacam_yolo" format described below

My implementation of Faster RCNN outputs annotations in the "faster" format
described below.

All other formats are from https://github.com/rafaelpadilla/review_object_detection_metrics,
and are based on formats that have been established in previous papers or common detection models.


- opendatacam_yolo: json file of the form
	[
		{
			"frame_id":x, 
			"objects": [
				{
					"class_id":0,
					"name": "person", 
					"relative_coordinates": {
						"center_x":0,
						"center_y":0,
						"width":0,
						"height":0
					},
					"confidence":0
				}.....
			]
		}
	]
- openimages: .csv file of the form
	ImageID, Source, LabelName, Confidence, XMin (absolute), XMax, YMin, YMax, IsOccluded, IsTruncated, IsGroupOf, IsDepiction, IsInside
	NOTE: evaluator buggy when using this as ground truth format
- yolo: folder containing one .txt for each frame, in which each line of form
	class_id rel_centerx rel_centery rel_width rel_height 
	represents one detection in that frame
- absxywh: folder containing one .txt for each frame, in which each line of form
	class_id abs_xmin abs_ymin abs_xmax abs_ymax
	represents one detection in that frame
- relxywh: folder containing one .txt for each frame, in which each line of form
	class_id rel_xmin rel_ymin rel_xmax rel_ymax
	represents one detection in that frame
	See See https://github.com/rafaelpadilla/review_object_detection_metrics/tree/main/data/database/dets/rel_xywh
- absolute: folder containing one .txt for each frame, in which each line of form
	class_id abs_xmin abs_ymin abs_xmax abs_ymax
	represents one detection in that frame
	NOTE: evaluator buggy when using this as ground truth format
- faster: the format my implementation of Faster RCNN outputs, a json file of the form
	[
		{"image_id": 0, "category_id": 1, "bbox": [abs_xmin, abs_ymin, abs_xmax, abs_ymax], "confidence_score": 0.9972374439239502},....
	]
	image_id should be the frame number in the video (zero-indexed)



For using the detection evaluation module, I recommend using yolo as ground truth format  
and either absxywh or relxywh as detection format. That currently entails:
1) Convert faster rcnn output (faster) to openimages using faster_to_openimages()
2) Convert that openimages to yolo using openimages_to_yolo()
TODO: eliminate above conversion intermediate
3) Convert opendatacamyolo to absxywh or relxywh using opendatacamyolo_to_absxywh/relxywh

Currently the evaluator is erroring out with openimages or absolute ground truth annotation format.
TODO: Figure out why this is happening. 
Note that results still appear to be trustworthy, as the illustration and analysis features
confirm that the bounding boxes are being correctly parsed by the evaluator


"""




def decompose(video, out_folder):
	"""
	get all frames for video
	and store them in folder
	"""
	
	vidcap = cv2.VideoCapture(video)
	success,image = vidcap.read()
	count = 0
	while success:
	  cv2.imwrite(out_folder + "/frame%d.jpg" % count, image) 
	  success,image = vidcap.read()
	  print('Read a new frame: ', success)
	  count += 1
	  print(count)


#decompose("videos/timesquare-5-12-21.mp4", "timesquare-5-12-21-frames")

def get_resolution(video):
	""" 
	get the width and height of video (in pixels)
	this is important for scaling to relative or absolute
	coordinate formats for bounding boxes
	"""
	(W, H) = (None, None)
	vidcap = cv2.VideoCapture(video)
	success,image = vidcap.read()
	(H, W) = image.shape[:2]   
	return (H, W)


def draw_annotated_frame(input_video, input_annotations, target_frame, annotation_format="opendatacam_yolo"):
	"""
	draw bounding boxes on target_frame
	input_video: video containing frame to annotate
	input_annotations: depending on the annotation_format parameter, this should be
	either a folder containing a text file of annotations for each frame, or the json
	containing annotations (see annotation_format below)
	target_frame: the number of the frame (0-indexed) to annotate
	annotation_format: yolo (in the form outputted by opendatacam)
	"""

	(W, H) = (None, None)

	#extract specific frame
	vidcap = cv2.VideoCapture(input_video)
	success,image = vidcap.read()
	count = 1
	while success:
	  if count == target_frame:
		  (H, W) = image.shape[:2]   
		  print("H, W")
		  print((H, W))
		  frame = image
		  break
		 
	  success,image = vidcap.read()
	  # print('Read a new frame: ', success)
	  count += 1


	# initialize our lists of detected bounding boxes, confidences,
	# and class IDs, respectively
	boxes = []
	confidences = []
	classIDs = []

	if annotation_format == "opendatacam_yolo":
		detection_json = open(input_annotations, 'r')
		json_string = detection_json.read()
		frames_data = json.loads(json_string)

		for frame_data in frames_data:
			frame_id = frame_data["frame_id"]
			if frame_id > target_frame:
				break
			elif frame_id < target_frame:
				continue
			else:
				detections = frame_data["objects"]
				for detection in detections:
					# scale the bounding box coordinates to actual image 
					#resolution, since YOLO
					# returns the relative center (x, y)-coordinates of
					# the bounding box followed by the boxes' relative width and
					# height
					box_raw = detection["relative_coordinates"]
					box_unscaled = (box_raw["center_x"], box_raw['center_y'], box_raw['width'], box_raw['height'])
					box_scaled = box_unscaled * np.array([W, H, W, H])
					(centerX, centerY, width, height) = box_scaled.astype("int")
					
					# use the center (x, y)-coordinates to derive the top
					# and and left corner of the bounding box
					x = int(centerX - (width / 2))
					y = int(centerY - (height / 2))


					# update our list of bounding box coordinates,
					# confidences, and class IDs
					boxes.append([x, y, int(width), int(height)])
					confidences.append(float(detection["confidence"]))
					classIDs.append(detection["class_id"])
	elif annotation_format == "openimages":
		annotations = open(input_annotations, "r")
		headers = annotations.readline()
		for line in annotations:
			data = line.split(",")
			image_id = int(data[0].split("frame")[1].split(".jpg")[0])
			if image_id == target_frame:
				xmin = int(data[4])
				xmax = int(data[5])
				ymin = int(data[6])
				ymax = int(data[7])

				width = xmax - xmin
				height = ymax - ymin
				boxes.append([xmin, ymin, width, height])
				confidences.append(float(data[3]))
				classIDs.append(coco_names.COCO_INSTANCE_CATEGORY_NAMES.index(data[2])) #error here?
	elif annotation_format == "rel_xywh":
		annotation_files = glob.glob(input_annotations + "/*")
		for file in annotation_files:
			file_frame = int(file.strip(input_annotations).strip("/frame").strip(".txt"))
			if file_frame == target_frame:
				print(file)
				dets = open(file, "r")
				for line in dets:
					data = line.split(" ")
					print(data)
					center_x = float(data[2]) * W 
					center_y = float(data[3]) * H 
					width = float(data[4]) * W
					height = float(data[5]) * H

					xmin =  center_x - (width/2)#scale it
					ymin =  center_y - (height/2)#scale it
					

					boxes.append([xmin, ymin, width, height])
					classIDs.append(int(data[0]))
					confidences.append(float(data[1]))
	elif annotation_format == "abs_xywh":
		print("Abs format")
		annotation_files = glob.glob(input_annotations + "/*")
		for file in annotation_files:
			file_frame = int(file.strip(input_annotations).strip("/frame").strip(".txt"))
			if file_frame == target_frame:
				print(file)
				dets = open(file, "r")
				for line in dets:
					data = line.split(" ")
					print(data)

					xmin = float(data[2])
					ymin = float(data[3]) 
					width = float(data[4]) 
					height = float(data[5]) 

					boxes.append([xmin, ymin, width, height])
					classIDs.append(int(data[0]))
					confidences.append(float(data[1]))


	#draw all the boxes on the frame image
	for i in range(len(boxes)):
		# extract the bounding box coordinates
		(x, y) = (int(boxes[i][0]), int(boxes[i][1]))
		(w, h) = (int(boxes[i][2]), int(boxes[i][3]))
		# draw a bounding box rectangle and label on the fram
		param2 = (x, y)
		param3 = (x + w, y + h)
		print(param2)
		print(param3)
		cv2.rectangle(frame, param2 , param3, (155, 255, 0), 2)
		text = "{}: {:.4f}".format(classIDs[i],
			confidences[i])
		cv2.putText(frame, text, (x, y - 5),
			cv2.FONT_HERSHEY_SIMPLEX, 0.5, (155, 255, 0), 2)


	# show the output image (optional)
	# cv2.imshow("Image", frame)
	# cv2.waitKey(0)

	#save the output image
	cv2.imwrite("annotated-frame%d.jpg" % target_frame, frame)

	print("NUMBER OF FRAMES: ")
	print(count)



#draw_annotated_frame("videos/timesquare-5-12-21.mp4", "timesquare-5-12-21-det-relxywh", 20, "rel_xywh")



def faster_to_openimages(input_file):
	"""
	convert json annotations outputted by my faster_rcnn implementation
	to openimages format shown here: https://github.com/rafaelpadilla/review_object_detection_metrics/blob/main/data/database/gts/openimages_format/all_bounding_boxes.csv
	"""
	headers = "ImageID, Source, LabelName, Confidence, XMin, XMax, YMin, YMax, IsOccluded, IsTruncated, IsGroupOf, IsDepiction, IsInside"
	output = open(input_file.replace("-faster.json", "") + "-openimages.csv", "w")
	output.write(headers + "\n")



	faster_annotations = json.loads(open(input_file, "r").read())
	for ann in faster_annotations:
		in_image_id = ann["image_id"]
		out_image_id = "frame" + str(ann["image_id"]) + ".jpg"

		label_name = coco_names.COCO_INSTANCE_CATEGORY_NAMES[ann["category_id"]]

		confidence = ann["score"]

		bbox = ann["bbox"]
		xmin = bbox[0]
		ymin = bbox[1]
		xmax = bbox[2]
		ymax = bbox[3]

		row = out_image_id + ",," + label_name + "," + str(confidence) + "," + str(xmin) + "," + str(xmax) + "," + str(ymin) + "," + str(ymax) + ",,,,,\n"
		output.write(row)

	output.close()




#faster_to_openimages("timesquare-5-12-21-gt-faster.json")


def openimages_to_absolute(input_anns, out_folder):
	"""
	Convert openimages csv file to folder of absolute annotations

	IMPORTANT: Make sure (H, W) is correct for video. 
	Use get_resolution() to confirm if needed
	"""


	(H, W) = (720, 1280) #IMPORTANT



	annotations = open(input_anns, "r")
	headers = annotations.readline()
	count = 0
	for line in annotations:
		# if count > 3:
		# 	break
		data = line.split(",")
	
		image_id = int(data[0].split("frame")[1].split(".jpg")[0])
		outfile = open(out_folder + "/frame" + str(image_id) + ".txt", "a")


		xmin = int(data[4])
		xmax = int(data[5])
		ymin = int(data[6])
		ymax = int(data[7])

		width = xmax - xmin
		height = ymax- ymin

	
		class_id = coco_names.COCO_INSTANCE_CATEGORY_NAMES.index(data[2])
		print(class_id)
		count = count + 1
		new_line = str(class_id) + " " + str(xmin) + " " + str(ymin) + " " + str(width) + " " + str(height) + "\n"
		outfile.write(new_line)


def openimages_to_yolo(input_anns, out_folder):
	"""
	Convert openimages csv file to folder of yolo annotations 

	IMPORTANT: Make sure (H, W) is correct for video. 
	Use get_resolution() to confirm if needed
	"""

	(H, W) = (720, 1280) #IMPORTANT



	annotations = open(input_anns, "r")
	headers = annotations.readline()
	for line in annotations:
		# if count > 3:
		# 	break
		data = line.split(",")
	
		image_id = int(data[0].split("frame")[1].split(".jpg")[0])
		outfile = open(out_folder + "/frame" + str(image_id) + ".txt", "a")


		xmin = int(data[4]) / W
		xmax = int(data[5]) / W
		ymin = int(data[6]) / H
		ymax = int(data[7]) / H

		width = xmax - xmin
		height = ymax- ymin

		center_x = xmin + (width/2)
		center_y = ymin + (height/2)

	
		class_id = coco_names.COCO_INSTANCE_CATEGORY_NAMES.index(data[2])
		
		new_line = str(class_id) + " " + str(center_x) + " " + str(center_y) + " " + str(width) + " " + str(height) + "\n"
		outfile.write(new_line)




openimages_to_yolo("timesquare-5-12-21-gt-openimages.csv", "timesquare-5-12-2-gt-yolo")


FILTER = [0]
def opendatacamyolo_to_relxywh(yolo_anns, out_folder):
	"""Convert opendatacamyolo format to relxywh
	with one file per frame
	"""
	input_anns = json.loads(open(yolo_anns, "r").read())

	for frame_data in input_anns:
		frame_id = frame_data["frame_id"]
		outfile = open(out_folder + "/frame" + str(frame_id) + ".txt", "w")
		det_num = 1
		for det in frame_data["objects"]:
			width = det["relative_coordinates"]["width"]
			height = det["relative_coordinates"]["height"]
			center_x = det["relative_coordinates"]["center_x"]
			center_y = det["relative_coordinates"]["center_y"]
			if det_num != 1:
				outfile.write("\n")


			class_id = det["class_id"]
			if class_id in FILTER:	
				new_line = str(class_id) + " " + str(det["confidence"]) + " " + str(center_x) + " " + str(center_y) + " " + str(width) + " " + str(height)
				outfile.write(new_line)
				det_num = det_num + 1

		outfile.close()

#opendatacamyolo_to_relxywh("timesquare-5-12-21.json", "timesquare-5-12-21-det-relxywh")



#evaluator doesn't like for some reason
def opendatacamyolo_to_absxywh(yolo_anns, out_folder):
	"""Convert opendataccamyolo format to absxywh
	with one file per frame
	
	
	IMPORTANT: Make sure (H, W) is correct for video. 
	Use get_resolution() to confirm if needed
	"""


	(H, W) = (720, 1280) #IMPORTANT



	input_anns = json.loads(open(yolo_anns, "r").read())

	for frame_data in input_anns:
		frame_id = frame_data["frame_id"]
		outfile = open(out_folder + "/frame" + str(frame_id) + ".txt", "w")
		det_num = 1
		for det in frame_data["objects"]:
			width = det["relative_coordinates"]["width"] * W
			height = det["relative_coordinates"]["height"] * H
			center_x = det["relative_coordinates"]["center_x"] * W
			center_y = det["relative_coordinates"]["center_y"] * H
			xmin = (center_x - (width/2)) 
			ymin = (center_y - (height/2))
			if det_num != 1:
				outfile.write("\n")

			new_line = str(det["class_id"]) + " " + str(det["confidence"]) + " " + str(xmin) + " " + str(ymin) + " " + str(width) + " " + str(height)
			outfile.write(new_line)
			det_num = det_num + 1

		outfile.close()







