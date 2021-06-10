"""
utils.py

Various utility functions to aid in visualizing videos and detection annotations
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
ANNOTATION FORMAT SPECIFICATIONS:
The object detection evaluation module at https://github.com/rafaelpadilla/review_object_detection_metrics
(and most other detection evaluation modules) require very specific formats for annotations. 

The code I have integrated into our opendatacam setup to extract YOLO detections from
the stream outputs annotations in the "opendatacam_yolo" format described below

My implementation of Faster RCNN outputs annotations in the "faster" format
described below.

All other described formats are based off of samples in https://github.com/rafaelpadilla/review_object_detection_metrics. 
These formats were all established in various previous papers or common detection models. 



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
	This format is exclusively being used for ground truths - this is why it doesn't include a confidence score
- absxywh: folder containing one .txt for each frame, in which each line of form
	class_id confidence abs_xmin abs_ymin abs_width abs_heeight
	represents one detection in that frame
- relxywh: folder containing one .txt for each frame, in which each line of form
	class_id confidence  rel_centerx rel_centery rel_width rel_height
	represents one detection in that frame
	See See https://github.com/rafaelpadilla/review_object_detection_metrics/tree/main/data/database/dets/rel_xywh
	Note the only difference between this format and the yolo format is the inclusion of confidence score
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
and relxywh as the detection format. If using faster_rcnn results as gt, that currently entails:
1) Convert faster rcnn output (faster) to openimages using faster_to_openimages()
2) Convert that openimages to yolo using openimages_to_yolo()
TODO: eliminate above conversion intermediate
3) Convert opendatacamyolo to relxywh using opendatacamyolo_to_relxywh

Currently the evaluator is erroring out with openimages or absolute ground truth annotation format.
TODO: Figure out why this is happening. 
Note that results still appear to be trustworthy, as the illustration and analysis features
confirm that the bounding boxes are being correctly parsed by the evaluator




"""


# construct the argument parser
parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input', required=True, help='path to input video')
parser.add_argument('-a', '--annotations', required=True, help='path to annotations for video')
parser.add_argument('-o', '--output', required=True, help='path to put output video or image at')
parser.add_argument('-f', '--annotation_format', required=True, help='format of video annotaitons')
parser.add_argument('-n', '--frame_num', help='number of frame to annotate (if only annotating one frame)')
parser.add_argument("-v", "--video", action="store_true", help='annnotate full video')
parser.add_argument("-d", "--decomposed", action="store_true", help='output individual annotated frames for video')
args = vars(parser.parse_args())






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






def annotate_frame(frame, frame_num, resolution, input_annotations, annotation_format="opendatacamyolo"):
	"""
	Returns image object representing image with bounding box superimposed
	
	frame: cv2-like object representing frame to draw on
	frame_num: the number of the frame in the video (zero-indexed). This is used to find the correct
	annotation for the frmae
	resolution: resolution of the frame in pixels, represented by the tuple (W, H)
	input_annotations: depending on the annotation_format parameter, this should be
	either a folder containing a text file of annotations for each frame, or the json
	containing annotations (see annotation_format below)
	annotation_format: yolo (in the form outputted by opendatacam)
	"""

	(W, H) = resolution

	# initialize our lists of detected bounding boxes, confidences,
	# and class IDs, respectively
	boxes = []
	confidences = []
	classIDs = []

	if annotation_format == "opendatacamyolo":
	
		


		#make sure there is [ at the beginning, a ] at the end of the file, and no extraneous comma
		opendatacam_anns = open(input_annotations, "r").read()
		opendatacam_anns = opendatacam_anns.replace("}, \n{", "},{")
		final = ""

		if opendatacam_anns[0] != "[":
			final = "[" + opendatacam_anns[:-2] 
		else:
			final = opendatacam_anns[:-2] 

		if opendatacam_anns[-2] == ",":
			print("Fixing small formatting errors in opendatacamyolo annotations...")
			final = final + "]"

		frames_data = json.loads(final)


		for frame_data in frames_data:
			frame_id = frame_data["frame_id"]
			if frame_id > frame_num:
				break
			elif frame_id < frame_num:
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
			if image_id == frame_num:
				xmin = int(data[4])
				xmax = int(data[5])
				ymin = int(data[6])
				ymax = int(data[7])

				width = xmax - xmin
				height = ymax - ymin
				boxes.append([xmin, ymin, width, height])
				confidences.append(float(data[3]))
				classIDs.append(coco_names.COCO_INSTANCE_CATEGORY_NAMES.index(data[2])) #error here?
	elif annotation_format == "relxywh":
		annotation_files = glob.glob(input_annotations + "/*")
		for file in annotation_files:
			file_frame = int(file.strip(input_annotations).strip("/frame").strip(".txt"))
			if file_frame == frame_num:
				print(file)
				dets = open(file, "r")
				for line in dets:
					if line == "\n":
						continue


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
	elif annotation_format == "absxywh":
		print("Abs format")
		annotation_files = glob.glob(input_annotations + "/*")
		for file in annotation_files:
			file_frame = int(file.strip(input_annotations).strip("/frame").strip(".txt"))
			if file_frame == frame_num:
				print(file)
				dets = open(file, "r")
				for line in dets:
					if line == "\n":
						continue


					data = line.split(" ")
					print(data)

					xmin = float(data[2])
					ymin = float(data[3]) 
					width = float(data[4]) 
					height = float(data[5]) 

					boxes.append([xmin, ymin, width, height])
					classIDs.append(int(data[0]))
					confidences.append(float(data[1]))
	elif annotation_format == "yolo":
		annotation_files = glob.glob(input_annotations + "/*")
		for file in annotation_files:
			file_frame = int(file.strip(input_annotations).strip("/frame").strip(".txt"))
			if file_frame == frame_num:
				print(file)
				dets = open(file, "r")
				for line in dets:
					data = line.split(" ")
					print(data)

					center_x = float(data[1]) * W 
					center_y = float(data[2]) * H 
					width = float(data[3]) * W
					height = float(data[4]) * H

					xmin =  center_x - (width/2)
					ymin =  center_y - (height/2)
					

					boxes.append([xmin, ymin, width, height])
					classIDs.append(int(data[0]))
					confidences.append(1.0) #because this is gt format

					
			

	#draw all the boxes on the frame image
	for i in range(len(boxes)):
		# extract the bounding box coordinates
		(x, y) = (int(boxes[i][0]), int(boxes[i][1]))
		(w, h) = (int(boxes[i][2]), int(boxes[i][3]))
		# draw a bounding box rectangle and label on the fram
		param2 = (x, y)
		param3 = (x + w, y + h)
	
		cv2.rectangle(frame, param2 , param3, (155, 255, 0), 2)
		text = "{}: {:.4f}".format(classIDs[i],
			confidences[i])
		cv2.putText(frame, text, (x, y - 5),
			cv2.FONT_HERSHEY_SIMPLEX, 0.5, (155, 255, 0), 2)



	return frame



def create_annotated_video(input_video, input_annotations, out_video_name, annotation_format="yolo", output_frames=False):
	"""
	Generate annotated version of input video (i.e., superimpose all bounding boxes)

	input_video: video containing frame to annotate
	input_annotations: depending on the annotation_format parameter, this should be
	either a folder containing a text file of annotations for each frame, or the json
	containing annotations (see annotation_format below)
	out_video_name: desired name for output video
	annotation_format: format of annotations (see format specifications at top of file)
	output_frames: If true, will generate a folder out_video_name/ containing each annotated frame
	"""

	#if output_frames is True, make directory to hold annoated frames
	dir_exists = os.path.isdir(out_video_name + "/") 
	if dir_exists:
		#delete and recreate
		os.system("rm -r " + out_video_name)
		os.system("mkdir " + out_video_name)
	else:
		#just create
		os.system("mkdir " + out_video_name)


	cap = cv2.VideoCapture(input_video)
	if (cap.isOpened() == False):
	    print('Error while trying to read video. Please check path again')
	# get the frame width and height
	frame_width = int(cap.get(3))
	frame_height = int(cap.get(4))
	resolution = (frame_width, frame_height)
	
	# define codec and create VideoWriter object 
	out = cv2.VideoWriter(out_video_name + ".mp4", 
	                      cv2.VideoWriter_fourcc(*'mp4v'), 30, 
	                      resolution)
	frame_count = 0

	# read until end of video
	while(cap.isOpened()):
	    # capture each frame of the video
	    ret, frame = cap.read()
	    if ret == True:
	    	#annotate 
	    	image = annotate_frame(frame, frame_count, resolution, input_annotations, annotation_format)
	    	frame_count += 1
	    	out.write(image)

	    	if output_frames:
	    		#save the output image
	    		cv2.imwrite(out_video_name + "/annotated-frame%d.jpg" % frame_count, image)
	    else:
	        break

	# release VideoCapture()
	cap.release()
	# close all frames and video windows
	cv2.destroyAllWindows()



#create_annotated_video("videos/inria.mp4", "inria-annotations2", "annotated_inria", "yolo", True)



def draw_annotated_frame(input_video, input_annotations, target_frame, output_name, annotation_format="opendatacam_yolo"):
	"""
	draw bounding boxes on target_frame

	input_video: video containing frame to annotate
	input_annotations: depending on the annotation_format parameter, this should be
	either a folder containing a text file of annotations for each frame, or the json
	containing annotations (see annotation_format below)
	target_frame: the number of the frame (0-indexed) to annotate
	annotation_format: format of annotations (see format specifications at top of file)
	"""

	(W, H) = (None, None)


	#extract specific frame
	vidcap = cv2.VideoCapture(input_video)
	success,image = vidcap.read()
	count = 0
	frame = None
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

	resolution = (W, H)

	#pass extracted frame to annotate_frame to be drawn on
	annotated_frame = annotate_frame(frame, target_frame, resolution, input_annotations, annotation_format)

	#save the output image
	cv2.imwrite(output_name + ".jpg", annotated_frame)




#draw_annotated_frame("videos/inria.mp4", "inria-annotations2", 622, "yolo")



if args['video']:
	print("yo")
	create_annotated_video(args["input"], args["annotations"], args["output"], args["annotation_format"], args["decomposed"])
else:
	print('hi')
	draw_annotated_frame(args["input"], args["annotations"], int(args["frame_num"]), args["output"], args["annotation_format"])








