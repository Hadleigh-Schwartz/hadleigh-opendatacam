"""
utils.py

Functions to convert between various detection annotation formats

"""

import numpy as np
import argparse
import json
import coco_names 
import os



# construct the argument parser
parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input', help='path to input annotations')
parser.add_argument('-if', '--input_format', help='format of input annotations')
parser.add_argument('-o', '--output', help='path to put output annotations at (typically a folder namee)')
parser.add_argument('-of', '--output_format', help='desired format of output annotations')
parser.add_argument("-cf", "--class_filter", nargs="+", default=[], help='optional list of class_ids for filter')
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


	

def opendatacamyolo_to_relxywh(yolo_anns, out_folder, class_filter):
	"""Convert opendatacamyolo format to relxywh
	with one file per frame

	class_filter: List of coco_names indices. 
	Only include annotations for objects whose class_id is in class_filter
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
			if class_filter == [] or class_id in class_filter:	
				new_line = str(class_id) + " " + str(det["confidence"]) + " " + str(center_x) + " " + str(center_y) + " " + str(width) + " " + str(height)
				outfile.write(new_line)
				det_num = det_num + 1

		outfile.close()

#opendatacamyolo_to_relxywh("timesquare-5-12-21.json", "timesquare-5-12-21-det-relxywh", [0])





def faster_to_openimages(input_file, output_name):
	"""
	convert json annotations outputted by my faster_rcnn implementation
	to openimages format shown here: https://github.com/rafaelpadilla/review_object_detection_metrics/blob/main/data/database/gts/openimages_format/all_bounding_boxes.csv
	"""
	headers = "ImageID, Source, LabelName, Confidence, XMin, XMax, YMin, YMax, IsOccluded, IsTruncated, IsGroupOf, IsDepiction, IsInside"
	output = open(output_name, "w")
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


def openimages_to_yolo(input_anns, out_folder, class_filter):
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

		
		class_id = coco_names.COCO_INSTANCE_CATEGORY_NAMES.index(data[2]) - 1 #adjust down because of extraneous inclusion of background
		print(class_id)
		
		if class_filter == [] or class_id in class_filter:
			new_line = str(class_id) + " " + str(center_x) + " " + str(center_y) + " " + str(width) + " " + str(height) + "\n"
			outfile.write(new_line)




#openimages_to_yolo("timesquare-5-12-21-gt-openimages.csv", "timesquare-5-12-2-gt-yolo")





#evaluator doesn't like for some reason
def opendatacamyolo_to_absxywh(yolo_anns, out_folder, class_filter):
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

			class_id = det["class_id"]
			if class_filter == [] or class_id in class_filter:	
				new_line = str(class_id) + " " + str(det["confidence"]) + " " + str(xmin) + " " + str(ymin) + " " + str(width) + " " + str(height)
				outfile.write(new_line)
				det_num = det_num + 1

		outfile.close()



def openimages_to_absolute(input_anns, out_folder):
	"""
	DEPRECATED - NOT TESTED, DON'T USE
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





"""
driver code

call the appropriate functions according to command line args
"""

if args['input_format'] == "opendatacamyolo" and args['output_format'] == "relxywh":
	#make sure path to input annotations provided is a file
	file_exists = os.path.isfile(args["input"]) 

	#create output_folder if necessary
	out_dir = args["output"]
	dir_exists = os.path.isdir(out_dir + "/") 
	if dir_exists:
		#delete and recreate
		os.system("rm -r " + out_dir)
		os.system("mkdir " + out_dir )
	else:
		#just create
		os.system("mkdir " + out_dir)

	class_filter = [int(i) for i in args["class_filter"]]
	if file_exists:
		opendatacamyolo_to_relxywh(args["input"], out_dir, class_filter)

elif args['input_format'] == "opendatacamyolo" and args['output_format'] == "absxywh":
	#make sure path to input annotations provided is a file
	file_exists = os.path.isfile(args["input"]) 

	#create output_folder if necessary
	out_dir = args["output"]
	dir_exists = os.path.isdir(out_dir + "/") 
	if dir_exists:
		#delete and recreate
		os.system("rm -r " + out_dir)
		os.system("mkdir " + out_dir)
	else:
		#just create
		os.system("mkdir " + out_dir)

	class_filter = [int(i) for i in args["class_filter"]]
	if file_exists:
		opendatacamyolo_to_absxywh(args["input"], out_dir, class_filter)

elif args['input_format'] == "faster" and args['output_format'] == "yolo":
	#make sure path to input annotations provided is a file
	file_exists = os.path.isfile(args["input"]) 

	faster_to_openimages(args["input"], "temp.csv")


	#create output_folder if necessary
	out_dir = args["output"]
	dir_exists = os.path.isdir(out_dir + "/") 
	if dir_exists:
		#delete and recreate
		os.system("rm -r " + out_dir)
		os.system("mkdir " + out_dir)
	else:
		#just create
		os.system("mkdir " + out_dir)

	class_filter = [int(i) for i in args["class_filter"]]
	print(class_filter)
	openimages_to_yolo("temp.csv", out_dir, class_filter)

	os.system("rm temp.csv")











