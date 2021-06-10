# hadleigh-opendatacam
This repository contains code for a streamlined end-to-end implementation of object detection on opendatacam, supporting steps from initial video upload to evaluation of generated annotations. It includes 
1) A guide and code for setting up and running a custom version of opendatacam on a Jetson Nano or Xavier. 
2) Various utility tools for analyzing, formatting, and converting object detection annotations.
3) Sample videos and annotations in various formats.


# Initial setup 

Once you have set up your Jetson computer and followed the instructions [here](https://github.com/opendatacam/opendatacam) to set up the original version of opendatacam on your machine, complete the steps below. This will prepare you to use our custom version of opendatacam and run start_cam.sh. You only need to do this once. 



1. Mount a directory to store opendatacam’s annotations. To do so, modify docker-compose.yml by adding an entry in the volumes field under the opendatacam service. For example, I mounted a folder  /home/hadleigh/annotations by adding an entry under volumes as such:  

```
volumes:
      - './config.json:/var/local/opendatacam/config.json'
      - '/home/hadleigh/annotations:/var/local/opendatacam/annotations'
```


2. Edit the entry of the “file” field in config.json to be empty, as such:

 ```
"VIDEO_INPUTS_PARAMS": {
 		 "file": "",
```


3. Install a few necessary libraries.

```
sudo apt-get update 
sudo apt-get install wpasupplicant 
sudo apt-get install inotify-tools
sudo apt-get install dhclient
```


4. Edit the file /etc/wpa-supplicant.conf to contain ssid and password for your wlan0 network.  	

```
network={
       ssid="network_name"
       psk="password"
}
```


5. Add the custom image. scp custom-opendatacam.tar.gz to your Jetson machine. Then run the following command.
```
sudo docker load < custom-opendatacam.tar.gz
```
Run 
```
sudo docker images
```
to confirm that the image has been loaded (you should see custom-opendatacam listed). Now edit the "image" field in docker-compose.yml to be custom-opendatacam, so it looks like this:
```
opendatacam:
    restart: always
    image: custom-opendatacam
```

# Running opendatacam on a Jetson computer

I have written a bash file (start_cam.sh) that streamlines getting opendatacam up and running on a Jetson machine (provided you have completed the initial setup steps above). I recommend scp-ing start_cam.sh onto the machine, sshing onto it, and running it on there so you can see the instructions it outputs (including the IP address the opendatacam GUI can be accessed on). 

In the custom docker image, I have slightly modified the original opendatacam code to intercept YOLO annotations for a video and write them to a json file. Once a file has been fully processed, you can find the annotations for a video in the mounted annotations folder, in the form of a file named according to the date-time the video was uploaded to opendatacam browser GUI.

If you get an error saying “Unable to connect to WiFi” upon running start_cam.sh, I recommend simply re-running the script, as this usually resolves the issue. If you continue to encounter that error, you will likely need to troubleshoot by running each of the following commands individually (in this specific order), examining their output, and then seeing if you can ping a random service:


```
sudo pkill wpa_supplicant 
sudo wpa_supplicant -B -i wlan0 -c /etc/wpa_supplicant.conf -D wext
sudo dhclient wlan0
```



# Other helpful commands

Safely shut down the machine once finished using it


```
sudo shutdown -h now
```


View information on running containers, including their ids and images. 


```
sudo docker ps -a
```


Save a docker container 


```
sudo docker commit <CONTAINER_ID> <NEW_NAME>
```


Access opendatacam container bash:


```
sudo docker exec -t -i  <CONTAINER_NAME>  /bin/bash
```



# Working with annotations

convert_annotations.py and visual_utils.py provide functions for converting between various common object detection annotation formats and visualizing annotations on images and videos. 


### Formats and conversions

Converting between annotation formats is critical for comparing annotations made by different object detection models - since each object detection model typically uses its own format - and for compatibility with object detection evaluation models. Below, I describe five annotation formats that are compatible with convert_annotations and visual_utils.py. 



1. **opendatacamyolo**: a json file of the form

	


```
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
```




2. **openimages:** a csv file, where each row is  the form

```
ImageID, Source, LabelName, Confidence, XMin (absolute), XMax, YMin, YMax, IsOccluded, IsTruncated, IsGroupOf, IsDepiction, IsInside
```


3.  **yolo:** a  folder containing one .txt for each frame, in which each line of form
```
class_id rel_centerx rel_centery rel_width rel_height
```

represents one detection in that frame.
This format is exclusively being used for ground truths - this is why it doesn't include a confidence score



4.  **relxywh:** folder containing one .txt for each frame, in which each line of form
```
class_id confidence  rel_centerx rel_centery rel_width rel_height
```

represents one detection in that frame.
See files [ here](https://github.com/rafaelpadilla/review_object_detection_metrics/tree/main/data/database/dets/rel_xywh) file for an example. Note the only difference between this format and the yolo format is the inclusion of confidence score



5. ** faster:** the format my implementation of Faster RCNN outputs, a json file of the form

	


```
[
		{
			"image_id": 0,
			 "category_id": 1, 
			"bbox": [abs_xmin, abs_ymin, abs_xmax, abs_ymax], 

			"confidence_score": 0.9972374439239502
		}, ....
]
```
image_id should be the frame number in the video (zero-indexed)





convert_annotations.py can currently perform the following conversions:
*   opendatacamyolo -> relxywh
*   opendatacamyolo -> absxywh
*   faster -> yolo

Run it with the following parameters: 
*   -i :  path to input annotations
*   -if : format of input annotations
*   -o : path to put output annotations at (typically a folder name)
*   -of : desired format of output annotations
*   -cf : optional list of class_ids for filter. Class_ids and their corresponding class names can be found here. If this parameter is specified, only objects who are labeled as a class whose class_id is present in the provided list will be included in the outputted annotations file(s). 


### Visualizing annotations 

visual_py contains functions for visualizing annotations on an individual frame of a video, and visualizing annotations on the whole video. Run it with the following parameters
*   -i path to video to annotate
*   -a path to annotations corresponding to video
*   -f annotation format 
*   -n target frame number (i.e., the zero-indexed number of the frame of the video you would like to annotate)
*   -o path to output video or image (i.e., the name of the annotations video or image)
*   -v optional; include this flag if you would like to output a fully annotated a video. When not included, a target frame number is expected in order to annotate a single video.
*   -d optional; include this flag if you have set -v and would like to output a folder containing individual annotated frames in addition to the fully annotated video.

 


# Evaluating annotations

To evaluate annotations, use the tool and follow the instructions provided here:

[https://github.com/rafaelpadilla/review_object_detection_metrics](https://github.com/rafaelpadilla/review_object_detection_metrics)

This module re-implements existing object detection evaluation methods  (e.g., AP, mAP, AP per class), but provides usability and flexibility between formats. It also provides a helpful GUI for loading and visualizing annotations on individual frames (but unlike my code for visualizing annotations, it does not provide functionality for exporting annotated videos). 

Note that although this tool is compatible with many different annotation formats, it does not support opendatacamyolo. You will need to use convert_annotations.py to convert the json file outputted by opendatacam to an accepted ground truth annotation format - I recommend converting to yolo (i.e., opendatacam -> yolo). I recommend using relxywh (corresponding to the GUI’s ) as the det annotation format. 

You will also need to provide a folder containing an image file for each frame of the video. You cannot load a video. Use the decompose() function in video_utils.py to generate this from video. Lastly, you will need to provide a file in the “classes” field - this file allows the evaluator to accurately map class id’s to class names. Use the file classes.txt for this. 

![Evaluator GUI](/assets/evaluator-gui.png)
