def decompose(video, out_folder):
	"""
	get all frames for video
	and store each as a jpg in out_folder
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




def recompose(frame_folder, output_name):
	"""
	generate a video in mp4 form from a folder of individual frames
	frames in the folder should be named img0, img1, ....
	the video will be generated according to the frame sequence
	specificed by the numbers in the frame file names

	NOTE: Must have ffmpeg installed on machine
	"""
	os.system("ffmpeg  -i " + frame_folder + "/img%d.png " + output_name + ".mp4")
