import glob
import os

def get_all_files():
	test_files = glob.glob("INRIAPerson/Test/pos/*")
	train_files = glob.glob("INRIAPerson/Train/pos/*")

	for file in test_files:
		file_name = file.split("/")[-1]
		os.system("cp " + file + " inria-images/test-" + file_name)


	for file in train_files:
		file_name = file.split("/")[-1]
		os.system("cp " + file + " inria-images/train-" + file_name)




def map():
	all_images = glob.glob("inria-images/*")
	image_names = []

	count = 0
	for file in all_images:
		image_names.append(file.split("inria-images/")[1].split(".")[0])
		os.system("mv " + file + " inria-images2/frame" + str(count) + ".png")
		count = count + 1

	print(image_names)


	all_annotations = glob.glob("inria-annotations-yolo/*")
	for file in all_annotations:
		annotation_name = file.split("inria-annotations-yolo/")[1].split(".")[0]
		num = image_names.index(annotation_name)
		print(num)
		os.system("mv " + file + " inria-annotations-yolo2/frame" + str(num) + ".txt")


	# all_files.sort(key = str)
	# print(all_files)
	# count = 0
	# for file in all_files:
		# os.system("mv " + file + " inria-annotations-yolo/frame" + str(count) + ".txt")
		# count = count + 1





def generate_annotations():
	test_annotations = glob.glob("INRIAPerson/Test/annotations/*")
	train_annotations = glob.glob("INRIAPerson/Train/annotations/*")

	out_folder = "inria-annotations-yolo"

	for file in test_annotations:
		file_name = file.split("/")[-1]
		print(file_name)
		#open annotations file 
		outfile = open(out_folder + "/test-" + file_name, "w")

		curr_file = open(file, "r", encoding="latin-1")
		curr_annotations = curr_file.read()


		#get number of objects to be looking for
		image_size = curr_annotations.split("Image size (X x Y x C) : ")[1].split("\n")[0]
		W = int(image_size.split(" x ")[0])
		H = int(image_size.split(" x ")[1])
		print((H, W))
		num_objects = int(curr_annotations.split("Objects with ground truth : ")[1].split(" ")[0])
		print(num_objects)

		for i in range(1, num_objects + 1):
			class_info = curr_annotations.split("Details for object " + str(i))[1].split("\n")[0].split('("')[1].split('")')[0]
			# print(class_info)
			box_info = curr_annotations.split("Bounding box for object " + str(i))[1].split("\n\n")[0]
			# print(box_info)
			xmin = int(box_info.split(": (")[1].split(",")[0])
			ymin = int(box_info.split(": ")[1].split(", ")[1].split(")")[0])
			xmax = int(box_info.split("- (")[2].split(",")[0])
			ymax = int(box_info.split("- ")[2].split(",")[1].split(")")[0])
			


			#only writinng for people
			if class_info == "PASperson":

				width = (xmax - xmin) / W
				height = (ymax - ymin) / H
				center_x = (xmin / W) + (width / 2)
				center_y = (ymin / H) + (height / 2)

				# print(width)
				# print(height)
				# print(center_x)
				# print(center_y)

			
				new_line = str(1) + " " + str(center_x) + " " + str(center_y) + " " + str(width) + " " + str(height) + "\n"
				outfile.write(new_line)
		outfile.close()

	for file in train_annotations:
		file_name = file.split("/")[-1]
		print(file_name)
		#open annotations file 
		outfile = open(out_folder + "/train-" + file_name, "w")

		curr_file = open(file, "r", encoding="latin-1")
		curr_annotations = curr_file.read()


		#get number of objects to be looking for
		image_size = curr_annotations.split("Image size (X x Y x C) : ")[1].split("\n")[0]
		W = int(image_size.split(" x ")[0])
		H = int(image_size.split(" x ")[1])
		print((H, W))
		num_objects = int(curr_annotations.split("Objects with ground truth : ")[1].split(" ")[0])
		print(num_objects)

		for i in range(1, num_objects + 1):
			class_info = curr_annotations.split("Details for object " + str(i))[1].split("\n")[0].split('("')[1].split('")')[0]
			# print(class_info)
			box_info = curr_annotations.split("Bounding box for object " + str(i))[1].split("\n\n")[0]
			# print(box_info)
			xmin = int(box_info.split(": (")[1].split(",")[0])
			ymin = int(box_info.split(": ")[1].split(", ")[1].split(")")[0])
			xmax = int(box_info.split("- (")[2].split(",")[0])
			ymax = int(box_info.split("- ")[2].split(",")[1].split(")")[0])
			


			#only writinng for people
			if class_info == "PASperson":

				width = (xmax - xmin) / W
				height = (ymax - ymin) / H
				center_x = (xmin / W) + (width / 2)
				center_y = (ymin / H) + (height / 2)

				# print(width)
				# print(height)
				# print(center_x)
				# print(center_y)

			
				new_line = str(1) + " " + str(center_x) + " " + str(center_y) + " " + str(width) + " " + str(height) + "\n"
				outfile.write(new_line)
		outfile.close()


get_all_files()
generate_annotations()
map()