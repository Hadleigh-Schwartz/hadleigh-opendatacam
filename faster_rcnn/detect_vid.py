import torchvision
import cv2
import torch
import argparse
import time
import detect_utils
import json
from PIL import Image
# construct the argument parser
parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input', help='path to input video')
parser.add_argument('-m', '--min-size', dest='min_size', default=800, 
                    help='minimum input size for the FasterRCNN network')
args = vars(parser.parse_args())
# download or load the model from disk
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True, 
                                                    min_size=args['min_size'])
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


cap = cv2.VideoCapture(args['input'])
if (cap.isOpened() == False):
    print('Error while trying to read video. Please check path again')
# get the frame width and height
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
save_name = f"{args['input'].split('/')[-1].split('.')[0]}_{args['min_size']}"
# define codec and create VideoWriter object 
out = cv2.VideoWriter(f"outputs/{save_name}.mp4", 
                      cv2.VideoWriter_fourcc(*'mp4v'), 30, 
                      (frame_width, frame_height))



frame_count = 0 # to count total frames
total_fps = 0 # to get the final frames per second
# load the model onto the computation device
model = model.eval().to(device)





"""
[{
"image_id": int, "category_id": int, "bbox": [x,y,width,height], "score": float,
}]
"""
annotations = []


# read until end of video
while(cap.isOpened()):
    # capture each frame of the video
    ret, frame = cap.read()
    if ret == True:
        # get the start time
        start_time = time.time()
        with torch.no_grad():
            # get predictions for the current frame
            boxes, classes, labels, scores = detect_utils.predict(frame, model, device, 0.0)

            #generate json objects in correct form for each frame
            # print(boxes)
            # print(classes)
            # print(labels)
            # print(scores)
            for i in range(len(boxes)):
                print(labels[i].item())
                print(boxes[i].tolist())
                print(scores[i])
                data = {}
                data['image_id'] = int(frame_count)
                data['category_id'] = int(labels[i].item())
                data["bbox"] = boxes[i].tolist()
                data["score"] = float(scores[i])
                annotations.append(data)

        
        # draw boxes and show current frame on screen
        image = detect_utils.draw_boxes(boxes, classes, labels, frame)
        # get the end time
        end_time = time.time()
        # get the fps
        fps = 1 / (end_time - start_time)
        print(fps)
        # add fps to total fps
        total_fps += fps
        # increment frame count
        frame_count += 1
        print(frame_count)
        # press `q` to exit
        wait_time = max(1, int(fps/4))
        """
        same issue as in detect.py - no display window on Colab
        Can uncomment if running on actual machine tho
        """
        # cv2.imshow('image', image)

        out.write(image)
        if cv2.waitKey(wait_time) & 0xFF == ord('q'):
            break
    else:
        break


#convert and save annotations output 
output = "outputs/" + save_name + "-annotations.json"
with open(output, 'w') as outfile:
    json.dump(annotations, outfile)



# release VideoCapture()
cap.release()
# close all frames and video windows
cv2.destroyAllWindows()
# calculate and print the average FPS
avg_fps = total_fps / frame_count
print(f"Average FPS: {avg_fps:.3f}")







