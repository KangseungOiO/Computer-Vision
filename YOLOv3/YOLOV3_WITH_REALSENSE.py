from __future__ import division
import time
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import cv2
from tutorialUtil import *
import argparse
import os
import os.path as osp
from tutorialDarknet import Darknet
import pickle as pkl
import pandas as pd
import random
from realsense_test import *
import numpy as np


def arg_parse():
    """
    Parse arguements to the detect module

    """

    parser = argparse.ArgumentParser(description='YOLO v3 Detection Module')
    parser.add_argument("--bs", dest = "bs", help = "Batch size", default = 1)
    parser.add_argument("--confidence", dest = "confidence", help = "Object Confidence to filter predictions", default = 0.5)
    parser.add_argument("--nms_thresh", dest = "nms_thresh", help = "NMS Threshhold", default = 0.4)
    parser.add_argument("--cfg", dest = 'cfgfile', help =
                        "Config file",
                        default = "cfg/yolov3_custom.cfg", type = str)
    parser.add_argument("--weights", dest = 'weightsfile', help =
                        "weightsfile",
                        default = "yolov3_custom_final.weights", type = str)
    parser.add_argument("--reso", dest = 'reso', help =
                        "Input resolution of the network. Increase to increase accuracy. Decrease to increase speed",
                        default = "416", type = str)
    parser.add_argument("--video", dest = "videofile", help = "Video file to     run detection on", default = "video.avi", type = str)

    return parser.parse_args()

args = arg_parse()
batch_size = int(args.bs)
confidence = float(args.confidence)
nms_thesh = float(args.nms_thresh)
start = 0
CUDA = torch.cuda.is_available()



num_classes = 5
classes = load_classes("data/classes.names")




#Set up the neural network
print("Loading network.....")
model = Darknet(args.cfgfile)
model.load_weights(args.weightsfile)
print("Network successfully loaded")

model.net_info["height"] = args.reso
inp_dim = int(model.net_info["height"])
assert inp_dim % 32 == 0
assert inp_dim > 32

#If there's a GPU availible, put the model on GPU
if CUDA:
    model.cuda()


#Set the model in evaluation mode
model.eval()

cc = ColorCamera()
# rs = RealsenseCamera()

def write(x, results, depth):
    if np.any(results):
        c1 = tuple(x[1:3].to(torch.int).tolist())
        c2 = tuple(x[3:5].to(torch.int).tolist())
        c3 = tuple(x[3:5].to(torch.int).tolist())

        center_x = int((c3[0]-c1[0])/2 + c1[0])
        center_y = int((c1[1] - c3[1])/2 + c3[1])

        img = results
        cls = int(x[-1])
        color = random.choice(colors)
        label = "{0}".format(classes[cls])
        cv2.rectangle(img, c1, c2, color, 1)
        t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1, 1)[0]
        c2 = c1[0] + t_size[0] + 3, c1[1] + t_size[1] + 4
        #cv2.circle(frame, (c1[0], c1[1]), 10, (0, 255, 0), 1)
        #cv2.circle(frame, (c3[0], c3[1]), 10, (0, 255, 255), 1)
        cv2.circle(frame, (center_x, center_y), 5, (255, 0, 0), 1)

        distance = depth[center_y, center_x]
        print(distance)
        #external matrix and intrinsic matrix
        external_matrix = np.array([[ 0.99943164 ,-0.01196106 , 0.0315172 ,-151.50923769],
 [-0.00190691 , 0.91337939,  0.40710496, 35.17481235],
 [-0.03365657, -0.40693368 , 0.91283745, 472.91520004],
                                    [0,0,0,1]])
        intrinsic_matrix = np.array([[383.80123901 ,  0.     ,    314.93920898],
                                     [  0.      ,   383.30053711 ,243.23779297],
                                     [0.      ,     0.     ,      1.        ]])

        tran = np.linalg.inv(external_matrix)
        mtx_inv = np.linalg.inv(intrinsic_matrix)
        target_coord_pixel = np.array([center_x, center_y, 1])
        target_coord_mm = mtx_inv.dot(target_coord_pixel)*distance

        target_coord_mm = np.concatenate((target_coord_mm, np.array([1])))
        coords_board = tran.dot(target_coord_mm)

        coords_board[0] = coords_board[0]+105
        coords_board[1] = -coords_board[1]+170
        coords_board[2] = -coords_board[2]+6

        print(label, coords_board[0], coords_board[1], coords_board[2])
        # print(label, center_x, center_y)
       # cv2.putText(frame, "{}mm".format(distance), (center_x, center_y - 20), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 0), 2)

        cv2.rectangle(img, c1, c2, color, -1)
        cv2.putText(img, label, (c1[0], c1[1] + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 1, [225,255,255], 1);
        return img




frames = 0
start = time.time()

while cc:
    ret, frame, depth = cc.get_frame()

    if ret:
        img = prep_image(frame, inp_dim)
#        cv2.imshow("a", frame)
        im_dim = frame.shape[1], frame.shape[0]
        im_dim = torch.FloatTensor(im_dim).repeat(1,2)

        if CUDA:
            im_dim = im_dim.cuda()
            img = img.cuda()

        with torch.no_grad():
            output = model(Variable(img, volatile = True), CUDA)
        output = write_results(output, confidence, num_classes, nms_conf = nms_thesh)


        if type(output) == int:
            frames += 1
            print("FPS of the video is {:5.4f}".format( frames / (time.time() - start)))
            cv2.imshow("frame", frame)
            key = cv2.waitKey(1)
            if key & 0xFF == ord('q'):
                break
            continue




        im_dim = im_dim.repeat(output.size(0), 1)
        scaling_factor = torch.min(416/im_dim,1)[0].view(-1,1)

        output[:,[1,3]] -= (inp_dim - scaling_factor*im_dim[:,0].view(-1,1))/2
        output[:,[2,4]] -= (inp_dim - scaling_factor*im_dim[:,1].view(-1,1))/2

        output[:,1:5] /= scaling_factor

        for i in range(output.shape[0]):
            output[i, [1,3]] = torch.clamp(output[i, [1,3]], 0.0, im_dim[i,0])
            output[i, [2,4]] = torch.clamp(output[i, [2,4]], 0.0, im_dim[i,1])




        classes = load_classes('data/classes.names')
        colors = pkl.load(open("pallete", "rb"))

        list(map(lambda x: write(x, frame, depth), output))

        cv2.imshow("frame", frame)
        key = cv2.waitKey(1)
        if key & 0xFF == ord('q'):
            break
        frames += 1
        print(time.time() - start)
        print("FPS of the video is {:5.2f}".format( frames / (time.time() - start)))
    else:
        break
