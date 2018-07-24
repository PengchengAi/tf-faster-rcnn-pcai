#!/usr/bin/env python

# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Xinlei Chen, based on code from Ross Girshick
# --------------------------------------------------------

"""
Demo script showing detections in sample images.

See README.md for installation instructions before running.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
from model.config import cfg
from model.test import im_detect
from model.nms_wrapper import nms

from utils.timer import Timer
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os, cv2
import argparse
import serial
import time

from nets.vgg16 import vgg16
from nets.resnet_v1 import resnetv1

import pyrealsense2 as rs

CLASSES = ('__background__',
           'box', 'sucker')

NETS = {'vgg16': ('vgg16_faster_rcnn_iter_2000.ckpt',),'res101': ('res101_faster_rcnn_iter_110000.ckpt',)}
DATASETS= {'pascal_voc': ('voc_2007_trainval',),'pascal_voc_0712': ('voc_2007_trainval+voc_2012_trainval',)}

def vis_detections(color_image, depth_colormap, class_col, dets_col, thresh=0.5):
    """Draw detected bounding boxes."""

    for cls_ind, class_name in enumerate(class_col):
        dets = dets_col[cls_ind]

        inds = np.where(dets[:, -1] >= thresh)[0]
        if len(inds) == 0:
            continue

        for i in inds:
            bbox = [int(e) for e in dets[i, :4]]
            score = dets[i, -1]
            
            cv2.rectangle(color_image, (bbox[0], bbox[1]),
                        (bbox[2], bbox[3]), (0, 0, 255), 3)
            cv2.rectangle(depth_colormap, (bbox[0], bbox[1]),
                        (bbox[2], bbox[3]), (0, 0, 255), 3)

            font = cv2.FONT_HERSHEY_SIMPLEX
            color_image = cv2.putText(color_image, '{:s} {:.3f}'.format(class_name, score),
                                      (bbox[0], max(bbox[1] - 2, 1)), font, 0.5, (255, 255, 255), 2)
            depth_colormap = cv2.putText(depth_colormap, '{:s} {:.3f}'.format(class_name, score),
                                      (bbox[0], max(bbox[1] - 2, 1)), font, 0.5, (255, 255, 255), 2)
    
    # Stack both images horizontally
    images = np.hstack((color_image, depth_colormap))

    # Show images
    cv2.imshow('RealSense', images)

def calc_histogram(depth_image, class_col, dets_col, thresh=0.5):
    # return value
    depth_col = np.zeros((len(class_col), 2), dtype=float)
    bbox_col = np.zeros((len(class_col), 4), dtype=float)

    # per class
    for cls_ind in range(len(class_col)):
        dets = dets_col[cls_ind]

        inds = np.where(dets[:, -1] >= thresh)[0]
        if len(inds) == 0:
            continue
        
        ind = np.argmax(dets[:, -1])

        bbox = [int(e) for e in dets[ind, :4]]
        depth_select = depth_image[bbox[1]:bbox[3], bbox[0]:bbox[2]]

        # plt.imshow(depth_select)
        # plt.colorbar()
        # plt.show()

        depth_select = np.reshape(depth_select, (-1))
        depth_index = np.array([i for i, elem in enumerate(depth_select) if elem > 1500],
                               dtype=np.int32)
        depth_select = depth_select[depth_index]
        depth_hist, bin_edge = np.histogram(depth_select, bins="fd")

        # plt.hist(depth_select, bins="fd")
        # plt.show()
        # plt.close("all")

        depth_mean = np.mean([elem for elem in depth_hist])
        front = bin_edge[0]
        end = bin_edge[-1]
        in_middle = False
        for i, elem in enumerate(depth_hist):
            if elem >= depth_mean:
                front = bin_edge[i]
                in_middle = True
            if in_middle and elem <= depth_mean:
                end = bin_edge[i]
                in_middle = False
                break

        depth_col[cls_ind, :] = np.array((front, end))
        bbox_col[cls_ind, :] = np.array((dets[ind, :4]))

    return depth_col, bbox_col

def send_cmd(ser, cmd):
    UD_MOTOR = 0x10
    DOWN_DIR = 0x10
    UP_DIR = 0x1f

    LR_MOTOR = 0x11
    LEFT_DIR = 0x10
    RIGHT_DIR = 0x1f

    FB_MOTOR = 0x12
    BACK_DIR = 0x10
    FOR_DIR = 0x1f

    EN_CMD = 0x30
    DS_CMD = 0x31

    SUCK_CMD = 0x40
    UNSUCK_CMD = 0x41

    for c in cmd:
        l = []
        read_length = 0

        # first part
        if c[0] == "enable":
            l.append(EN_CMD)
            read_length = 1
        elif c[0] == "disable":
            l.append(DS_CMD)
            read_length = 1
        elif c[0] == "suck":
            l.append(SUCK_CMD)
            read_length = 1
        elif c[0] == "unsuck":
            l.append(UNSUCK_CMD)
            read_length = 1
        elif c[0] == "up":
            l = [UD_MOTOR, UP_DIR, c[1]]
            read_length = 4
        elif c[0] == "down":
            l = [UD_MOTOR, DOWN_DIR, c[1]]
            read_length = 4
        elif c[0] == "left":
            l = [LR_MOTOR, LEFT_DIR, c[1]]
            read_length = 4
        elif c[0] == "right":
            l = [LR_MOTOR, RIGHT_DIR, c[1]]
            read_length = 4
        elif c[0] == "for":
            l = [FB_MOTOR, FOR_DIR, c[1]]
            read_length = 4
        elif c[0] == "back":
            l = [FB_MOTOR, BACK_DIR, c[1]]
            read_length = 4
        else:
            print("Unknown command.")
            continue
        
        # second part
        l.append(0xff)

        # send the command to the serial port
        ser.write(serial.to_bytes(l))
        ser.read(read_length)

        # print information
        print("Perform command:", c[0], "Send:", l)

        # sleep
        time.sleep(0.03)

def overlap(bbox1, bbox2):
    ret = True

    if bbox1[2] < bbox2[0]:
        ret = False
    elif bbox1[0] > bbox2[2]:
        ret = False
    else:
        if bbox1[3] < bbox2[1]:
            ret = False
        elif bbox1[1] > bbox2[3]:
            ret = False
        else:
            ret = True

    return ret

def in_interval(x, a, b):
    if x >= a and x <= b:
        return True
    else:
        return False

stage = 1
memory = 0
switch = False
count_no_det = 0
has_sucked = False

def move_sucker_to_target(depth_col, bbox_col, ser):
    global memory
    global switch
    global count_no_det

    UD_CO = 0.8
    LR_CO = 1.8
    FB_CO = 1.2

    TARGET_H_SUB = 17

    box_depth = depth_col[0, :]
    sucker_depth = depth_col[1, :]
    box_bbox = bbox_col[0, :]
    sucker_bbox = bbox_col[1, :]

    has_sucker_d = True
    if np.sum(sucker_depth) < 1e-5:
        has_sucker_d = False
    else:
        has_sucker_d = True
    
    # calculate the target position
    target_w = (box_bbox[0] + box_bbox[2]) / 2
    target_h = (box_bbox[1] + box_bbox[3]) / 2 - TARGET_H_SUB
    target_d = (box_depth[0] + box_depth[1]) / 2

    # calculate the sucker position
    sucker_w = (sucker_bbox[0] + sucker_bbox[2]) / 2
    sucker_h = (sucker_bbox[1] + sucker_bbox[3]) / 2
    sucker_d = (sucker_depth[0] + sucker_depth[1]) / 2

    # return if no detection for box or sucker
    if np.sum(box_bbox) < 1e-5 or np.sum(sucker_bbox) < 1e-5:
        if switch and count_no_det < 20:
            count_no_det = count_no_det + 1
            return False
        if switch and count_no_det >= 20:
            send_cmd(ser, [["for", 0x70]])
            send_cmd(ser, [["down", 0x80], ["down", 0x80], ["down", 0x80]])
            memory = memory + 1
            if memory == 3:
                send_cmd(ser, [["suck", 0xff]])
                return True
            else:
                return False
        else:
            return False

    count_no_det = 0

    if has_sucker_d and np.abs(sucker_d - target_d) < 40 and \
       np.abs(sucker_w - target_w) < 20 and np.abs(sucker_h - target_h) < 20:
        send_cmd(ser, [["suck", 0xff]])
        return True       

    elif sucker_w - target_w > 30 or sucker_w - target_w < -30:
        switch = False

        if sucker_h > target_h - 80: # raise the robot arm
            print("Up fix.")
            send_cmd(ser, [["up", 0xff], ["up", 0x80]])

        if sucker_w < target_w - 15: # move right
            value = min(0xff, int((target_w - 15 - sucker_w) * LR_CO))
            send_cmd(ser, [["left", value]])
        elif sucker_w > target_w - 15: # move left
            value = min(0xff, int((sucker_w - target_w + 15) * LR_CO))
            send_cmd(ser, [["right", value]])

        if has_sucker_d:
            if sucker_d < target_d - 30: # move backward
                value = min(0xff, int((target_d - 30 - sucker_d) * FB_CO))
                send_cmd(ser, [["back", value]])
            elif sucker_d > target_d - 30: # move forward
                value = min(0xff, int((sucker_d - target_d + 30) * FB_CO))
                send_cmd(ser, [["for", value]])  
    
    else:
        switch = True

        print("Switch on.")

        if sucker_w < target_w - 15: # move right
            value = min(0xff, int((target_w - 15 - sucker_w) * LR_CO))
            send_cmd(ser, [["left", value]])
        elif sucker_w > target_w - 15: # move left
            value = min(0xff, int((sucker_w - target_w + 15) * LR_CO))
            send_cmd(ser, [["right", value]])

        if has_sucker_d:
            if sucker_d < target_d - 30: # move backward
                value = min(0xff, int((target_d - 30 - sucker_d) * FB_CO))
                send_cmd(ser, [["back", value]])
            elif sucker_d > target_d - 30: # move forward
                value = min(0xff, int((sucker_d - target_d + 30) * FB_CO))
                send_cmd(ser, [["for", value]])

        if sucker_h < target_h: # move down
            value = min(0xff, int((target_h - sucker_h) * UD_CO))
            send_cmd(ser, [["down", value]])

    return False

def move_sucker_away(depth_col, bbox_col, ser):
    global has_sucked

    UD_CO = 0.8
    LR_CO = 1.8
    FB_CO = 1.2

    TARGET_W = 365
    TARGET_H = 300
    TARGET_D = 4200

    box_depth = depth_col[0, :]
    box_bbox = bbox_col[0, :]

    sucker_bbox = bbox_col[1, :]

    if np.sum(box_bbox) < 1e-5 or np.sum(box_depth) < 1e-5:
        return False

    if has_sucked:
        send_cmd(ser, [["up", 0xff], ["up", 0xff], ["up", 0xff], ["up", 0xff], ["up", 0xff]])
        send_cmd(ser, [["back", 0xff], ["back", 0xff]])
        has_sucked = False
        return False

    box_w = (box_bbox[0] + box_bbox[2]) / 2
    box_h = (box_bbox[1] + box_bbox[3]) / 2
    box_d = (box_depth[0] + box_depth[1]) / 2

    sucker_w = (sucker_bbox[0] + sucker_bbox[2]) / 2
    sucker_h = (sucker_bbox[1] + sucker_bbox[3]) / 2

    if np.abs(box_w - TARGET_W) <= 40 and np.abs(box_h - TARGET_H) <= 40 and \
       np.abs(box_d - TARGET_D) <= 80:
        send_cmd(ser, [["down", 0xff], ["down", 0xff], ["down", 0xff], ["down", 0xff], ["down", 0xff]])
        time.sleep(1)
        send_cmd(ser, [["unsuck", 0xff]])
        time.sleep(1)
        send_cmd(ser, [["up", 0xff], ["up", 0xff], ["up", 0xff], ["up", 0xff],  ["up", 0xff]])

        return True
    else:
        if np.sum(sucker_bbox) > 1e-5:
            if sucker_w > TARGET_W + 120 or sucker_h < TARGET_H - 120:
                send_cmd(ser, [["unsuck", 0xff]])
                return True

        if np.abs(box_w - TARGET_W) > 40:
            if box_w < TARGET_W:
                value = min(0xff, int((TARGET_W - box_w) * LR_CO))
                send_cmd(ser, [["left", value]])
            elif box_w > TARGET_W:
                value = min(0xff, int((box_w - TARGET_W) * LR_CO))
                send_cmd(ser, [["right", value]])
        
        if np.abs(box_d - TARGET_D) > 80:
            if box_d < TARGET_D:
                value = min(0xff, int((TARGET_D - box_d) * FB_CO))
                send_cmd(ser, [["back", value]])
            elif box_d > TARGET_D:
                value = min(0xff, int((box_d - TARGET_D) * FB_CO))
                send_cmd(ser, [["for", value]])

        if np.abs(box_h - TARGET_H) > 40:
            if box_h < TARGET_H:
                value = min(0xff, int((TARGET_H - box_h) * UD_CO))
                send_cmd(ser, [["down", value]])
            elif box_h > TARGET_H:
                value = min(0xff, int((box_h - TARGET_H) * UD_CO))
                send_cmd(ser, [["up", value]])

    return False


def demo(sess, net, color_image, depth_colormap, depth_image, ser):
    """Detect object classes in an image using pre-computed object proposals."""

    global stage
    global memory
    global switch
    global count_no_det
    global has_sucked

    # Detect all object classes and regress object bounds
    timer = Timer()
    timer.tic()
    scores, boxes = im_detect(sess, net, color_image)
    timer.toc()
    # print('Detection took {:.3f}s for {:d} object proposals'.format(timer.total_time, boxes.shape[0]))

    # Visualize detections for each class
    CONF_THRESH = 0.05
    NMS_THRESH = 0.3
    dets_col = []
    cls_col = []
    for cls_ind, cls in enumerate(CLASSES[1:]):
        cls_ind += 1 # because we skipped background
        cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)
        keep = nms(dets, NMS_THRESH)
        dets = dets[keep, :]
        dets_col.append(dets)
        cls_col.append(cls)

    vis_detections(color_image, depth_colormap, cls_col, dets_col, thresh=CONF_THRESH)

    depth_col, bbox_col = calc_histogram(depth_image, cls_col, dets_col, thresh=CONF_THRESH)
    print("box:", depth_col[0], "sucker:", depth_col[1])

    if stage == 1:
        ret = move_sucker_to_target(depth_col, bbox_col, ser)
        if ret:
            print("Finish reaching target.")
            memory = 0
            switch = False
            count_no_det = 0
            has_sucked = True

            key = cv2.waitKey(2000)
            if (key & 0xff) == ord('e'):
                print("Exit the process.")
                send_cmd(ser, [["disable", 0xff]])
                send_cmd(ser, [["unsuck", 0xff]])
                ser.close()
                exit(0)
            elif (key & 0xff) == ord('r'):
                print("Repeat reaching target.")
                stage = 1
            else:
                print("Go to the next stage.")
                stage = 2
    
    elif stage == 2:
        ret = move_sucker_away(depth_col, bbox_col, ser)
        if ret:
            print("Finish moving the sucker away.")
            has_sucked = False
            
            key = cv2.waitKey(5000)
            if (key & 0xff) == ord('e'):
                print("Exit the process.")
                send_cmd(ser, [["disable", 0xff]])
                send_cmd(ser, [["unsuck", 0xff]])
                ser.close()
                exit(0)
            else:
                print("Go to the next stage.")
                stage = 3

    elif stage == 3:
        print("Wait for next turn.")

        key = cv2.waitKey(5000)
        if (key & 0xff) == ord('e'):
            print("Exit the process.")
            send_cmd(ser, [["disable", 0xff]])
            send_cmd(ser, [["unsuck", 0xff]])
            ser.close()
            exit(0)
        elif (key & 0xff) == ord('s'):
            print("Restart reaching the target.")
            stage = 1
        else:
            print("stay in the current stage.")
            stage = 3

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Tensorflow Faster R-CNN demo')
    parser.add_argument('--net', dest='demo_net', help='Network to use [vgg16 res101]',
                        choices=NETS.keys(), default='vgg16')
    parser.add_argument('--dataset', dest='dataset', help='Trained dataset [pascal_voc pascal_voc_0712]',
                        choices=DATASETS.keys(), default='pascal_voc')
    args = parser.parse_args()

    return args

if __name__ == '__main__':
    cfg.TEST.HAS_RPN = True  # Use RPN for proposals
    args = parse_args()

    # model path
    demonet = args.demo_net
    dataset = args.dataset
    tfmodel = os.path.join('output', demonet, DATASETS[dataset][0], 'default',
                              NETS[demonet][0])


    if not os.path.isfile(tfmodel + '.meta'):
        raise IOError(('{:s} not found.\nDid you download the proper networks from '
                       'our server and place them properly?').format(tfmodel + '.meta'))

    # set config
    tfconfig = tf.ConfigProto(allow_soft_placement=True)
    tfconfig.gpu_options.allow_growth=True

    # init session
    sess = tf.Session(config=tfconfig)
    # load network
    if demonet == 'vgg16':
        net = vgg16()
    elif demonet == 'res101':
        net = resnetv1(num_layers=101)
    else:
        raise NotImplementedError
    net.create_architecture("TEST", 3,
                          tag='default', anchor_scales=[8, 16, 32])
    saver = tf.train.Saver()
    saver.restore(sess, tfmodel)

    print('Loaded network {:s}'.format(tfmodel))

    # Prepare serial port
    try:
        ser = serial.Serial('/dev/ttyUSB0', 9600, timeout=1)
        print(ser.name)
    except:
        print("Error occurs when connecting to the UART device. exit.")
        exit(-1)

    # unsuck the object
    send_cmd(ser, [["enable", 0xff]])
    send_cmd(ser, [["unsuck", 0xff]])

    # Configure depth and color streams
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    # Start streaming
    pipeline.start(config)


    while True:
        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if not depth_frame or not color_frame:
            continue

        # Convert images to numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

        cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)

        demo(sess, net, color_image, depth_colormap, depth_image, ser)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("I'm done")
            send_cmd(ser, [["disable", 0xff]])
            send_cmd(ser, [["unsuck", 0xff]])
            # close serial port
            ser.close()
            break
