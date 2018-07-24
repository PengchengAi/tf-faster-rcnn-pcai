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

def demo(sess, net, color_image, depth_colormap):
    """Detect object classes in an image using pre-computed object proposals."""

    # Detect all object classes and regress object bounds
    timer = Timer()
    timer.tic()
    scores, boxes = im_detect(sess, net, color_image)
    timer.toc()
    print('Detection took {:.3f}s for {:d} object proposals'.format(timer.total_time, boxes.shape[0]))

    # Visualize detections for each class
    CONF_THRESH = 0.7
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
    print("box depth:", depth_col[0], "sucker depth:", depth_col[1])
    print("box bbox:", bbox_col[0], "sucker bbox", bbox_col[1])

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

        demo(sess, net, color_image, depth_colormap)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("I'm done")
            break
