#!/usr/bin/env python

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

import sys
print('Tensorflow版本: ', tf.__version__)

os.environ['CUDA_VISIBLE_DEVICES']='2'

CLASSES = ('__background__',
           'yeslable', 'missingPin')

NETS = {'vgg16': ('vgg16_faster_rcnn_iter_70000.ckpt',),'res101': ('res101_faster_rcnn_iter_110000.ckpt',)}
DATASETS= {'pascal_voc': ('voc_2007_trainval',),'pascal_voc_0712': ('voc_2007_trainval+voc_2012_trainval',)}

def vis_detections(im, class_name, dets, thresh=0.5):
    """Draw detected bounding boxes."""
    inds = np.where(dets[:, -1] >= thresh)[0]
    if len(inds) == 0:
        return

    for i in inds:
        bbox = dets[i, :4]
        score = dets[i, -1]

        cv2.rectangle(im,(bbox[0],bbox[1]),(bbox[2],bbox[3]),(255, 0, 255),thickness=2)
        cv2.putText(im,class_name,(bbox[0],int(bbox[1])),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0))

    return 1

def demo(sess, net, image_name):
    """Detect object classes in an image using pre-computed object proposals."""

    # Load the demo image
    im_file = os.path.join(cfg.DATA_DIR, 'demo', image_name)
    im = cv2.imread(im_file)

    # Detect all object classes and regress object bounds
    timer = Timer()
    timer.tic()
    scores, boxes = im_detect(sess, net, im)
    
    timer.toc()
    print('Detection took {:.3f}s for {:d} object proposals'.format(timer.total_time, boxes.shape[0]))
    # Visualize detections for each class
    CONF_THRESH = 0.8
    NMS_THRESH = 0.3
    for cls_ind, cls in enumerate(CLASSES[1:]):
        cls_ind += 1 # because we skipped background
        cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)
        keep = nms(dets, NMS_THRESH)
        dets = dets[keep, :]
        vis_detections(im, cls, dets, thresh=CONF_THRESH)
        
        cv2.imwrite('./data/result/{0}'.format(image_name), im)

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
        #tfmodel="/data/wxh/workspace/tf_faster-rcnn/output/vgg16/voc_2007_trainval/default/vgg16_faster_rcnn_iter_65000.ckpt"
        raise IOError(('{:s} not found.\nDid you download the proper networks from '
                       'our server and place them properly?').format(tfmodel + '.meta'))

    print(tfmodel)
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
    """
    from tensorflow import saved_model as sm
    #graph_def = sess.graph.as_graph_def()
    export_path = '/data/wxh/workspace/tf_faster-rcnn/saved_model'
    builder = sm.builder.SavedModelBuilder(export_path)
    tf_input = {'input0' : tf.saved_model.utils.build_tensor_info(net._image),
             'input1' : tf.saved_model.utils.build_tensor_info(net._im_info)}
    tf_output = {'output0' : tf.saved_model.utils.build_tensor_info(net._predictions["cls_score"]),
              'output1' : tf.saved_model.utils.build_tensor_info(net._predictions["cls_prob"]),
              'output2' : tf.saved_model.utils.build_tensor_info(net._predictions["bbox_pred"]),
              'output3' : tf.saved_model.utils.build_tensor_info(net._predictions["rois"])}
   
    prediction_signature = (
        tf.saved_model.signature_def_utils.build_signature_def(
            tf_input,tf_output,
            method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME))
    
    builder.add_meta_graph_and_variables(sess, [tf.saved_model.tag_constants.SERVING], signature_def_map={'tf_faster_rcnn_cls': prediction_signature})
    builder.save()
    print("Saved Model succeed!")
    
    sys.exit(1)
    """
    print('Loaded network {:s}'.format(tfmodel))
    
    im_names = os.listdir('./data/demo')
    
    for im_name in im_names:
        print('------------------------------------')
        print('Demo for data/demo/{}'.format(im_name))
        demo(sess, net, im_name)
