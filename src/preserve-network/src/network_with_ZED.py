#!/usr/bin/env python

### Import packages used
import sys
import rospy
from sensor_msgs.msg import Image
import cv2
from cv_bridge import CvBridge, CvBridgeError

from operator import mul
import threading

import numpy as np
from time import time

# YOU SHOULD SELECT THE MODULES YOU WANT TO USE
# Currently, there are two module versions, which are
#     - modules_IROS
#     - modules_CVPR
# If you want to use the IROS version, you should write as follows:
#     - from modules_IROS import FCN_NHWC
#     - from modules_IROS.ringbuff import RingBuffer
from modules_CVPR import FCN_NHWC
from modules_CVPR.ringbuff import RingBuffer

import tensorflow as tf

##### DEFINITIONS

#### ZED RESOLUTION
ZED_CAMERA_RESOLUTION = 2
if ZED_CAMERA_RESOLUTION == 0:
    # 1. RESOLUTION_HD2K
    CAM_WIDTH = 2208
    CAM_HEIGHT = 1242
    CAM_CHANNELS = 3
elif ZED_CAMERA_RESOLUTION == 1:
    # 2. RESOLUTION_HD1280
    CAM_WIDTH = 1920
    CAM_HEIGHT = 1080
    CAM_CHANNELS = 3
elif ZED_CAMERA_RESOLUTION == 2:
    # 3. RESOLUTION_HD720
    CAM_WIDTH = 1280
    CAM_HEIGHT = 720
    CAM_CHANNELS = 3
elif ZED_CAMERA_RESOLUTION == 3:
    # 4. RESOLUTION_VGA
    CAM_WIDTH = 672
    CAM_HEIGHT = 376
    CAM_CHANNELS = 3
else:
   raise Exception('Invalid CAMERA MODE')
IMG_CHANNELS = 3

#### IMG SIZE for each step
RATE = 1.5
#--- Step 1
IMG_WIDTH_1 = 48
IMG_HEIGHT_1 = 32
IMG_SHAPE_1 = (IMG_HEIGHT_1, IMG_WIDTH_1, IMG_CHANNELS)
IMG_SIZE_1 = reduce(mul, IMG_SHAPE_1) # mul : IMG_HEIGHT*IMG_WIDTH*IMG_CHANNELS
#--- Step 2
IMG_WIDTH_2 = int((RATE*IMG_WIDTH_1)-(RATE*IMG_WIDTH_1)%4+4)
IMG_HEIGHT_2 = int(3*IMG_WIDTH_2/4)
IMG_SHAPE_2 = (IMG_HEIGHT_2, IMG_WIDTH_2, IMG_CHANNELS)
IMG_SIZE_2 = reduce(mul, IMG_SHAPE_2)
#--- Step 3
IMG_WIDTH_3 = int((RATE*IMG_WIDTH_2)-(RATE*IMG_WIDTH_2)%4+4)
IMG_HEIGHT_3 = int(3*IMG_WIDTH_3/4)
IMG_SHAPE_3 = (IMG_HEIGHT_3, IMG_WIDTH_3, IMG_CHANNELS)
IMG_SIZE_3 = reduce(mul, IMG_SHAPE_3)
#--- Step 4
IMG_WIDTH_4 =int((RATE*IMG_WIDTH_3)-(RATE*IMG_WIDTH_3)%4+4)
IMG_HEIGHT_4 = int(3*IMG_WIDTH_4/4)
IMG_SHAPE_4 = (IMG_HEIGHT_4, IMG_WIDTH_4, IMG_CHANNELS)
IMG_SIZE_4 = reduce(mul, IMG_SHAPE_4)
#--- ORIGINAL IMAGE
IMG_SHAPE_ORIGIN = (CAM_HEIGHT, CAM_WIDTH, CAM_CHANNELS)
IMG_SIZE_ORIGIN = reduce(mul, IMG_SHAPE_ORIGIN)
#--- DEPTH IMAGE
IMG_SHAPE_DEPTH = (CAM_HEIGHT, CAM_WIDTH, 1)
IMG_SIZE_DEPTH = reduce(mul, IMG_SHAPE_DEPTH)

print IMG_WIDTH_4
print IMG_HEIGHT_4
#### 

#### PARAMETERS for CROP
#--- parameters for crop filter
(winW, winH) = (10, 10)
CROP_SHAPE = ((winW),(winH),3)
CROP_SIZE = reduce(mul, CROP_SHAPE)

#--- Parameters for cropping the final results
MARGIN_WIDTH = 10
MARGIN_HEIGHT = int(MARGIN_WIDTH*(float(CAM_HEIGHT)/float(CAM_WIDTH)))
CROPPED_RESULT_SHAPE = (CAM_HEIGHT-2*MARGIN_HEIGHT, CAM_WIDTH-2*MARGIN_WIDTH, IMG_CHANNELS)
CROPPED_IMAGE_SIZE = reduce(mul, CROPPED_RESULT_SHAPE)

#### PARAMETERS fo BUFFERS
BUFF_NUM_ELEMENT= 20
CROP_BUFF_NUM_ELEMENT = 500

IMGBUFF_SIZE_1 = IMG_SIZE_1 * BUFF_NUM_ELEMENT
cropoutBUFF_SIZE_1 = CROP_SIZE * CROP_BUFF_NUM_ELEMENT

IMGBUFF_SIZE_2 = IMG_SIZE_2 * BUFF_NUM_ELEMENT
cropoutBUFF_SIZE_2 = CROP_SIZE * CROP_BUFF_NUM_ELEMENT

IMGBUFF_SIZE_3 = IMG_SIZE_3 * BUFF_NUM_ELEMENT
cropoutBUFF_SIZE_3 = CROP_SIZE * CROP_BUFF_NUM_ELEMENT

IMGBUFF_SIZE_4 = IMG_SIZE_4 * BUFF_NUM_ELEMENT
cropoutBUFF_SIZE_4 = CROP_SIZE * CROP_BUFF_NUM_ELEMENT

IMGBUFF_ORIGIN = IMG_SIZE_ORIGIN * BUFF_NUM_ELEMENT

IMGBUFF_RESULT = IMG_SIZE_ORIGIN * BUFF_NUM_ELEMENT

DEPTHBUFF = IMG_SIZE_DEPTH * BUFF_NUM_ELEMENT

#### BUFFER INITIALIZATION
#--- BUFFERS for IMAGES 
img_buffer_1 = RingBuffer(IMGBUFF_SIZE_1, 'float32')
img_buffer_lock_1 = threading.RLock()

img_buffer_2 = RingBuffer(IMGBUFF_SIZE_2, 'float32')
img_buffer_lock_2 = threading.RLock()

img_buffer_3 = RingBuffer(IMGBUFF_SIZE_3, 'float32')
img_buffer_lock_3 = threading.RLock()

img_buffer_4 = RingBuffer(IMGBUFF_SIZE_4, 'float32')
img_buffer_lock_4 = threading.RLock()

img_buffer_origin = RingBuffer(IMGBUFF_ORIGIN, 'float32')
img_buffer_origin_lock = threading.RLock()

img_depth_buffer = RingBuffer(DEPTHBUFF, 'float32')
img_depth_buffer_lock = threading.RLock()

#--- BUFFERS for faces detected by network for each step
cropout_buffer_1 = RingBuffer(cropoutBUFF_SIZE_1, 'int')
cropout_buffer_lock_1 = threading.RLock()

cropout_buffer_2 = RingBuffer(cropoutBUFF_SIZE_2, 'int')
cropout_buffer_lock_2 = threading.RLock()

cropout_buffer_3 = RingBuffer(cropoutBUFF_SIZE_3, 'int')
cropout_buffer_lock_3 = threading.RLock()

cropout_buffer_4 = RingBuffer(cropoutBUFF_SIZE_4, 'int')
cropout_buffer_lock_4 = threading.RLock()

#--- BUFFERS including number of boxes after NMS
NMS_buffer_1 = RingBuffer(cropoutBUFF_SIZE_1, 'int')
NMS_buffer_lock_1 = threading.RLock()

NMS_buffer_2 = RingBuffer(cropoutBUFF_SIZE_2, 'int')
NMS_buffer_lock_2 = threading.RLock()

NMS_buffer_3 = RingBuffer(cropoutBUFF_SIZE_3, 'int')
NMS_buffer_lock_3 = threading.RLock()

NMS_buffer_4 = RingBuffer(cropoutBUFF_SIZE_4, 'int')
NMS_buffer_lock_4 = threading.RLock()

#--- BUFFERS for Final Results
RESULT_buffer = RingBuffer(IMGBUFF_ORIGIN, 'float32')
RESULT_buffer_lock = threading.RLock()

RESULT_buffer_depth = RingBuffer(DEPTHBUFF, 'float32')
RESULT_buffer_depth_lock = threading.RLock()

#### INITIALIZATION of NETWORK 
#--- PARAMETERS for input and output of NETWORK
keep_probability = tf.placeholder(tf.float32, name="keep_probabilty")
image = tf.placeholder(tf.float32, shape=[None, None, None,3], name="input_image")
nms_boxes, bb_xy_of_maxScore = FCN_NHWC.inference(image, keep_probability)

#--- set gpu memory usage
with tf.Session() as sess:
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 1
    sess=tf.Session(config=config)
    sess.run(tf.initialize_all_variables())
    print "Initialization End"

graph = tf.get_default_graph()

#### GLOBAL VARIABLE
BUFFERED_ORIGINAL_IMAGE = 0
BUFFERED_DEPTH_IMAGE = 0
NUM_PUBLISH = 0

#### THREADS
#--- FUNCTIONS running in THREADS
#- FUNCTIONS for STEP 1
def img_thread_func():
    while network_thread_running or img_buffer_1.nb_data > 0:
        if img_buffer_1.nb_data > 0:
            with img_buffer_lock_1:
                imgs = img_buffer_1.get(IMG_SIZE_1)
                img_buffer_1.pop(IMG_SIZE_1)

            imgs_reshape = imgs.reshape((imgs.size / IMG_SIZE_1,) + IMG_SHAPE_1)
            feed_dict = {image: imgs_reshape, keep_probability: 1.0}

            global graph
            with graph.as_default():
                nms_boxes_r, bb_xy_of_maxScore_r = sess.run([nms_boxes, bb_xy_of_maxScore], feed_dict=feed_dict)

            if np.array(nms_boxes_r).shape[0] == 0:
                with NMS_buffer_lock_1:
                    NMS_buffer_1.push(np.array(0))
            else:
                with cropout_buffer_lock_1:
                    array_bb = np.array(nms_boxes_r)
                    img_bb = imgs_reshape[0]
                    valid_bb_cnt = 0

                    for boxes_index in xrange(array_bb.shape[0]): #array_bb.shape[0] :number of bounding boxes
                        cropping = img_bb[int(array_bb[boxes_index][0]):int(array_bb[boxes_index][2]),int(array_bb[boxes_index][1]):int(array_bb[boxes_index][3])] #y1:y2,x1:x2
                        #if cropping.shape[0] == 0:
                        #    raise Exception("Cropping error: there is no cropped data\n")
                        if array_bb[boxes_index][0] > IMG_HEIGHT_1:
                            raise Exception("(x,y) in bounding box is out of bound in terms of HEIGHT_1\n")
                        if array_bb[boxes_index][1] > IMG_WIDTH_1:
                            raise Exception("(x,y) in bounding box is out of bound in terms of WIDHT_1\n")

                        if reduce(mul,cropping.shape) == 300 and cropping.shape[0] != 0:
                            valid_bb_cnt += 1
                            cropout_buffer_1.push(np.concatenate((cropping.flatten().astype('int'),(int(array_bb[boxes_index][1]),int(array_bb[boxes_index][0]))),axis =0))
                            # cropout_buffer.push(cropping.flatten().astype('int'))
                            # cropout_buffer.push(np.array(array_bb[boxes_index][1]))
                            # cropout_buffer.push(np.array(array_bb[boxes_index][0]))
                with NMS_buffer_lock_1:
                    NMS_buffer_1.push(np.array(valid_bb_cnt))

#- FUNCTIONS for STEP 2
def img_thread_func2():
    while network_thread_running2 or (img_buffer_2.nb_data > 0):
        if img_buffer_2.nb_data > 0:
            with img_buffer_lock_2:
                imgs = img_buffer_2.get(IMG_SIZE_2)
                img_buffer_2.pop(IMG_SIZE_2)

            imgs_reshape = imgs.reshape((imgs.size / IMG_SIZE_2,) + IMG_SHAPE_2)
            feed_dict = {image: imgs_reshape, keep_probability: 1.0}

            global graph
            with graph.as_default():
                nms_boxes_r, bb_xy_of_maxScore_r = sess.run([nms_boxes, bb_xy_of_maxScore], feed_dict=feed_dict)

            if np.array(nms_boxes_r).shape[0] == 0:
                with NMS_buffer_lock_2:
                    NMS_buffer_2.push(np.array(0))
            else:
                with cropout_buffer_lock_2:

                    array_bb = np.array(nms_boxes_r)
                    img_bb = imgs_reshape[0]
                    valid_bb_cnt = 0

                    for boxes_index in xrange(array_bb.shape[0]):
                        cropping = img_bb[int(array_bb[boxes_index][0]):int(array_bb[boxes_index][2]),int(array_bb[boxes_index][1]):int(array_bb[boxes_index][3])]
                        #if cropping.shape[0] == 0:
                        #    raise Exception("Cropping error: there is no cropped data\n")
                        if array_bb[boxes_index][0] > IMG_HEIGHT_2:
                            raise Exception("(x,y) in bounding box is out of bound in HEIGHT_2\n")
                        if array_bb[boxes_index][1] > IMG_WIDTH_2:
                            raise Exception("(x,y) in bounding box is out of bound in WIDTH_2\n")
                        if reduce(mul,cropping.shape) == 300 and cropping.shape[0] != 0:
                            valid_bb_cnt += 1
                            cropout_buffer_2.push(cropping.flatten().astype('int'))
                            cropout_buffer_2.push(np.array(array_bb[boxes_index][1]))
                            cropout_buffer_2.push(np.array(array_bb[boxes_index][0]))

                with NMS_buffer_lock_2:
                    NMS_buffer_2.push(np.array(valid_bb_cnt))

#- FUNCTIONS for STEP 3
def img_thread_func3():
    while network_thread_running3 or (img_buffer_3.nb_data > 0):
        if img_buffer_3.nb_data > 0:
            with img_buffer_lock_3:
                imgs = img_buffer_3.get(IMG_SIZE_3)
                img_buffer_3.pop(IMG_SIZE_3)

            imgs_reshape = imgs.reshape((imgs.size / IMG_SIZE_3,) + IMG_SHAPE_3)
            feed_dict = {image: imgs_reshape, keep_probability: 1.0}

            global graph
            with graph.as_default():
                nms_boxes_r, bb_xy_of_maxScore_r = sess.run([nms_boxes, bb_xy_of_maxScore], feed_dict=feed_dict)

            if np.array(nms_boxes_r).shape[0] == 0:
                with NMS_buffer_lock_3:
                    NMS_buffer_3.push(np.array(0))
            else:
                with cropout_buffer_lock_3:
                    array_bb = np.array(nms_boxes_r)
                    img_bb = imgs_reshape[0]
                    valid_bb_cnt = 0

                    for boxes_index in xrange(array_bb.shape[0]):
                        cropping = img_bb[int(array_bb[boxes_index][0]):int(array_bb[boxes_index][2]),int(array_bb[boxes_index][1]):int(array_bb[boxes_index][3])]
                        #if cropping.shape[0] == 0:
                        #    raise Exception("Cropping error: there is no cropped data\n")
                        if array_bb[boxes_index][0] > IMG_HEIGHT_3:
                            raise Exception("height in bounding box is out of bound in HEIGHT_3 \n")
                        if array_bb[boxes_index][1] > IMG_WIDTH_3:
                            raise Exception("width in bounding box is out of bound in WIDHT_3\n")

                        if reduce(mul,cropping.shape) == 300 and cropping.shape[0] != 0:
                            valid_bb_cnt += 1
                            cropout_buffer_3.push(cropping.flatten().astype('int'))
                            cropout_buffer_3.push(np.array(array_bb[boxes_index][1]))
                            cropout_buffer_3.push(np.array(array_bb[boxes_index][0]))

                with NMS_buffer_lock_3:
                    NMS_buffer_3.push(np.array(valid_bb_cnt))


#- FUNCTIONS for STEP 4
def img_thread_func4():
    while network_thread_running4 or (img_buffer_4.nb_data > 0):
        if img_buffer_4.nb_data > 0:
            with img_buffer_lock_4:
                imgs = img_buffer_4.get(IMG_SIZE_4)
                img_buffer_4.pop(IMG_SIZE_4)

            imgs_reshape = imgs.reshape((imgs.size / IMG_SIZE_4,) + IMG_SHAPE_4)
            feed_dict = {image: imgs_reshape, keep_probability: 1.0}

            global graph
            with graph.as_default():
                nms_boxes_r, bb_xy_of_maxScore_r = sess.run([nms_boxes, bb_xy_of_maxScore], feed_dict=feed_dict)

            if np.array(nms_boxes_r).shape[0] == 0:
                with NMS_buffer_lock_4:
                    NMS_buffer_4.push(np.array(0))
            else:
                with cropout_buffer_lock_4:
                    array_bb = np.array(nms_boxes_r)
                    img_bb = imgs_reshape[0]
                    valid_bb_cnt = 0

                    for boxes_index in xrange(array_bb.shape[0]):
                        cropping = img_bb[int(array_bb[boxes_index][0]):int(array_bb[boxes_index][2]),int(array_bb[boxes_index][1]):int(array_bb[boxes_index][3])]
                        #if cropping.shape[0] == 0:
                        #    raise Exception("Cropping error: there is no cropped data\n")
                        if array_bb[boxes_index][0] > IMG_HEIGHT_4:
                            raise Exception("height in bounding box is out of bound in HEIGHT_4\n")
                        if array_bb[boxes_index][1] > IMG_WIDTH_4:
                            raise Exception("width in bounding box is out of bound in WIDTH_4\n")

                        if reduce(mul,cropping.shape) == 300 and cropping.shape[0] != 0:
                            valid_bb_cnt += 1
                            cropout_buffer_4.push(cropping.flatten().astype('int'))
                            cropout_buffer_4.push(np.array(array_bb[boxes_index][1]))
                            cropout_buffer_4.push(np.array(array_bb[boxes_index][0]))

                with NMS_buffer_lock_4:
                    NMS_buffer_4.push(np.array(valid_bb_cnt))

#- FUNCTION obtaining the result image
def processing_func():
    while processing_thread_running or (NMS_buffer_1.nb_data > 0
                                     and NMS_buffer_2.nb_data > 0
                                     and NMS_buffer_3.nb_data > 0
                                     and NMS_buffer_4.nb_data > 0):

        if (NMS_buffer_1.nb_data > 0 and NMS_buffer_2.nb_data > 0 and NMS_buffer_3.nb_data > 0 and NMS_buffer_4.nb_data > 0 and img_buffer_origin.nb_data > 0):
            with NMS_buffer_lock_1:
                num_boxes = NMS_buffer_1.get(1)[0]
                NMS_buffer_1.pop(1)
            with NMS_buffer_lock_2:
                num_boxes_2 = NMS_buffer_2.get(1)[0]
                NMS_buffer_2.pop(1)
            with NMS_buffer_lock_3:
                num_boxes_3 = NMS_buffer_3.get(1)[0]
                NMS_buffer_3.pop(1)
            with NMS_buffer_lock_4:
                num_boxes_4 = NMS_buffer_4.get(1)[0]
                NMS_buffer_4.pop(1)

            with img_buffer_origin_lock:
                if img_buffer_origin.nb_data == 0:
                    raise Exception("image buffer has no entry")
                original_img = img_buffer_origin.get(IMG_SIZE_ORIGIN)
                img_buffer_origin.pop(IMG_SIZE_ORIGIN)

            original_img = original_img.reshape((original_img.size / IMG_SIZE_ORIGIN,) + IMG_SHAPE_ORIGIN)
            original_img = original_img[0]

            if num_boxes_4 > 0:
                dim_5 = (int((float(CAM_HEIGHT) / float(IMG_HEIGHT_4)) * winH),
                       int((float(CAM_WIDTH) / float(IMG_WIDTH_4)) * winW))
                with cropout_buffer_lock_4:
                    for i in xrange(num_boxes_4):
                        if cropout_buffer_4.nb_data == 0:
                            raise Exception("cropout buffer4 has no entry")
                        crop_outs = cropout_buffer_4.get(CROP_SIZE)
                        cropout_buffer_4.pop(CROP_SIZE)
                        crop_xy = cropout_buffer_4.get(2)
                        cropout_buffer_4.pop(2)

                        crop_outs = crop_outs.reshape((crop_outs.size / CROP_SIZE,) + CROP_SHAPE)

                        crop_outs_5 = cv2.resize(crop_outs[0], dim_5, interpolation=cv2.INTER_NEAREST)


                        # Overlay the original image by crop_outs_5
                        crop_x = int(crop_xy[0] * float(CAM_WIDTH) / float(IMG_WIDTH_4))
                        crop_y = int(crop_xy[1] * float(CAM_HEIGHT) / float(IMG_HEIGHT_4))

                        try:
                            original_img[crop_y:crop_y + crop_outs_5.shape[0], crop_x:crop_x + crop_outs_5.shape[1]] = crop_outs_5
                        except ValueError as e:
                            print e
                            print crop_outs_5.shape, crop_y, crop_x

                        #cv2.rectangle(original_img,
                        #              (crop_x, crop_y),
                        #              (crop_x + crop_outs_5.shape[1], crop_y + crop_outs_5.shape[0]),
                        #              (255, 51, 153), 1)

            if num_boxes_3 > 0:
                dim_5 = (int((float(CAM_HEIGHT) / float(IMG_HEIGHT_3)) * winH),
                       int((float(CAM_WIDTH) / float(IMG_WIDTH_3)) * winW))

                with cropout_buffer_lock_3:
                    for i in xrange(num_boxes_3):
                        if cropout_buffer_3.nb_data == 0:
                            raise Exception("cropout buffer3 has no entry")
                        crop_outs = cropout_buffer_3.get(CROP_SIZE)
                        cropout_buffer_3.pop(CROP_SIZE)
                        crop_xy = cropout_buffer_3.get(2)
                        cropout_buffer_3.pop(2)

                        crop_outs = crop_outs.reshape((crop_outs.size / CROP_SIZE,) + CROP_SHAPE)

                        crop_outs_5 = cv2.resize(crop_outs[0], dim_5, interpolation=cv2.INTER_NEAREST)
                       
                        # Overlay the original image by crop_outs
                        crop_x = int(crop_xy[0] * float(CAM_WIDTH) / float(IMG_WIDTH_3))
                        crop_y = int(crop_xy[1] * float(CAM_HEIGHT) / float(IMG_HEIGHT_3))
                        
                        try:
                            original_img[crop_y:crop_y + crop_outs_5.shape[0], crop_x:crop_x + crop_outs_5.shape[1]] = crop_outs_5
                        except ValueError as e:
                            print e
                            print crop_outs_5.shape, crop_y, crop_x

                        #cv2.rectangle(original_img,
                        #              (crop_x, crop_y),
                        #              (crop_x + crop_outs_5.shape[1], crop_y + crop_outs_5.shape[0]),
                        #              (255, 255, 0), 1)

            if num_boxes_2 > 0:
                dim_5 = (int((float(CAM_HEIGHT) / float(IMG_HEIGHT_2)) * (winH)),
                       int((float(CAM_WIDTH) / float(IMG_WIDTH_2)) * (winW)))

                with cropout_buffer_lock_2:
                    for i in xrange(num_boxes_2):
                        if cropout_buffer_2.nb_data == 0:
                            raise Exception("cropout buffer2 has no entry")
                        crop_outs = cropout_buffer_2.get(CROP_SIZE)
                        cropout_buffer_2.pop(CROP_SIZE)
                        crop_xy = cropout_buffer_2.get(2)
                        cropout_buffer_2.pop(2)

                        crop_outs = crop_outs.reshape((crop_outs.size / CROP_SIZE,) + CROP_SHAPE)

                        crop_outs_5 = cv2.resize(crop_outs[0], dim_5, interpolation=cv2.INTER_NEAREST)

                        # Overlay the original image by crop_outs_5
                        crop_x = int(crop_xy[0] * float(CAM_WIDTH) / float(IMG_WIDTH_2))
                        crop_y = int(crop_xy[1] * float(CAM_HEIGHT) / float(IMG_HEIGHT_2))
                        try:
                            original_img[crop_y:crop_y + crop_outs_5.shape[0], crop_x:crop_x + crop_outs_5.shape[1]] = crop_outs_5
                        except ValueError as e:
                            print e
                            print crop_outs_5.shape, crop_y, crop_x
                        #cv2.rectangle(original_img,
                        #              (crop_x, crop_y),
                        #              (crop_x + crop_outs_5.shape[1], crop_y + crop_outs_5.shape[0]),
                        #              (0, 255, 255), 1)

            if num_boxes > 0:
                dim_5 = (int((float(CAM_HEIGHT)/float(IMG_HEIGHT_1))*(winH)),int((float(CAM_WIDTH)/float(IMG_WIDTH_1))*(winW)))

                with cropout_buffer_lock_1:
                    for i in xrange(num_boxes):
                        if cropout_buffer_1.nb_data == 0:
                            raise Exception("cropout buffer has no entry")
                        crop_outs = cropout_buffer_1.get(CROP_SIZE)
                        cropout_buffer_1.pop(CROP_SIZE)
                        crop_xy = cropout_buffer_1.get(2)
                        cropout_buffer_1.pop(2)

                        crop_outs = crop_outs.reshape((crop_outs.size / CROP_SIZE,) + CROP_SHAPE)

                        crop_outs_5 = cv2.resize(crop_outs[0], dim_5, interpolation=cv2.INTER_NEAREST)

                        # Overlay the original image by crop_outs_5
                        crop_x = int(crop_xy[0]*float(CAM_WIDTH)/float(IMG_WIDTH_1))
                        crop_y = int(crop_xy[1]*float(CAM_HEIGHT)/float(IMG_HEIGHT_1))
                        try:
                            original_img[crop_y:crop_y + crop_outs_5.shape[0], crop_x:crop_x + crop_outs_5.shape[1]] = crop_outs_5
                        except ValueError as e:
                            print e
                            print crop_outs_5.shape, crop_y, crop_x
                        #cv2.rectangle(original_img,
                        #              (crop_x, crop_y),
                        #              (crop_x + crop_outs_5.shape[1], crop_y + crop_outs_5.shape[0]),
                        #              (0, 255, 0), 1)

            original_img_with_mosaic = original_img
            
            with img_depth_buffer_lock:
                depth_image = img_depth_buffer.get(IMG_SIZE_DEPTH)
                img_depth_buffer.pop(IMG_SIZE_DEPTH)

            with RESULT_buffer_lock:
                RESULT_buffer.push(original_img_with_mosaic.flatten())
            with RESULT_buffer_depth_lock:
                RESULT_buffer_depth.push(depth_image)            



#- Function to publish the result image
def resultPublisher(ResPub):
    global publish_thread_running
    global NUM_PUBLISH

    while publish_thread_running:
        if RESULT_buffer.nb_data > 0 and  RESULT_buffer_depth.nb_data > 0:
            with RESULT_buffer_lock:
                popped_image = RESULT_buffer.get(IMG_SIZE_ORIGIN)
                RESULT_buffer.pop(IMG_SIZE_ORIGIN)
            popped_image = popped_image.reshape(IMG_SHAPE_ORIGIN).astype('uint8')
            popped_image = cv2.cvtColor(popped_image, cv2.COLOR_RGB2BGR)

            with RESULT_buffer_depth_lock:
                popped_depth_image = RESULT_buffer_depth.get(IMG_SIZE_DEPTH)
                RESULT_buffer_depth.pop(IMG_SIZE_DEPTH)
            popped_depth_image = popped_depth_image.reshape(IMG_SHAPE_DEPTH).astype('float32')

            try:
                ResPub.imageResult_pub.publish(ResPub.bridge.cv2_to_imgmsg(popped_image, 'bgr8'))
                ResPub.depthResult_pub.publish(ResPub.bridge.cv2_to_imgmsg(popped_depth_image, "32FC1"))
                NUM_PUBLISH += 1
                print ('Number of publish: ', NUM_PUBLISH)
            except CvBridgeError as e:
                print('Exception Occurs...')
                print(e)

def report_data_in_buffer():
    print("Num of Data in img_buffer_origin: ", img_buffer_origin.nb_data/IMG_SIZE_ORIGIN)
    print("Num of Data in img_depth_buffer: ", img_depth_buffer.nb_data/IMG_SIZE_DEPTH)
    print("Num of Data in img_buffer_1 :", img_buffer_1.nb_data/IMG_SIZE_1)
    print("Num of Data in img_buffer_2 :", img_buffer_2.nb_data/IMG_SIZE_2)
    print("Num of Data in img_buffer_3 :", img_buffer_3.nb_data/IMG_SIZE_3)
    print("Num of Data in img_buffer_4 :", img_buffer_4.nb_data/IMG_SIZE_4)

#### CLASS to obtain images from ZED
class image_From_ZED:

	def __init__(self):
		self.bridge = CvBridge()
		self.imageRight_sub = rospy.Subscriber('/zed/rgb/image_rect_color', Image, self.imageRGBPushBuffer)
                self.depthImage_sub = rospy.Subscriber('zed/depth/depth_registered', Image, self.depthImagePushBuffer)
		self.imageResult_pub = rospy.Publisher('/preserve_network/result_only_Network', Image, queue_size=10)
                self.depthResult_pub = rospy.Publisher('/preserve_network/depth_only_Network', Image, queue_size=10)

        def depthImagePushBuffer(self, msg):
                try:
			depth_image = np.array(self.bridge.imgmsg_to_cv2(msg, "32FC1"), dtype=np.float32)
		except CvBridgeError as e:
			print(e)
                
                Num_Original_img = img_buffer_origin.nb_data/IMG_SIZE_ORIGIN
                Num_Depth_img = img_depth_buffer.nb_data/IMG_SIZE_DEPTH
                if (Num_Original_img - Num_Depth_img > 0):
                    with img_depth_buffer_lock:
                        img_depth_buffer.push(depth_image.flatten().astype('float32'))

	def imageRGBPushBuffer(self, msg):
		try:
			frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
		except CvBridgeError as e:
			print(e)

                ## Resize and Buffer the frame to each buffer
                #- Buffer for Original
                if (NMS_buffer_1.nb_data == 0 and NMS_buffer_2.nb_data == 0 and NMS_buffer_3.nb_data == 0 and NMS_buffer_4.nb_data == 0 and img_buffer_origin.nb_data == 0):
                    
		    with img_buffer_origin_lock:
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        img_buffer_origin.push(frame_rgb.flatten().astype('float32'))
                    #- Buffer for Step 4
                    with img_buffer_lock_4:
                        frame_resized = cv2.resize(frame, (IMG_WIDTH_4, IMG_HEIGHT_4), interpolation=cv2.INTER_AREA)
                        frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
                        img_buffer_4.push(frame_rgb.flatten().astype('float32'))  # NOTE rescaling needed
                    #- Buffer for Step 3
                    with img_buffer_lock_3:
                        frame_resized = cv2.resize(frame, (IMG_WIDTH_3, IMG_HEIGHT_3), interpolation=cv2.INTER_AREA)
                        frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
                        img_buffer_3.push(frame_rgb.flatten().astype('float32'))  # NOTE rescaling needed
                    #- Buffer for Step 2
                    with img_buffer_lock_2:
                        frame_resized = cv2.resize(frame, (IMG_WIDTH_2, IMG_HEIGHT_2), interpolation=cv2.INTER_AREA)
                        frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
                        img_buffer_2.push(frame_rgb.flatten().astype('float32'))  # NOTE rescaling needed
                    #- Buffer for Step 1
                    with img_buffer_lock_1:
                        frame_resized = cv2.resize(frame, (IMG_WIDTH_1, IMG_HEIGHT_1), interpolation=cv2.INTER_AREA)
                        frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
                        img_buffer_1.push(frame_rgb.flatten().astype('float32'))  # NOTE rescaling needed

	def unsubscribe_imageFromZED(self):
		self.imageRight_sub.unregister()

### Node initialization
rospy.init_node('zed_image_subscriber_publisher')
ZED_Image_obj = image_From_ZED()

### Threading start
#--- FLAGS for STATUS of THREADS
publish_thread_running = True
network_thread_running = True
network_thread_running2 = True
network_thread_running3 = True
network_thread_running4 = True
processing_thread_running = True
report_thread_running = True

#--- START THREADS
if report_thread_running == True:
    report_thread = threading.Thread(target=report_data_in_buffer)
    report_thread.start()
    print('start buffer_report_thread')

if publish_thread_running == True:
    publish_thread = threading.Thread(target=resultPublisher, args = (ZED_Image_obj,))
    publish_thread.start()
    print('Start publish_thread')

if network_thread_running == True:
    network_thread = threading.Thread(target=img_thread_func)
    network_thread.start()
    print('Start network_1_thread')

if network_thread_running2 == True:
    network_thread2 = threading.Thread(target=img_thread_func2)
    network_thread2.start()
    print('Start network_2_thread')

if network_thread_running3 == True:
    network_thread3 = threading.Thread(target=img_thread_func3)
    network_thread3.start()
    print('Start network_3_thread')

if network_thread_running4 == True:
    network_thread4 = threading.Thread(target=img_thread_func4)
    network_thread4.start()
    print('Start network_4_thread')

if processing_thread_running == True:
    sliding_thread = threading.Thread(target=processing_func)
    sliding_thread.start()
    print('Start processing_thread')


######

### Run nodes
try:
	rospy.spin()
except KeyboardInterrupt:
	print("Shutting down")

### Procedures for termination threads
print("Shutting down")
if report_thread_running == True:
    report_thread_running = False
    report_thread.join()
if publish_thread_running == True:
    publish_thread_running = False
    publish_thread.join()
if network_thread_running == True:
    network_thread_running = False
    network_thread.join()
if network_thread_running2 == True:
    network_thread_running2 = False
    network_thread2.join()
if network_thread_running3 == True:
    network_thread_running3 = False
    network_thread3.join()
if network_thread_running4 == True:
    network_thread_running4 = False
    network_thread4.join()
if processing_thread_running == True:
    processing_thread_running = False
    sliding_thread.join()
cv2.destroyAllWindows()
