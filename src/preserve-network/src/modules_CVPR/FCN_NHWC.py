import tensorflow as tf
import numpy as np
import os
import TensorflowUtils as utils
from six.moves import xrange

FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_integer("batch_size", "1", "batch size for training")
tf.flags.DEFINE_string("logs_dir", "/home/nvidia/catkin_ws/src/preserve-network/src/modules_CVPR/", "path to logs directory")
tf.flags.DEFINE_string("logs_dir_vali", "./", "path to logs directory")
tf.flags.DEFINE_string("data_dir", "DNN/\[7\]_FCN_7by7/Face_10by10/", "path to dataset")
tf.flags.DEFINE_float("learning_rate", "1e-4", "Learning rate for Adam Optimizer")
tf.flags.DEFINE_bool('debug', "False", "Debug mode: True/ False")
tf.flags.DEFINE_string('mode', "test", "Mode train/ test/ visualize")
tf.flags.DEFINE_string("weight", "weight_1024_1.npy","the latest weight saved")


MAX_ITERATION = int(11) # need to be changed (size of dataset)
NUM_OF_CLASSESS = 2

# Check if there is a saved weight file.
FILE_EXISTENCE = os.path.exists(FLAGS.logs_dir+FLAGS.weight)

def inference(image, keep_prob, win, threshold):

    with tf.variable_scope("inference"):

        var_dict = {}

        if FILE_EXISTENCE:
            weight_dict = np.load(FLAGS.logs_dir + FLAGS.weight, encoding='latin1').item()
            load_flag = True
            print ("-----------------Successfully load the saved weight!!!--------------------")
        else:
           load_flag = False
           print ("...................Fail to load the saved weight....................")

        # VGG19
        if load_flag == False:
            conv1_1_w = utils.weight_variable([3,3,3,64], name="conv1_1_w")
            conv1_1_b = utils.bias_variable([64], name="conv1_1_b")
        else:
            conv1_1_w = utils.get_constant(weight_dict[("conv1_1_w")], name="conv1_1_w")
            conv1_1_b = utils.get_constant(weight_dict[("conv1_1_b")], name="conv1_1_b")
        conv1_1 = utils.conv2d_basic(image, conv1_1_w, conv1_1_b)
        relu1_1 = tf.nn.relu(conv1_1, name="relu1_1")

        var_dict[("conv1_1_w")] = conv1_1_w
        var_dict[("conv1_1_b")] = conv1_1_b

        if load_flag == False:
            conv1_2_w = utils.weight_variable([3, 3, 64, 64], name="conv1_2_w")
            conv1_2_b = utils.bias_variable([64], name="conv1_2_b")
        else:
            conv1_2_w = utils.get_constant(weight_dict[("conv1_2_w")], name="conv1_2_w")
            conv1_2_b = utils.get_constant(weight_dict[("conv1_2_b")], name="conv1_2_b")
        conv1_2 = utils.conv2d_basic(relu1_1, conv1_2_w, conv1_2_b)
        relu1_2 = tf.nn.relu(conv1_2, name="relu1_2")

        var_dict[("conv1_2_w")] = conv1_2_w
        var_dict[("conv1_2_b")] = conv1_2_b

        pool1 = utils.max_pool_2x2(relu1_2)

        if load_flag == False:
            conv2_1_w = utils.weight_variable([3, 3, 64, 128], name="conv2_1_w")
            conv2_1_b = utils.bias_variable([128], name="conv2_1_b")
        else:
            conv2_1_w = utils.get_constant(weight_dict[("conv2_1_w")], name="conv2_1_w")
            conv2_1_b = utils.get_constant(weight_dict[("conv2_1_b")], name="conv2_1_b")
        conv2_1 = utils.conv2d_basic(pool1, conv2_1_w, conv2_1_b)
        relu2_1 = tf.nn.relu(conv2_1, name="relu2_1")

        var_dict[("conv2_1_w")] = conv2_1_w
        var_dict[("conv2_1_b")] = conv2_1_b

        if load_flag == False:
            conv2_2_w = utils.weight_variable([3, 3, 128, 128], name="conv2_2_w")
            conv2_2_b = utils.bias_variable([128], name="conv2_2_b")
        else:
            conv2_2_w = utils.get_constant(weight_dict[("conv2_2_w")], name="conv2_2_w")
            conv2_2_b = utils.get_constant(weight_dict[("conv2_2_b")], name="conv2_2_b")
        conv2_2 = utils.conv2d_basic(relu2_1, conv2_2_w, conv2_2_b)
        relu2_2 = tf.nn.relu(conv2_2, name="relu2_2")

        var_dict[("conv2_2_w")] = conv2_2_w
        var_dict[("conv2_2_b")] = conv2_2_b

        pool2 = utils.max_pool_2x2(relu2_2)

        if load_flag == False:
            conv3_1_w = utils.weight_variable([3, 3, 128, 256], name="conv3_1_w")
            conv3_1_b = utils.bias_variable([256], name="conv3_1_b")
        else:
            conv3_1_w = utils.get_constant(weight_dict[("conv3_1_w")], name="conv3_1_w")
            conv3_1_b = utils.get_constant(weight_dict[("conv3_1_b")], name="conv3_1_b")
        conv3_1 = utils.conv2d_basic(pool2, conv3_1_w, conv3_1_b)
        relu3_1 = tf.nn.relu(conv3_1, name="relu3_1")

        var_dict[("conv3_1_w")] = conv3_1_w
        var_dict[("conv3_1_b")] = conv3_1_b

        if load_flag == False:
            conv3_2_w = utils.weight_variable([3, 3, 256, 256], name="conv3_2_w")
            conv3_2_b = utils.bias_variable([256], name="conv3_2_b")
        else:
            conv3_2_w = utils.get_constant(weight_dict[("conv3_2_w")], name="conv3_2_w")
            conv3_2_b = utils.get_constant(weight_dict[("conv3_2_b")], name="conv3_2_b")
        conv3_2 = utils.conv2d_basic(relu3_1, conv3_2_w, conv3_2_b)
        relu3_2 = tf.nn.relu(conv3_2, name="relu3_2")

        var_dict[("conv3_2_w")] = conv3_2_w
        var_dict[("conv3_2_b")] = conv3_2_b

        if load_flag == False:
            conv3_3_w = utils.weight_variable([3, 3, 256, 256], name="conv3_3_w")
            conv3_3_b = utils.bias_variable([256], name="conv3_3_b")
        else:
            conv3_3_w = utils.get_constant(weight_dict[("conv3_3_w")], name="conv3_3_w")
            conv3_3_b = utils.get_constant(weight_dict[("conv3_3_b")], name="conv3_3_b")
        conv3_3 = utils.conv2d_basic(relu3_2, conv3_3_w, conv3_3_b)
        relu3_3 = tf.nn.relu(conv3_3, name="relu3_3")

        var_dict[("conv3_3_w")] = conv3_3_w
        var_dict[("conv3_3_b")] = conv3_3_b

        if load_flag == False:
            conv3_4_w = utils.weight_variable([3, 3, 256, 256], name="conv3_4_w")
            conv3_4_b = utils.bias_variable([256], name="conv3_4_b")
        else:
            conv3_4_w = utils.get_constant(weight_dict[("conv3_4_w")], name="conv3_4_w")
            conv3_4_b = utils.get_constant(weight_dict[("conv3_4_b")], name="conv3_4_b")
        conv3_4 = utils.conv2d_basic(relu3_3, conv3_4_w, conv3_4_b)
        relu3_4 = tf.nn.relu(conv3_4, name="relu3_4")

        var_dict[("conv3_4_w")] = conv3_4_w
        var_dict[("conv3_4_b")] = conv3_4_b

        pool3 = utils.max_pool_2x2(relu3_4)

        if load_flag == False:
            conv4_1_w = utils.weight_variable([3, 3, 256, 512], name="conv4_1_w")
            conv4_1_b = utils.bias_variable([512], name="conv4_1_b")
        else:
            conv4_1_w = utils.get_constant(weight_dict[("conv4_1_w")], name="conv4_1_w")
            conv4_1_b = utils.get_constant(weight_dict[("conv4_1_b")], name="conv4_1_b")
        conv4_1 = utils.conv2d_basic(pool3, conv4_1_w, conv4_1_b)
        relu4_1 = tf.nn.relu(conv4_1, name="relu4_1")

        var_dict[("conv4_1_w")] = conv4_1_w
        var_dict[("conv4_1_b")] = conv4_1_b

        if load_flag == False:
            conv4_2_w = utils.weight_variable([3, 3, 512, 512], name="conv4_2_w")
            conv4_2_b = utils.bias_variable([512], name="conv4_2_b")
        else:
            conv4_2_w = utils.get_constant(weight_dict[("conv4_2_w")], name="conv4_2_w")
            conv4_2_b = utils.get_constant(weight_dict[("conv4_2_b")], name="conv4_2_b")
        conv4_2 = utils.conv2d_basic(relu4_1, conv4_2_w, conv4_2_b)
        relu4_2 = tf.nn.relu(conv4_2, name="relu4_2")

        var_dict[("conv4_2_w")] = conv4_2_w
        var_dict[("conv4_2_b")] = conv4_2_b

        if load_flag == False:
            conv4_3_w = utils.weight_variable([3, 3, 512, 512], name="conv4_3_w")
            conv4_3_b = utils.bias_variable([512], name="conv4_3_b")
        else:
            conv4_3_w = utils.get_constant(weight_dict[("conv4_3_w")], name="conv4_3_w")
            conv4_3_b = utils.get_constant(weight_dict[("conv4_3_b")], name="conv4_3_b")
        conv4_3 = utils.conv2d_basic(relu4_2, conv4_3_w, conv4_3_b)
        relu4_3 = tf.nn.relu(conv4_3, name="relu4_3")

        var_dict[("conv4_3_w")] = conv4_3_w
        var_dict[("conv4_3_b")] = conv4_3_b

        if load_flag == False:
            conv4_4_w = utils.weight_variable([3, 3, 512, 512], name="conv4_4_w")
            conv4_4_b = utils.bias_variable([512], name="conv4_4_b")
        else:
            conv4_4_w = utils.get_constant(weight_dict[("conv4_4_w")], name="conv4_4_w")
            conv4_4_b = utils.get_constant(weight_dict[("conv4_4_b")], name="conv4_4_b")
        conv4_4 = utils.conv2d_basic(relu4_3, conv4_4_w, conv4_4_b)
        relu4_4 = tf.nn.relu(conv4_4, name="relu4_4")

        var_dict[("conv4_4_w")] = conv4_4_w
        var_dict[("conv4_4_b")] = conv4_4_b

        pool4 = utils.max_pool_2x2(relu4_4)

        if load_flag == False:
            conv5_1_w = utils.weight_variable([3, 3, 512, 512], name="conv5_1_w")
            conv5_1_b = utils.bias_variable([512], name="conv5_1_b")
        else:
            conv5_1_w = utils.get_constant(weight_dict[("conv5_1_w")], name="conv5_1_w")
            conv5_1_b = utils.get_constant(weight_dict[("conv5_1_b")], name="conv5_1_b")
        conv5_1 = utils.conv2d_basic(pool4, conv5_1_w, conv5_1_b)
        relu5_1 = tf.nn.relu(conv5_1, name="relu5_1")

        var_dict[("conv5_1_w")] = conv5_1_w
        var_dict[("conv5_1_b")] = conv5_1_b

        if load_flag == False:
            conv5_2_w = utils.weight_variable([3, 3, 512, 512], name="conv5_2_w")
            conv5_2_b = utils.bias_variable([512], name="conv5_2_b")
        else:
            conv5_2_w = utils.get_constant(weight_dict[("conv5_2_w")], name="conv5_2_w")
            conv5_2_b = utils.get_constant(weight_dict[("conv5_2_b")], name="conv5_2_b")
        conv5_2 = utils.conv2d_basic(relu5_1, conv5_2_w, conv5_2_b)
        relu5_2 = tf.nn.relu(conv5_2, name="relu5_2")

        var_dict[("conv5_2_w")] = conv5_2_w
        var_dict[("conv5_2_b")] = conv5_2_b

        if load_flag == False:
            conv5_3_w = utils.weight_variable([3, 3, 512, 512], name="conv5_3_w")
            conv5_3_b = utils.bias_variable([512], name="conv5_3_b")
        else:
            conv5_3_w = utils.get_constant(weight_dict[("conv5_3_w")], name="conv5_3_w")
            conv5_3_b = utils.get_constant(weight_dict[("conv5_3_b")], name="conv5_3_b")
        conv5_3 = utils.conv2d_basic(relu5_2, conv5_3_w, conv5_3_b)
        relu5_3 = tf.nn.relu(conv5_3, name="relu5_3")

        var_dict[("conv5_3_w")] = conv5_3_w
        var_dict[("conv5_3_b")] = conv5_3_b

        if load_flag == False:
            conv5_4_w = utils.weight_variable([3, 3, 512, 512], name="conv5_4_w")
            conv5_4_b = utils.bias_variable([512], name="conv5_4_b")
        else:
            conv5_4_w = utils.get_constant(weight_dict[("conv5_4_w")], name="conv5_4_w")
            conv5_4_b = utils.get_constant(weight_dict[("conv5_4_b")], name="conv5_4_b")
        conv5_4 = utils.conv2d_basic(relu5_3, conv5_4_w, conv5_4_b)
        relu5_4 = tf.nn.relu(conv5_4, name="relu5_4")

        var_dict[("conv5_4_w")] = conv5_4_w
        var_dict[("conv5_4_b")] = conv5_4_b

        conv_final_layer = conv5_3

        pool5 = utils.max_pool_2x2(relu5_4)

        # FCN
        if load_flag == False:
            W6 = utils.weight_variable([7, 7, 512, 1024], name="W6")
            b6 = utils.bias_variable([1024], name="b6")
        else:
            W6 = utils.get_constant(weight_dict[("W6")], name="W6")
            b6 = utils.get_constant(weight_dict[("b6")], name="b6")
        conv6 = utils.conv2d_basic(pool5, W6, b6)
        relu6 = tf.nn.relu(conv6, name="relu6")
        # if FLAGS.debug:
        #     utils.add_activation_summary(relu6)
        relu_dropout6 = tf.nn.dropout(relu6, keep_prob=keep_prob)

        var_dict[("W6")] = W6
        var_dict[("b6")] = b6

        if load_flag == False:
            W7 = utils.weight_variable([1, 1, 1024, 1024], name="W7")
            b7 = utils.bias_variable([1024], name="b7")
        else:
            W7 = utils.get_constant(weight_dict[("W7")], name="W7")
            b7 = utils.get_constant(weight_dict[("b7")], name="b7")
        conv7 = utils.conv2d_basic(relu_dropout6, W7, b7)
        relu7 = tf.nn.relu(conv7, name="relu7")
        # if FLAGS.debug:
        #     utils.add_activation_summary(relu7)
        relu_dropout7 = tf.nn.dropout(relu7, keep_prob=keep_prob)

        var_dict[("W7")] = W7
        var_dict[("b7")] = b7

        if load_flag == False:
            W8 = utils.weight_variable([1, 1, 1024, NUM_OF_CLASSESS], name="W8")
            b8 = utils.bias_variable([NUM_OF_CLASSESS], name="b8")
        else:
            W8 = utils.get_constant(weight_dict[("W8")], name="W8")
            b8 = utils.get_constant(weight_dict[("b8")], name="b8")
        conv8 = utils.conv2d_basic(relu_dropout7, W8, b8)

        var_dict[("W8")] = W8
        var_dict[("b8")] = b8

        # now to upscale to actual image size
        deconv_shape1 = pool4.get_shape()
        if load_flag == False:
            W_t1 = utils.weight_variable([4, 4, deconv_shape1[3].value, NUM_OF_CLASSESS], name="W_t1")
            b_t1 = utils.bias_variable([deconv_shape1[3].value], name="b_t1")
        else:
            W_t1 = utils.get_constant(weight_dict[("W_t1")], name="W_t1")
            b_t1 = utils.get_constant(weight_dict[("b_t1")], name="b_t1")
        conv_t1 = utils.conv2d_transpose_strided(conv8, W_t1, b_t1, output_shape=tf.shape(pool4))
        fuse_1 = tf.add(conv_t1, pool4, name="fuse_1")

        var_dict[("W_t1")] = W_t1
        var_dict[("b_t1")] = b_t1

        deconv_shape2 = pool3.get_shape()
        if load_flag == False:
            W_t2 = utils.weight_variable([4, 4, deconv_shape2[3].value, deconv_shape1[3].value], name="W_t2")
            b_t2 = utils.bias_variable([deconv_shape2[3].value], name="b_t2")
        else:
            W_t2 = utils.get_constant(weight_dict[("W_t2")], name="W_t2")
            b_t2 = utils.get_constant(weight_dict[("b_t2")], name="b_t2")
        conv_t2 = utils.conv2d_transpose_strided(fuse_1, W_t2, b_t2, output_shape=tf.shape(pool3))
        fuse_2 = tf.add(conv_t2, pool3, name="fuse_2")

        var_dict[("W_t2")] = W_t2
        var_dict[("b_t2")] = b_t2

        img_shape = tf.shape(image)
        deconv_shape3 = tf.stack([img_shape[0], img_shape[1], img_shape[2], NUM_OF_CLASSESS])
        if load_flag == False:
            W_t3 = utils.weight_variable([16, 16, NUM_OF_CLASSESS, deconv_shape2[3].value], name="W_t3")
            b_t3 = utils.bias_variable([NUM_OF_CLASSESS], name="b_t3")
        else:
            W_t3 = utils.get_constant(weight_dict[("W_t3")], name="W_t3")
            b_t3 = utils.get_constant(weight_dict[("b_t3")], name="b_t3")
        conv_t3 = utils.conv2d_transpose_strided(fuse_2, W_t3, b_t3, output_shape=deconv_shape3, stride=8)

        var_dict[("W_t3")] = W_t3
        var_dict[("b_t3")] = b_t3
        prob_face_0, prob_face_1 = tf.split(tf.nn.softmax(conv_t3,dim=3), 2, 3)
        annotation_pred = tf.argmax(conv_t3, dimension=3, name="prediction")

    with tf.name_scope('Sliding'):
        kernel = tf.ones(shape=[win, win, 1, 1], dtype=tf.float32)
        sliding_result = tf.nn.conv2d(prob_face_1, kernel, [1, 1, 1, 1], padding='SAME')
        sliding_result = tf.divide(sliding_result, tf.square(tf.cast(win, dtype=tf.float32)))

    # gather all boxes
    sliding_result = tf.squeeze(sliding_result, [0, 3])

    graph_mask = tf.subtract(tf.ones(shape=[img_shape[1], img_shape[2]],dtype=tf.float64), 1-threshold)
    greater = tf.greater(tf.cast(sliding_result, dtype=tf.float64), graph_mask)

    g_bb_xy = tf.where(greater)
    g_bb_xy_left_upper = tf.subtract(g_bb_xy, tf.cast(tf.divide(win, 2), dtype=tf.int64))
    g_bb_xy_right_lower = tf.add(g_bb_xy, tf.cast(tf.divide(win, 2), dtype=tf.int64))

    g_bb_info = tf.concat([g_bb_xy_left_upper, g_bb_xy_right_lower], 1)
    g_score = tf.gather_nd(sliding_result, g_bb_xy)

    concat = tf.concat([tf.cast(g_bb_info, dtype=tf.float32), tf.reshape(g_score, shape=[tf.shape(g_bb_info)[0], 1])], 1)
    reordered = tf.gather(concat, tf.nn.top_k(concat[:, 4], k=tf.shape(g_score)[0]).indices)

    g_nms_indexes = tf.image.non_max_suppression(tf.cast(reordered[:, 0:4], tf.float32),
                                                 reordered[:, 4],
                                                 50,
                                                 iou_threshold=0.5)
    g_nms_boxes = tf.gather_nd(reordered, tf.reshape(g_nms_indexes, shape=[tf.shape(g_nms_indexes)[0], 1]))

    # gather xy info of a point where score is maximum
    g_maximum_ = tf.equal(sliding_result, tf.reduce_max(tf.cast(sliding_result, dtype=tf.float32)))
    g_bb_xy = tf.where(g_maximum_)

    return g_nms_boxes, g_bb_xy
