
# coding: utf-8

# In[1]:

import tensorflow as tf
import tensorflow.contrib.slim as slim
import cv2
import os
import time
import scipy.io as sio   
import numpy as np
# gpu_id = 0
# os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)


# In[2]:

class Config(object):
    def __init__(self):
        pass
config = Config()
config.input_height = 384
config.input_width = 512
config.input_channel = 6
config.label_height = 384
config.label_width = 512
config.label_channel = 2

config.patch_size = 320
config.gradient_clip = 0.01

config.batch_size = 4
config.num_batches = 1200000
config.learning_rate = 1e-3
config.min_queue_examples = 20
config.num_threads = 16
config.EPS = 1e-8

config.train_data_path = '/data/kitti/FlyingChairs_release/train_tfrecord'
config.test_data_path = '/data/kitti/FlyingChairs_release/test_tfrecord'
config.isFinetune = False

config.vis_iter = 10000
config.print_iter = 1000
config.test_iter = 5000
config.save_iter = 50000

if config.isFinetune:
    config.learning_rate *= .5

config.save_dir ='./model_sd/'
config.log_dir = './log_sd'
config.result_dir = './result_sd'



# In[3]:

try:
    os.mkdir(config.log_dir)
except:
    if not config.isFinetune:
        os.system("rm "+config.log_dir+"/*")

try:
    os.mkdir(config.save_dir)
except:
    pass

try:
    os.mkdir(config.result_dir)
except:
    os.system("rm "+config.result_dir+"/*")
    
try:
    os.mkdir(config.result_dir+'/test')
except:
    os.system("rm "+config.result_dir+"/test/*")


# In[4]:

def read_and_decode(filename, shuffle=False):
    """ Return tensor to read from TFRecord """
    filename_queue = tf.train.string_input_producer([filename])
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'input' : tf.FixedLenFeature([], tf.string),
                                           'label' : tf.FixedLenFeature([], tf.string)
                                       })
    
    img = tf.decode_raw(features['input'], tf.uint8)
    img = tf.reshape(img, [config.input_height, config.input_width, config.input_channel])
    img = tf.random_crop(img, [config.patch_size, config.patch_size, config.input_channel], seed=0)
    img = tf.to_float(img)
    img = (img-127.)/255.
    
    dep = tf.decode_raw(features['label'], tf.float32)
    dep = tf.reshape(dep, [config.label_height, config.label_width, config.label_channel])
    dep = tf.random_crop(dep, [config.patch_size, config.patch_size, config.label_channel], seed=0)
    dep = tf.to_float(dep)

    
    return _generate_image_and_label_batch(img, dep, config.min_queue_examples,
                                           config.batch_size, shuffle = shuffle, num_threads = config.num_threads)

def _generate_image_and_label_batch(image, label, min_queue_examples,
                                    batch_size, shuffle, num_threads):
    num_preprocess_threads = num_threads
    if shuffle:
        images, labels = tf.train.shuffle_batch([image, label], batch_size=batch_size,
            num_threads=num_preprocess_threads, capacity=min_queue_examples + 3 * batch_size,
            min_after_dequeue=min_queue_examples)
    else:
        images, labels = tf.train.batch([image, label], batch_size=batch_size,
        num_threads=num_preprocess_threads, capacity=min_queue_examples + 3 * batch_size)

    return images, labels


# In[5]:

def leakyrelu(inputs, alpha=0.1):
    return tf.maximum(alpha*inputs, inputs)

def msra(ks, output_num):
    return tf.truncated_normal_initializer(mean=0, stddev=np.sqrt(2./(ks[0]*ks[1]*output_num)))

def conv(inputs, output_num, ks, stride, padding, scope, alpha=0.1):
    initializer =  msra(ks, output_num)
    conv_val = slim.conv2d(inputs, output_num, ks, stride, padding, scope=scope, activation_fn=None, weights_initializer=initializer)
    conv_val = leakyrelu(conv_val, alpha=alpha)
    return conv_val

def deconv(inputs, output_num, ks, stride, padding, scope, alpha=0.1):
    initializer = msra(ks, output_num)
    conv_val = slim.conv2d_transpose(inputs, output_num, ks, stride, padding, scope=scope, activation_fn=None, weights_initializer=initializer)
    conv_val = leakyrelu(conv_val, alpha=alpha)
    return conv_val

def epe_loss(predict, labels):
    return tf.reduce_mean(tf.sqrt(tf.reduce_sum((predict-labels)**2, axis=3)+config.EPS))


# In[6]:

class Model(object):
    def __init__(self):
#         
        tfconfig = tf.ConfigProto(allow_soft_placement=True)
        tfconfig.gpu_options.allow_growth=True
        self.sess = tf.Session(config = tfconfig)
        
        self.inputs, self.labels = read_and_decode(config.train_data_path, shuffle=True)
        self.test_inputs, self.test_labels = read_and_decode(config.test_data_path, shuffle=True)
        self.labels_small_64 = tf.image.resize_images(self.labels, [config.patch_size/64, config.patch_size/64])
        self.labels_small_32 = tf.image.resize_images(self.labels, [config.patch_size/32, config.patch_size/32])
        self.labels_small_16 = tf.image.resize_images(self.labels, [config.patch_size/16, config.patch_size/16])
        self.labels_small_8 = tf.image.resize_images(self.labels, [config.patch_size/8, config.patch_size/8])
        self.labels_small_4 = tf.image.resize_images(self.labels, [config.patch_size/4, config.patch_size/4])
        
        self.test_labels_small_4 = tf.image.resize_images(self.test_labels, [config.patch_size/4, config.patch_size/4])
        
        self.summary_inputs_0 = tf.summary.image('train/input/image1', self.inputs[:, :, :, :3], max_outputs=1)
        self.summary_inputs_1 = tf.summary.image('train/input/image2', self.inputs[:, :, :, 3:], max_outputs=1)
#         self.summary_labels = tf.summary.image('train/labels', self.labels[:, :, :, :1], max_outputs=1)
#         self.summary_labels_small_4 = tf.summary.image('train/labels_small_4', self.labels_small_4[:, :, :, :1], max_outputs=1)
        
        predict_64, predict_32, predict_16, predict_8, predict_4 = self.FLOWNETS(self.inputs)
        _, _, _, _, test_predict_4 = self.FLOWNETS(self.test_inputs, True)
        self.summary_outputs = tf.summary.image('train/predict_4', predict_4[:, :, :, :1], max_outputs=1)
        
        self.predict_4 = predict_4
        
#         loss_train_4 = tf.losses.absolute_difference(predict_4, self.labels_small_4)
#         loss_train_8 = tf.losses.absolute_difference(predict_8, self.labels_small_8)
#         loss_train_16 = tf.losses.absolute_difference(predict_16, self.labels_small_16)
#         loss_train_32 = tf.losses.absolute_difference(predict_32, self.labels_small_32)
#         loss_train_64 = tf.losses.absolute_difference(predict_64, self.labels_small_64)

        loss_train_4 = epe_loss(predict_4, self.labels_small_4)
        loss_train_8 = epe_loss(predict_8, self.labels_small_8)
        loss_train_16 = epe_loss(predict_16, self.labels_small_16)
        loss_train_32 = epe_loss(predict_32, self.labels_small_32)
        loss_train_64 = epe_loss(predict_64, self.labels_small_64)
        
        loss_test_4 = epe_loss(test_predict_4, self.test_labels_small_4)
        
        self.loss_train = 0.005*loss_train_4+0.01*loss_train_8+0.02*loss_train_16+0.08*loss_train_32+0.32*loss_train_64
        self.loss_test = loss_test_4
        
        self.summary_loss_train = tf.summary.scalar('train/loss', self.loss_train)
        
        update_vars = tf.global_variables()
        print "All variables ", [v.name for v in update_vars]
        update_vars1 = []
        update_vars2 = []
        for var in update_vars:
            if 'bias' in var.name or 'deconv' in var.name:
                update_vars2.append(var)
            else:
                update_vars1.append(var)
        print "Learning rate = ", config.learning_rate, " vars: ", [v.name for v in update_vars1]
        print "Learning rate = ", config.learning_rate*.1, " vars: ", [v.name for v in update_vars2]
        
        
        self.learning_rate_mult = tf.placeholder(tf.float32)
        opt = tf.train.AdamOptimizer(learning_rate=self.learning_rate_mult*config.learning_rate)
        
        grads1 = tf.gradients(self.loss_train, update_vars1)
        grads2 = tf.gradients(self.loss_train, update_vars2)
        grads2 = [v*.1 for v in grads2]
        
        if config.gradient_clip > 0:
            grads1 = [tf.clip_by_value(v, -config.gradient_clip, config.gradient_clip) for v in grads1]
            grads2 = [tf.clip_by_value(v, -config.gradient_clip, config.gradient_clip) for v in grads2]
        
        train_op1 = opt.apply_gradients(zip(grads1, update_vars1))
        train_op2 = opt.apply_gradients(zip(grads2, update_vars2))
        self.opt = tf.group(train_op1, train_op2)
        
        self.merge_summary_train = tf.summary.merge([self.summary_loss_train])
        
    
    
            
    def FLOWNETS(self, inputs, reuse = False):
        with tf.variable_scope('FSRCNN') as scope:
            if reuse:
                scope.reuse_variables()
            
            # shrink part
            conv0 = conv(inputs, 64, [3, 3], 1, 'SAME', scope='conv0')
            conv1 = conv(conv0, 64, [3, 3], 2, 'SAME', scope='conv1')
            conv1_1 = conv(conv1, 128, [3, 3], 1, 'SAME', scope='conv1_1')
            conv2 = conv(conv1_1, 128, [3, 3], 2, 'SAME', scope='conv2')
            conv2_1 = conv(conv2, 128, [3, 3], 1, 'SAME', scope='conv2_1')
            conv3 = conv(conv2_1, 256, [3, 3], 2, 'SAME', scope='conv3')
            conv3_1 = conv(conv3, 256, [3, 3], 1, 'SAME', scope='conv3_1')
            conv4 = conv(conv3_1, 512, [3, 3], 2, 'SAME', scope='conv4')
            conv4_1 = conv(conv4, 512, [3, 3], 1, 'SAME', scope='conv4_1')
            conv5 = conv(conv4_1, 512, [3, 3], 2, 'SAME', scope='conv5')
            conv5_1 = conv(conv5, 512, [3, 3], 1, 'SAME', scope='conv5_1')
            conv6 = conv(conv5_1, 1024, [3, 3], 2, 'SAME', scope='conv6')
            conv6_1 = conv(conv6, 1024, [3, 3], 1, 'SAME', scope='conv6_1')
            # 6 * 8 flow
            predict6 = conv(conv6_1, 2, [3, 3], 1, 'SAME', scope='predict6')  #0.32
            # 12 * 16 flow
            deconv5 = deconv(conv6_1, 512, [4, 4], 2, 'SAME', scope='deconv5')
            deconvflow6 = deconv(predict6, 2, [4, 4], 2, 'SAME', scope='deconvflow6')
            concat5 = tf.concat([deconv5, conv5_1, deconvflow6], 3, name='concat5')
            interconv5 = conv(concat5, 512, [3, 3], 1, 'SAME', scope='interconv5')
            predict5 = conv(interconv5, 2, [3, 3], 1, 'SAME', scope='predict5')   #0.08
            # 24 * 32 flow
            deconv4 = deconv(concat5, 256, [4, 4], 2, 'SAME', scope='deconv4')
            deconvflow5 = deconv(predict5, 2, [4, 4], 2, 'SAME', scope='deconvflow5')
            concat4 = tf.concat([deconv4, conv4_1, deconvflow5], 3, name='concat4')
            interconv4 = conv(concat4, 256, [3, 3], 1, 'SAME', scope='interconv4')
            predict4 = conv(interconv4, 2, [3, 3], 1, 'SAME', scope='predict4')  #0.02
            # 48 * 64 flow
            deconv3 = deconv(concat4, 128, [4, 4], 2, 'SAME', scope='deconv3')
            deconvflow4 = deconv(predict4, 2, [4, 4], 2, 'SAME', scope='deconvflow4')
            concat3 = tf.concat([deconv3, conv3_1, deconvflow4], 3, name='concat3')
            interconv3 = conv(concat3, 128, [3, 3], 1, 'SAME', scope='interconv3')
            predict3 = conv(interconv3, 2, [3, 3], 1, 'SAME', scope='predict3')  #0.01
            # 96 * 128 flow
            deconv2 = deconv(concat3, 64, [4, 4], 2, 'SAME', scope='deconv2')
            deconvflow3 = deconv(predict3, 2, [4, 4], 2, 'SAME', scope='deconvflow3')
            concat2 = tf.concat([deconv2, conv2, deconvflow3], 3, name='concat2')
            interconv2 = conv(concat2, 64, [3, 3], 1, 'SAME', scope='interconv2')
            predict2 = conv(interconv2, 2, [3, 3], 1, 'SAME', scope='predict2') #0.005

            return predict6, predict5, predict4, predict3, predict2
    

  
    def train(self):
        writer = tf.summary.FileWriter(config.log_dir, tf.get_default_graph())
        saver = tf.train.Saver(max_to_keep=0)
        
        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        self.sess.run(init_op)
        
        if config.isFinetune:
            ckpt = tf.train.get_checkpoint_state(config.save_dir)
            saver.restore(self.sess, ckpt.model_checkpoint_path)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=self.sess, coord=coord)
        
        time_ = time.clock()
        
#         input_data = np.zeros((config.batch_size, config.input_height, config.input_width, config.input_channel))
#         input_label = np.zeros((config.batch_size, config.label_height, config.label_width, config.label_channel))
    
        fhand = open('progress.txt', 'wb', 0)
        learning_rate_mult = 1.
        for t in range(0, config.num_batches):
            if t % 100 == 0:
#                 print "iter ", t, " time ", time.clock()-time_
                fhand.write("iter "+str(t)+" time "+str(time.clock()-time_)+'==>')
                time_ = time.clock()
            
            if t > 2000000 and (t % 1000000 == 0):
                learning_rate_mult/=2

#             _, merge_summary = \
#                 self.sess.run([self.opt, self.merge_summary_train], feed_dict={self.inputs:input_data, self.labels:input_label, self.learning_rate:config.learning_rate})
            if t % config.vis_iter == 0:
                _, merge_summary, loss, label_, predict_ =                    self.sess.run([self.opt, self.merge_summary_train, self.loss_train, self.labels_small_4[0, :, :, 0], self.predict_4[0, :, :, 0]], feed_dict={self.learning_rate_mult:learning_rate_mult})
                cv2.imwrite(config.result_dir+'/'+str(t)+'_label.jpg', np.clip(label_*20, 0, 255).astype(uint8))
                cv2.imwrite(config.result_dir+'/'+str(t)+'_pred.jpg', np.clip(predict_*20, 0, 255).astype(uint8))
            else:
                _, merge_summary, loss =                     self.sess.run([self.opt, self.summary_loss_train, self.loss_train], feed_dict={self.learning_rate_mult:learning_rate_mult})
            
            if t%config.print_iter== 0:
#                 print "train loss is: ", loss
                fhand.write("train loss is: "+str(loss)+'\n')
                writer.add_summary(merge_summary, t)
                
            if t%config.test_iter == 0:
                loss_test = self.sess.run(self.loss_test, feed_dict={self.learning_rate_mult:learning_rate_mult})
                fhand.write('test loss is: '+str(loss_test)+'\n')
        
        
            if t%config.save_iter == 0:
                saver.save(self.sess, config.save_dir, global_step=t)
                
        coord.request_stop()
        coord.join(threads)
        
        writer.close()
        


# In[7]:

myNet = Model()
myNet.train()


# In[ ]:



