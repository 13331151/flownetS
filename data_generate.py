
# coding: utf-8

# In[2]:

get_ipython().magic(u'pylab inline')
import h5py
import tensorflow as tf
import os
import cv2


# In[15]:

import codecs
import struct
import mmap
import StringIO

def read_chars(reader, count):
    return reader.read(chars=count, size=1)

def read_float(stream):
    return struct.unpack("f", stream.read(4))[0]

def read_int(stream):
    return struct.unpack("i", stream.read(4))[0]

def parsingFlo(filename):
    fhand = codecs.open(filename, 'rb', 'utf8', 'ignore')
    stream = fhand.stream
    
    name = read_chars(fhand.reader, 4)
    width = read_int(stream)
    height = read_int(stream)
#     print name, width, height
    
    flo = np.zeros((height, width, 2))
    for r in xrange(height):
        for c in xrange(width):
            flo[r, c, 0] = read_float(stream)
            flo[r, c, 1] = read_float(stream)
            
    fhand.close()
    return (height,width), flo


# In[4]:

# images = np.zeros((8, 384, 512, 6), dtype=uint8)
# labels = np.zeros((8, 384, 512, 2))

# for i in xrange(8):
#     img1 = cv2.imread('data/000000'+str(i)+'-img0.ppm')
#     img2 = cv2.imread('data/000000'+str(i)+'-img1.ppm')
    
#     images[i] = np.concatenate([img1, img2], axis=2)
#     _, labels[i] = parsingFlo('data/000000'+str(i)+'-gt.flo')


# In[5]:

tmp = [3, 2, 1]
np.random.seed(0)
np.random.shuffle(tmp)
tmp


# In[6]:

import glob
image1_list = glob.glob("/data/kitti/FlyingChairs_release/data/*img1.ppm")
np.random.seed(0)
np.random.shuffle(image1_list)
image2_list = glob.glob("/data/kitti/FlyingChairs_release/data/*img2.ppm")
np.random.seed(0)
np.random.shuffle(image2_list)
flo_list = glob.glob("/data/kitti/FlyingChairs_release/data/*.flo")
np.random.seed(0)
np.random.shuffle(flo_list)


train_image1_list = image1_list[:20000]
train_image2_list = image2_list[:20000]
train_flo_list = flo_list[:20000]

test_image1_list = image1_list[20000:]
test_image2_list = image2_list[20000:]
test_flo_list = flo_list[20000:]


# In[16]:

def data_to_tfrecord(image1s, image2s, labels, filename):
    """ Save data into TFRecord """
    print("Converting data into %s ..." % filename)
    cwd = os.getcwd()
    writer = tf.python_io.TFRecordWriter(filename)
    
    
    for index in xrange(len(image1s)):
        if index % 1000 == 0:
            print index
        img1 = cv2.imread(image1s[index])
        img2 = cv2.imread(image2s[index])
        images = np.concatenate([img1, img2], axis=2)
        _, label = parsingFlo(labels[index])
        images = images.tobytes()
        label = np.float32(label).tobytes()
        example = tf.train.Example(features=tf.train.Features(feature={
            "label": tf.train.Feature(bytes_list=tf.train.BytesList(value=[label])),
            'input': tf.train.Feature(bytes_list=tf.train.BytesList(value=[images])),
        }))
        writer.write(example.SerializeToString())  # Serialize To String
    writer.close()


# In[14]:

data_to_tfrecord(test_image1_list, test_image2_list, test_flo_list, './test_tfrecord')
# data_to_tfrecord(test_input, test_label, 'test_tfrecord')
# data_to_tfrecord(finetune_input, finetune_label, 'finetune_tfrecord')


# In[18]:

data_to_tfrecord(train_image1_list, train_image2_list, train_flo_list, '/data/kitti/FlyingChairs_release/train_tfrecord')


# In[23]:

type(images[0][0][0][0])


# In[ ]:



