import mxnet as mx
import sys, os
import random
import time
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import numpy as np
import cv2
import pandas as pd
import re
from mxnet.io import DataIter
from mxnet.io import DataBatch
import pandas as pd
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.interpolation import map_coordinates
from collections import namedtuple
import pickle

Batch = namedtuple('Batch', ['data'])

BATCH_SIZE,INPUT_SIZE_z,INPUT_SIZE_y, INPUT_SIZE_x = 8,32,32,32
def print_inferred_shape(net):
    ar, ou, au = net.infer_shape(data=(BATCH_SIZE, 1, INPUT_SIZE_z,INPUT_SIZE_y, INPUT_SIZE_x))
    print(net.name,ou)

class FileIter0(DataIter):
    def __init__(self, path,
                 data_name="data",
                 label_name="softmax_label",
                 batch_size=32,
                 do_augment=False,
                 mean_image=None):

        
        self.epoch = 0

        
        super(FileIter0, self).__init__()
        self.batch_size = batch_size
        self.do_augment=do_augment


        #self.mean = cv2.imread(mean_image, cv2.IMREAD_GRAYSCALE)

        self.data_name = data_name
        self.label_name = label_name

        self.record = mx.recordio.MXRecordIO(path, 'r')

        
        def readrecord(record):
            record.reset()
            num_data=0
            while True:
                item = record.read()
                num_data+=1
                if not item:
                    break
            return num_data-1
        
        
        self.num_data = readrecord(self.record)#len(open(self.flist_name, 'r').readlines())
        
        self.cursor = -1
        self.record.reset()

        self.data, self.label = self._read()
        self.reset()

    def _read(self):
        """get two list, each list contains two elements: name and nd.array value"""
                
        data = {}
        label = {}

        dd = []
        ll = []
        for i in range(0, self.batch_size):
            
            item = self.record.read()            
            header, l = mx.recordio.unpack_img(item)
            
            d=header.label
            try:
                d=d.reshape((52,52,52))
                d=d.astype(float)
                

                l=l.reshape((52,52,52))
                l=l.astype(float)

                if self.do_augment==True:            
                    d,l=augment(d,l)
                else:

                    d=crop_centerz(d)
                    l=crop_centerz(l)
                    d=list(d)
                    l=list(l)
                    d=np.array([crop_center2(i,32,32) for i in d])
                    l=np.array([crop_center2(i,32,32) for i in l])
            except:
                d=d.reshape((32,32,32))
                d=d.astype(float)

                l=l.reshape((32,32,32))
                l=l.astype(float)

            d = normalize(d)
            d = np.expand_dims(d, axis=0) 
            d = np.expand_dims(d, axis=0)

            l=l.reshape((32*32*32))
            l = np.expand_dims(l, axis=0)
            l=l.astype(float)

            dd.append(d)
            ll.append(l)

        d = np.vstack(dd)
        l = np.vstack(ll)
        data[self.data_name] = d
        label[self.label_name] = l
        res = list(data.items()), list(label.items())
        return res

    @property
    def provide_data(self):
        """The name and shape of data provided by this iterator"""
        res = [(k, tuple(list(v.shape[0:]))) for k, v in self.data]
        # print "data : " + str(res)
        return res

    @property
    def provide_label(self):
        """The name and shape of label provided by this iterator"""
        res = [(k, tuple(list(v.shape[0:]))) for k, v in self.label]
        #print "label : " + str(res)
        return res
    

    def reset(self):
        self.cursor = -1
        self.record.reset()
        self.epoch += 1

    def getpad(self):
        return 0

    def iter_next(self):
        self.cursor += self.batch_size
        if self.cursor < self.num_data:
            return True
        else:
            return False

    def eof(self):
        res = self.cursor >= self.num_data
        return res

    def next(self):
        """return one dict which contains "data" and "label" """
        if self.iter_next():
            self.data, self.label = self._read()
            #for i in range(0, 10):
            #    self.data, self.label = self._read()
            #    d.append(mx.nd.array(self.data[0][1]))
            #    l.append(mx.nd.array(self.label[0][1]))
            
            res = DataBatch(data=[mx.nd.array(self.data[0][1])], label=[mx.nd.array(self.label[0][1])], pad=self.getpad(), index=None)
            #if self.cursor % 100 == 0:
            #    print "cursor: " + str(self.cursor)
            return res
        else:
            raise StopIteration


class XYRange:
    def __init__(self, x_min, x_max, y_min, y_max, chance=1.0):
        self.chance = chance
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max
        self.last_x = 0
        self.last_y = 0

    def get_last_xy_txt(self):
        res = "x_" + str(int(self.last_x * 100)).replace("-", "m") + "-" + "y_" + str(int(self.last_y * 100)).replace("-", "m")
        return res


def random_translate_img(img, xy_range, border_mode="constant"):
    if random.random() > xy_range.chance:
        return img

    org_width, org_height = img.shape[-2:]
    translate_x = random.randint(xy_range.x_min, xy_range.x_max)
    translate_y = random.randint(xy_range.y_min, xy_range.y_max)
    
    trans_matrix = np.float32([[1, 0, translate_x], [0, 1, translate_y]])
    border_const = cv2.BORDER_CONSTANT
    if border_mode == "reflect":
        border_const = cv2.BORDER_REFLECT

    res = []
    for img_inst in img:
        img_inst = cv2.warpAffine(img_inst, trans_matrix, (org_width, org_height), borderMode=border_const)
        res.append(img_inst)
    if len(res) == 1:
        res = res[0]
    xy_range.last_x = translate_x
    xy_range.last_y = translate_y
    return res


ELASTIC_INDICES = None  # needed to make it faster to fix elastic deformation per epoch.
def elastic_transform(image, alpha, sigma, random_state=None):
    global ELASTIC_INDICES
    shape = image.shape

    if ELASTIC_INDICES == None:
        if random_state is None:
            random_state = np.random.RandomState(1301)

        dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
        dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
        x, y = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]))
        ELASTIC_INDICES = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1))
    return map_coordinates(image, ELASTIC_INDICES, order=1).reshape(shape)


def crop_center(img,cropx,cropy):
    y,x = img.shape
    startx = x//2-(cropx//2)
    starty = y//2-(cropy//2)    
    return img[starty:starty+cropy,startx:startx+cropx]

def crop_centerz(img,cropz=32):
    z=img.shape[0]
    startz= z//2-(cropz//2)    
    return img[startz:startz+cropz]

def random_flip_img(img, horizontal_chance=0, vertical_chance=0):
    import cv2
    flip_horizontal = False
    if random.random() < horizontal_chance:
        flip_horizontal = True

    flip_vertical = False
    if random.random() < vertical_chance:
        flip_vertical = True

    if not flip_horizontal and not flip_vertical:
        return img

    flip_val = 1
    if flip_vertical:
        flip_val = -1 if flip_horizontal else 0

    if not isinstance(img, list):
        res = cv2.flip(img, flip_val) # 0 = X axis, 1 = Y axis,  -1 = both
    else:
        res = []
        for img_item in img:
            img_flip = cv2.flip(img_item, flip_val)
            res.append(img_flip)
            
    #print flip_val
    
    return res


def augment(imgs,labels):
    



    translate_z = random.randint(-10,10)
    m=len(imgs)
    imgz=imgs[max(0,translate_z):min(m,m+translate_z)]
    labelz=labels[max(0,translate_z):min(m,m+translate_z)]


    imgz1=crop_centerz(imgz)
    labelz1=crop_centerz(labelz)

    data11=np.concatenate((imgz1,labelz1),axis=0)



    if random.randint(0, 100) > 50:
        data11=np.array([elastic_transform(d, 128, 15) for d  in data11])



    z,x,y=data11.shape

    n,rows,cols = data11.shape
    rot =20* random.random()-10
    M = cv2.getRotationMatrix2D((cols/2,rows/2),rot,1)


    augmented00=np.array([cv2.warpAffine(d,M,(cols,rows))for d in data11])



    augmented0=np.array(random_flip_img(list(augmented00), 0.5, 0.5))

    augmented = random_translate_img(augmented0, XYRange(-10, 10, -10, 10, 0.8))



    cropx=32
    cropy=32



    new1=np.array(augmented)

    img_aug,label_aug=np.vsplit(new1,2)
    
    img_aug=np.array([crop_center(i,cropx,cropy) for i in img_aug])
    label_aug=np.array([crop_center(i,cropx,cropy) for i in label_aug])
    #label_aug=np.ceil(label_aug)
    
    return img_aug,label_aug


def crop_center2(img,cropx,cropy):
    y,x = img.shape
    startx = x//2-(cropx//2)
    starty = y//2-(cropy//2)    
    return img[starty:starty+cropy,startx:startx+cropx]


def dice_coef(label, y):
    smooth = 1.
    intersection = mx.sym.sum(label * y)
    return (2. * intersection + smooth) / (mx.sym.sum(label) + mx.sym.sum(mx.sym.abs(y)) + smooth)

def dice_coef_loss(label, y):
    return -dice_coef(label, y)

def unet_loss(l, p):
    loss=-l*np.log(p+1e-12)#-(1-l)*np.log(1-p+1e-12)
    return loss

def get_net_313():
    source = mx.sym.Variable("data")
    label = mx.sym.Variable("softmax_label")
    #print_inferred_shape(source)

    kernel_size = (3, 3, 3)
    stride=(1, 1,1)
    pad_size = (1, 1, 1)
    filter_count = 32
    net =  mx.sym.Convolution(data=source, kernel=kernel_size, stride=stride, pad=pad_size, num_filter=filter_count)
    net = mx.sym.BatchNorm(net)
    net = mx.sym.Activation(net, act_type="relu")
    #print_inferred_shape(net)
    
    
    net = mx.sym.Convolution(net, kernel=kernel_size, stride=stride, pad=pad_size, num_filter=filter_count*2)
    net = mx.sym.BatchNorm(net)
    net = mx.sym.Activation(net, act_type="relu")
    #print_inferred_shape(net)
    
    net1=net
    
    net = mx.sym.Pooling(net, pool_type="max", kernel=(2, 2,2), stride=(2,2, 2))
    #print_inferred_shape(net)
    
    net = mx.sym.Convolution(net, kernel=kernel_size, stride=stride, pad=pad_size, num_filter=filter_count*2)
    net = mx.sym.BatchNorm(net)
    net = mx.sym.Activation(net, act_type="relu")

    #print_inferred_shape(net)

    
    net =  mx.sym.Convolution(net, kernel=kernel_size, stride=stride, pad=pad_size, num_filter=filter_count*4)
    net = mx.sym.BatchNorm(net)
    net = mx.sym.Activation(net, act_type="relu")
    #print_inferred_shape(net)
    
    net2=net
    
    net = mx.sym.Pooling(net, pool_type="max", kernel=(2, 2,2), stride=(2,2, 2))
    #print_inferred_shape(net)
    
    net = mx.sym.Convolution(net, kernel=kernel_size, stride=stride, pad=pad_size, num_filter=filter_count*4)
    net = mx.sym.BatchNorm(net)
    net = mx.sym.Activation(net, act_type="relu")
    #print_inferred_shape(net)

    net = mx.sym.Convolution(net, kernel=kernel_size, stride=stride, pad=pad_size, num_filter=filter_count*8)
    net = mx.sym.BatchNorm(net)
    net = mx.sym.Activation(net, act_type="relu")
    #print "net3"
    #print_inferred_shape(net)
    
    net3=net
    net = mx.sym.Pooling(net, pool_type="max", kernel=(2, 2,2), stride=(2,2, 2))
    #print_inferred_shape(net)
    
    net = mx.sym.Convolution(net, kernel=kernel_size, stride=stride, pad=pad_size, num_filter=filter_count*8)
    net = mx.sym.BatchNorm(net)
    net = mx.sym.Activation(net, act_type="relu")
    #print_inferred_shape(net)
    
    net = mx.sym.Convolution(net, kernel=kernel_size, stride=stride, pad=pad_size, num_filter=filter_count*16)
    net = mx.sym.BatchNorm(net)
    net = mx.sym.Activation(net, act_type="relu")
    #print_inferred_shape(net)
    
    net = mx.sym.Deconvolution(net, kernel=(2, 2,2), pad=(0, 0,0), stride=(2,2, 2), num_filter=filter_count*16)
    net = mx.sym.Activation(net, act_type="relu")

    #print_inferred_shape(net)

    net = mx.sym.Concat(*[net3, net])

    #print_inferred_shape(net)
    
    net = mx.sym.Convolution(net, kernel=kernel_size, stride=stride, pad=pad_size, num_filter=filter_count*8)
    net = mx.sym.BatchNorm(net)
    net = mx.sym.Activation(net, act_type="relu")
    #print_inferred_shape(net)

    net = mx.sym.Convolution(net, kernel=kernel_size, stride=stride, pad=pad_size, num_filter=filter_count*8)
    net = mx.sym.BatchNorm(net)
    net = mx.sym.Activation(net, act_type="relu")
    #print_inferred_shape(net)
    
    net = mx.sym.Deconvolution(net, kernel=(2, 2,2), pad=(0, 0,0), stride=(2,2, 2), num_filter=filter_count*8)
    net = mx.sym.Activation(net, act_type="relu")
    #print_inferred_shape(net)
    
    net = mx.sym.Concat(*[net2, net])
    #print_inferred_shape(net)

    net = mx.sym.Convolution(net, kernel=kernel_size, stride=stride, pad=pad_size, num_filter=filter_count*4)
    net = mx.sym.BatchNorm(net)
    net = mx.sym.Activation(net, act_type="relu")
    #print_inferred_shape(net)
    
    net = mx.sym.Convolution(net, kernel=kernel_size, stride=stride, pad=pad_size, num_filter=filter_count*4)
    net = mx.sym.BatchNorm(net)
    net = mx.sym.Activation(net, act_type="relu")
    #print_inferred_shape(net)
    
    
    
    net = mx.sym.Deconvolution(net, kernel=(2, 2,2), pad=(0, 0,0), stride=(2,2, 2), num_filter=filter_count*4)
    net = mx.sym.Activation(net, act_type="relu")
    #print_inferred_shape(net)
    
    net = mx.sym.Concat(*[net1, net])
    #print_inferred_shape(net)

    net = mx.sym.Convolution(net, kernel=kernel_size, stride=stride, pad=pad_size, num_filter=filter_count*2)
    net = mx.sym.BatchNorm(net)
    net = mx.sym.Activation(net, act_type="relu")
    #print_inferred_shape(net)
    
    net = mx.sym.Convolution(net, kernel=kernel_size, stride=stride, pad=pad_size, num_filter=filter_count*2)
    net = mx.sym.BatchNorm(net)
    net = mx.sym.Activation(net, act_type="relu")
    #print_inferred_shape(net)
        
    net = mx.sym.Convolution(net, kernel=kernel_size, stride=stride, pad=pad_size, num_filter=1)
    net = mx.sym.BatchNorm(net)
    net = mx.sym.Activation(net, act_type="sigmoid")
    #print_inferred_shape(net)
    
    y = mx.symbol.Flatten(net)
    
    loss= mx.sym.MakeLoss(dice_coef_loss(label, y))
    pred_loss = mx.sym.Group([mx.sym.BlockGrad(y), loss])

    return pred_loss


def save_obj(obj, name ):
    with open( name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name ):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)
def dice_coef2(label, y):
    smooth = 1.
    label=mx.nd.array(label).as_in_context(mx.gpu(0))
    y=mx.nd.array(y).as_in_context(mx.gpu(0))
    intersection = mx.nd.sum(label*y)
    return ((2. * intersection + smooth) / (mx.nd.sum(label) +mx.nd.sum(mx.nd.abs(y)) + smooth))

def normalize(image):
    MIN_BOUND = -1000.0
    MAX_BOUND = 400.0
    image = (image - MIN_BOUND) / (MAX_BOUND - MIN_BOUND)
    image[image > 1] = 1.
    image[image < 0] = 0.
    return image




class FileIter2(DataIter):
    def __init__(self, path,
                 data_name="data",
                 label_name="softmax_label",
                 batch_size=32,
                 do_augment=True,
                 mean_image=None):

        
        self.epoch = 0

        
        super(FileIter2, self).__init__()
        self.batch_size = batch_size
        self.do_augment=do_augment


        #self.mean = cv2.imread(mean_image, cv2.IMREAD_GRAYSCALE)

        self.data_name = data_name
        self.label_name = label_name

        self.record = mx.recordio.MXRecordIO(path, 'r')

        
        def readrecord(record):
            record.reset()
            num_data=0
            while True:
                item = record.read()
                num_data+=1
                if not item:
                    break
            return num_data-1
        
        
        self.num_data = readrecord(self.record)#len(open(self.flist_name, 'r').readlines())
        
        self.cursor = -1
        self.record.reset()

        self.data, self.label = self._read()
        self.reset()

    def _read(self):
        """get two list, each list contains two elements: name and nd.array value"""
                
        data = {}
        label = {}

        dd = []
        ll = []
        for i in range(0, self.batch_size):
            
            item = self.record.read()            
            header, l = mx.recordio.unpack_img(item)
            
            d=header.label
            d=d.reshape((32,32,32))
            d=d.astype(float)

            l=l.reshape((32,32,32))
            l=l.astype(float)


 

            d = np.expand_dims(d, axis=0) 
            d = np.expand_dims(d, axis=0)

            l=l.reshape((32*32*32))
            l = np.expand_dims(l, axis=0)
            l=l.astype(float)

            dd.append(d)
            ll.append(l)

        d = np.vstack(dd)
        l = np.vstack(ll)
        data[self.data_name] = d
        label[self.label_name] = l
        res = list(data.items()), list(label.items())
        return res

    @property
    def provide_data(self):
        """The name and shape of data provided by this iterator"""
        res = [(k, tuple(list(v.shape[0:]))) for k, v in self.data]
        # print "data : " + str(res)
        return res

    @property
    def provide_label(self):
        """The name and shape of label provided by this iterator"""
        res = [(k, tuple(list(v.shape[0:]))) for k, v in self.label]
        #print "label : " + str(res)
        return res
    

    def reset(self):
        self.cursor = -1
        self.record.reset()
        self.epoch += 1

    def getpad(self):
        return 0

    def iter_next(self):
        self.cursor += self.batch_size
        if self.cursor < self.num_data:
            return True
        else:
            return False

    def eof(self):
        res = self.cursor >= self.num_data
        return res

    def next(self):
        """return one dict which contains "data" and "label" """
        if self.iter_next():
            self.data, self.label = self._read()
            #for i in range(0, 10):
            #    self.data, self.label = self._read()
            #    d.append(mx.nd.array(self.data[0][1]))
            #    l.append(mx.nd.array(self.label[0][1]))
            
            res = DataBatch(data=[mx.nd.array(self.data[0][1])], label=[mx.nd.array(self.label[0][1])], pad=self.getpad(), index=None)
            #if self.cursor % 100 == 0:
            #    print "cursor: " + str(self.cursor)
            return res
        else:
            raise StopIteration
def get_net_315():
    source = mx.sym.Variable("data")
    label = mx.sym.Variable("softmax_label")
    #print_inferred_shape(source)

    kernel_size = (3, 3, 3)
    stride=(1, 1,1)
    pad_size = (1, 1, 1)
    filter_count = 32
    net =  mx.sym.Convolution(data=source, kernel=kernel_size, stride=stride, pad=pad_size, num_filter=filter_count)
    net = mx.sym.BatchNorm(net)
    net = mx.sym.Activation(net, act_type="relu")
    #print_inferred_shape(net)
    
    
    net = mx.sym.Convolution(net, kernel=kernel_size, stride=stride, pad=pad_size, num_filter=filter_count)
    net = mx.sym.BatchNorm(net)
    net = mx.sym.Activation(net, act_type="relu")
    #print_inferred_shape(net)
    
    net = mx.sym.Convolution(net, kernel=kernel_size, stride=stride, pad=pad_size, num_filter=filter_count)
    net = mx.sym.BatchNorm(net)
    net = mx.sym.Activation(net, act_type="relu")
    #print_inferred_shape(net)
    
    net1=net
    
    net = mx.sym.Pooling(net, pool_type="max", kernel=(2, 2,2), stride=(2,2, 2))
    #print_inferred_shape(net)
    
    net = mx.sym.Convolution(net, kernel=kernel_size, stride=stride, pad=pad_size, num_filter=filter_count)
    net = mx.sym.BatchNorm(net)
    net = mx.sym.Activation(net, act_type="relu")

    #print_inferred_shape(net)

    
    net =  mx.sym.Convolution(net, kernel=kernel_size, stride=stride, pad=pad_size, num_filter=filter_count)
    net = mx.sym.BatchNorm(net)
    net = mx.sym.Activation(net, act_type="relu")
    #print_inferred_shape(net)
    
    net = mx.sym.Convolution(net, kernel=kernel_size, stride=stride, pad=pad_size, num_filter=filter_count)
    net = mx.sym.BatchNorm(net)
    net = mx.sym.Activation(net, act_type="relu")
    #print_inferred_shape(net)
    
    net = mx.sym.Dropout(net,p=0.4)
    
    net = mx.sym.Deconvolution(net, kernel=(2, 2,2), pad=(0, 0,0), stride=(2,2, 2), num_filter=filter_count)
    net = mx.sym.Activation(net, act_type="relu")


    
    net = mx.sym.Concat(*[net1, net])
    #print_inferred_shape(net)

    net = mx.sym.Convolution(net, kernel=kernel_size, stride=stride, pad=pad_size, num_filter=filter_count)
    net = mx.sym.BatchNorm(net)
    net = mx.sym.Activation(net, act_type="relu")
    #print_inferred_shape(net)
    
    net = mx.sym.Convolution(net, kernel=kernel_size, stride=stride, pad=pad_size, num_filter=filter_count)
    net = mx.sym.BatchNorm(net)
    net = mx.sym.Activation(net, act_type="relu")
    #print_inferred_shape(net)
        
    net = mx.sym.Convolution(net, kernel=kernel_size, stride=stride, pad=pad_size, num_filter=1)
    net = mx.sym.BatchNorm(net)
    net = mx.sym.Activation(net, act_type="sigmoid")
    #print_inferred_shape(net)
    
    y = mx.symbol.Flatten(net)
    
    loss= mx.sym.MakeLoss(dice_coef_loss(label, y))
    pred_loss = mx.sym.Group([mx.sym.BlockGrad(y), loss])

    return pred_loss
def get_net_unet():
    source = mx.sym.Variable("data")
    label = mx.sym.Variable("softmax_label")
    print_inferred_shape(source)

    kernel_size = (3, 3, 3)
    stride=(1, 1,1)
    pad_size = (1, 1, 1)
    filter_count = 32
    net =  mx.sym.Convolution(data=source, kernel=kernel_size, stride=stride, pad=pad_size, num_filter=filter_count)
    print_inferred_shape(net)
    #net = mx.sym.BatchNorm(net)
    net = mx.sym.Activation(net, act_type="relu")
    print_inferred_shape(net)
    
    
    net = mx.sym.Convolution(net, kernel=kernel_size, stride=stride, pad=pad_size, num_filter=filter_count*2)
    print_inferred_shape(net)
    #net = mx.sym.BatchNorm(net)
    net = mx.sym.Activation(net, act_type="relu")
    print_inferred_shape(net)
    
    net1=net
    
    net = mx.sym.Pooling(net, pool_type="max", kernel=(2, 2,2), stride=(2,2, 2))
    print_inferred_shape(net)
    
    net = mx.sym.Convolution(net, kernel=kernel_size, stride=stride, pad=pad_size, num_filter=filter_count*2)
    print_inferred_shape(net)
    #net = mx.sym.BatchNorm(net)
    net = mx.sym.Activation(net, act_type="relu")

    print_inferred_shape(net)

    
    net =  mx.sym.Convolution(net, kernel=kernel_size, stride=stride, pad=pad_size, num_filter=filter_count*4)
    print_inferred_shape(net)
    #net = mx.sym.BatchNorm(net)
    net = mx.sym.Activation(net, act_type="relu")
    print_inferred_shape(net)
    
    net2=net
    
    net = mx.sym.Pooling(net, pool_type="max", kernel=(2, 2,2), stride=(2,2, 2))
    print_inferred_shape(net)
    
    net = mx.sym.Convolution(net, kernel=kernel_size, stride=stride, pad=pad_size, num_filter=filter_count*4)
    print_inferred_shape(net)
    #net = mx.sym.BatchNorm(net)
    net = mx.sym.Activation(net, act_type="relu")
    print_inferred_shape(net)

    net = mx.sym.Convolution(net, kernel=kernel_size, stride=stride, pad=pad_size, num_filter=filter_count*8)
    print_inferred_shape(net)
    #net = mx.sym.BatchNorm(net)
    net = mx.sym.Activation(net, act_type="relu")
    #print "net3"
    print_inferred_shape(net)
    
    net3=net
    net = mx.sym.Pooling(net, pool_type="max", kernel=(2, 2,2), stride=(2,2, 2))
    print_inferred_shape(net)
    
    net = mx.sym.Convolution(net, kernel=kernel_size, stride=stride, pad=pad_size, num_filter=filter_count*8)
    print_inferred_shape(net)
    #net = mx.sym.BatchNorm(net)
    net = mx.sym.Activation(net, act_type="relu")
    print_inferred_shape(net)
    
    net = mx.sym.Convolution(net, kernel=kernel_size, stride=stride, pad=pad_size, num_filter=filter_count*16)
    print_inferred_shape(net)
    #net = mx.sym.BatchNorm(net)
    net = mx.sym.Activation(net, act_type="relu")
    print_inferred_shape(net)
    
    net = mx.sym.Deconvolution(net, kernel=(2, 2,2), pad=(0, 0,0), stride=(2,2, 2), num_filter=filter_count*16)
    print_inferred_shape(net)
    net = mx.sym.Activation(net, act_type="relu")

    print_inferred_shape(net)

    net = mx.sym.Concat(*[net3, net])

    print_inferred_shape(net)
    
    net = mx.sym.Convolution(net, kernel=kernel_size, stride=stride, pad=pad_size, num_filter=filter_count*8)
    print_inferred_shape(net)
    #net = mx.sym.BatchNorm(net)
    net = mx.sym.Activation(net, act_type="relu")
    print_inferred_shape(net)

    net = mx.sym.Convolution(net, kernel=kernel_size, stride=stride, pad=pad_size, num_filter=filter_count*8)
    print_inferred_shape(net)
    #net = mx.sym.BatchNorm(net)
    net = mx.sym.Activation(net, act_type="relu")
    print_inferred_shape(net)
    
    net = mx.sym.Deconvolution(net, kernel=(2, 2,2), pad=(0, 0,0), stride=(2,2, 2), num_filter=filter_count*8)
    print_inferred_shape(net)
    net = mx.sym.Activation(net, act_type="relu")
    print_inferred_shape(net)
    
    net = mx.sym.Concat(*[net2, net])
    print_inferred_shape(net)

    net = mx.sym.Convolution(net, kernel=kernel_size, stride=stride, pad=pad_size, num_filter=filter_count*4)
    print_inferred_shape(net)
    #net = mx.sym.BatchNorm(net)
    net = mx.sym.Activation(net, act_type="relu")
    print_inferred_shape(net)
    
    net = mx.sym.Convolution(net, kernel=kernel_size, stride=stride, pad=pad_size, num_filter=filter_count*4)
    print_inferred_shape(net)
    #net = mx.sym.BatchNorm(net)
    net = mx.sym.Activation(net, act_type="relu")
    print_inferred_shape(net)
    
    
    
    net = mx.sym.Deconvolution(net, kernel=(2, 2,2), pad=(0, 0,0), stride=(2,2, 2), num_filter=filter_count*4)
    print_inferred_shape(net)
    net = mx.sym.Activation(net, act_type="relu")
    print_inferred_shape(net)
    
    net = mx.sym.Concat(*[net1, net])
    print_inferred_shape(net)

    net = mx.sym.Convolution(net, kernel=kernel_size, stride=stride, pad=pad_size, num_filter=filter_count*2)
    print_inferred_shape(net)
    #net = mx.sym.BatchNorm(net)
    net = mx.sym.Activation(net, act_type="relu")
    print_inferred_shape(net)
    
    net = mx.sym.Convolution(net, kernel=kernel_size, stride=stride, pad=pad_size, num_filter=filter_count*2)
    print_inferred_shape(net)
    #net = mx.sym.BatchNorm(net)
    net = mx.sym.Activation(net, act_type="relu")
    print_inferred_shape(net)
        
    net = mx.sym.Convolution(net, kernel=kernel_size, stride=stride, pad=pad_size, num_filter=3)
    #net = mx.sym.BatchNorm(net)
    print_inferred_shape(net)
    net = mx.sym.Activation(net, act_type="sigmoid")
    print_inferred_shape(net)
    
    y = mx.symbol.Flatten(net)
    
    print_inferred_shape(y)
    
    loss= mx.sym.MakeLoss(dice_coef_loss(label, y))
    pred_loss = mx.sym.Group([mx.sym.BlockGrad(y), loss])

    return pred_loss

class FileIter3(DataIter):
    def __init__(self, path,
                 data_name="data",
                 label_name="softmax_label",
                 batch_size=32,
                 do_augment=True,
                 mean_image=None):

        
        self.epoch = 0

        
        super(FileIter3, self).__init__()
        self.batch_size = batch_size
        self.do_augment=do_augment


        #self.mean = cv2.imread(mean_image, cv2.IMREAD_GRAYSCALE)

        self.data_name = data_name
        self.label_name = label_name

        self.record = mx.recordio.MXRecordIO(path, 'r')

        
        def readrecord(record):
            record.reset()
            num_data=0
            while True:
                item = record.read()
                num_data+=1
                if not item:
                    break
            return num_data-1
        
        
        self.num_data = readrecord(self.record)#len(open(self.flist_name, 'r').readlines())
        
        self.cursor = -1
        self.record.reset()

        self.data, self.label = self._read()
        self.reset()

    def _read(self):
        """get two list, each list contains two elements: name and nd.array value"""
                
        data = {}
        label = {}

        dd = []
        ll = []
        for i in range(0, self.batch_size):
            
            item = self.record.read()            
            header, l = mx.recordio.unpack_img(item)
            
            d=header.label
            
            d=d.reshape((52,52,52))
            d=d.astype(float)

            l=l.reshape((52,52,52))
            l=l.astype(float)
            
            if self.do_augment==True:            
                d,l=augment(d,l)
            else:

                d=crop_centerz(d)
                l=crop_centerz(l)
                d=list(d)
                l=list(l)
                d=np.array([crop_center2(i,32,32) for i in d])
                l=np.array([crop_center2(i,32,32) for i in l])

            
            d = np.repeat(d[:, :, np.newaxis], 3, axis=2)
            d=d.swapaxes(0,2)
            d=d.swapaxes(1,2)
            d = np.expand_dims(d, axis=0)

            l = np.repeat(l[:, :, np.newaxis], 3, axis=2)
            l=l.swapaxes(0,2)
            l=l.swapaxes(1,2)
            l=l.reshape((3*32*32*32))

            l=l.astype(float)
            l = np.expand_dims(l, axis=0)
            dd.append(d)
            ll.append(l)

        d = np.vstack(dd)
        l = np.vstack(ll)
        data[self.data_name] = d
        label[self.label_name] = l
        res = list(data.items()), list(label.items())
        return res

    @property
    def provide_data(self):
        """The name and shape of data provided by this iterator"""
        res = [(k, tuple(list(v.shape[0:]))) for k, v in self.data]
        # print "data : " + str(res)
        return res

    @property
    def provide_label(self):
        """The name and shape of label provided by this iterator"""
        res = [(k, tuple(list(v.shape[0:]))) for k, v in self.label]
        #print "label : " + str(res)
        return res
    

    def reset(self):
        self.cursor = -1
        self.record.reset()
        self.epoch += 1

    def getpad(self):
        return 0

    def iter_next(self):
        self.cursor += self.batch_size
        if self.cursor < self.num_data:
            return True
        else:
            return False

    def eof(self):
        res = self.cursor >= self.num_data
        return res

    def next(self):
        """return one dict which contains "data" and "label" """
        if self.iter_next():
            self.data, self.label = self._read()
            #for i in range(0, 10):
            #    self.data, self.label = self._read()
            #    d.append(mx.nd.array(self.data[0][1]))
            #    l.append(mx.nd.array(self.label[0][1]))
            
            res = DataBatch(data=[mx.nd.array(self.data[0][1])], label=[mx.nd.array(self.label[0][1])], pad=self.getpad(), index=None)
            #if self.cursor % 100 == 0:
            #    print "cursor: " + str(self.cursor)
            return res
        else:
            raise StopIteration


class XYRange:
    def __init__(self, x_min, x_max, y_min, y_max, chance=1.0):
        self.chance = chance
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max
        self.last_x = 0
        self.last_y = 0

    def get_last_xy_txt(self):
        res = "x_" + str(int(self.last_x * 100)).replace("-", "m") + "-" + "y_" + str(int(self.last_y * 100)).replace("-", "m")
        return res


def random_translate_img(img, xy_range, border_mode="constant"):
    if random.random() > xy_range.chance:
        return img

    org_width, org_height = img.shape[-2:]
    translate_x = random.randint(xy_range.x_min, xy_range.x_max)
    translate_y = random.randint(xy_range.y_min, xy_range.y_max)
    
    trans_matrix = np.float32([[1, 0, translate_x], [0, 1, translate_y]])
    border_const = cv2.BORDER_CONSTANT
    if border_mode == "reflect":
        border_const = cv2.BORDER_REFLECT

    res = []
    for img_inst in img:
        img_inst = cv2.warpAffine(img_inst, trans_matrix, (org_width, org_height), borderMode=border_const)
        res.append(img_inst)
    if len(res) == 1:
        res = res[0]
    xy_range.last_x = translate_x
    xy_range.last_y = translate_y
    return res


ELASTIC_INDICES = None  # needed to make it faster to fix elastic deformation per epoch.
def elastic_transform(image, alpha, sigma, random_state=None):
    global ELASTIC_INDICES
    shape = image.shape

    if ELASTIC_INDICES == None:
        if random_state is None:
            random_state = np.random.RandomState(1301)

        dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
        dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
        x, y = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]))
        ELASTIC_INDICES = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1))
    return map_coordinates(image, ELASTIC_INDICES, order=1).reshape(shape)


def crop_center(img,cropx,cropy):
    y,x = img.shape
    startx = x//2-(cropx//2)
    starty = y//2-(cropy//2)    
    return img[starty:starty+cropy,startx:startx+cropx]

def crop_centerz(img,cropz=32):
    z=img.shape[0]
    startz= z//2-(cropz//2)    
    return img[startz:startz+cropz]

def random_flip_img(img, horizontal_chance=0, vertical_chance=0):
    import cv2
    flip_horizontal = False
    if random.random() < horizontal_chance:
        flip_horizontal = True

    flip_vertical = False
    if random.random() < vertical_chance:
        flip_vertical = True

    if not flip_horizontal and not flip_vertical:
        return img

    flip_val = 1
    if flip_vertical:
        flip_val = -1 if flip_horizontal else 0

    if not isinstance(img, list):
        res = cv2.flip(img, flip_val) # 0 = X axis, 1 = Y axis,  -1 = both
    else:
        res = []
        for img_item in img:
            img_flip = cv2.flip(img_item, flip_val)
            res.append(img_flip)
            
    #print flip_val
    
    return res




class FileIter7(DataIter):
    def __init__(self, path,
                 data_name="data",
                 label_name="softmax_label",
                 batch_size=32,
                 do_augment=False,
                 mean_image=None):

        
        self.epoch = 0

        
        super(FileIter7, self).__init__()
        self.batch_size = batch_size
        self.do_augment=do_augment


        #self.mean = cv2.imread(mean_image, cv2.IMREAD_GRAYSCALE)

        self.data_name = data_name
        self.label_name = label_name

        self.record = mx.recordio.MXRecordIO(path, 'r')

        
        def readrecord(record):
            record.reset()
            num_data=0
            while True:
                item = record.read()
                num_data+=1
                if not item:
                    break
            return num_data-1
        
        
        self.num_data = readrecord(self.record)#len(open(self.flist_name, 'r').readlines())
        
        self.cursor = -1
        self.record.reset()

        self.data, self.label = self._read()
        self.reset()

    def _read(self):
        """get two list, each list contains two elements: name and nd.array value"""
                
        data = {}
        label = {}

        dd = []
        ll = []
        for i in range(0, self.batch_size):
            
            item = self.record.read()            
            header, l = mx.recordio.unpack_img(item)
            
            d=header.label
            try:
                d=d.reshape((52,52,52))
                d=d.astype(float)

                l=l.reshape((52,52,52))
                l=l.astype(float)

                if self.do_augment==True:            
                    d,l=augment(d,l)
                else:

                    d=crop_centerz(d)
                    l=crop_centerz(l)
                    d=list(d)
                    l=list(l)
                    d=np.array([crop_center2(i,32,32) for i in d])
                    l=np.array([crop_center2(i,32,32) for i in l])
            except:
                d=d.reshape((32,32,32))
                d=d.astype(float)

                l=l.reshape((32,32,32))
                l=l.astype(float)

            d = np.repeat(d[:, :, np.newaxis], 3, axis=2)
            d=d.swapaxes(0,2)
            d=d.swapaxes(1,2)
            d = np.expand_dims(d, axis=0)

            l = np.repeat(l[:, :, np.newaxis], 3, axis=2)
            l=l.swapaxes(0,2)
            l=l.swapaxes(1,2)
            l=l.reshape((3*32*32*32))

            l=l.astype(float)
            l = np.expand_dims(l, axis=0)

            dd.append(d)
            ll.append(l)

        d = np.vstack(dd)
        l = np.vstack(ll)
        data[self.data_name] = d
        label[self.label_name] = l
        res = list(data.items()), list(label.items())
        return res

    @property
    def provide_data(self):
        """The name and shape of data provided by this iterator"""
        res = [(k, tuple(list(v.shape[0:]))) for k, v in self.data]
        # print "data : " + str(res)
        return res

    @property
    def provide_label(self):
        """The name and shape of label provided by this iterator"""
        res = [(k, tuple(list(v.shape[0:]))) for k, v in self.label]
        #print "label : " + str(res)
        return res
    

    def reset(self):
        self.cursor = -1
        self.record.reset()
        self.epoch += 1

    def getpad(self):
        return 0

    def iter_next(self):
        self.cursor += self.batch_size
        if self.cursor < self.num_data:
            return True
        else:
            return False

    def eof(self):
        res = self.cursor >= self.num_data
        return res

    def next(self):
        """return one dict which contains "data" and "label" """
        if self.iter_next():
            self.data, self.label = self._read()
            #for i in range(0, 10):
            #    self.data, self.label = self._read()
            #    d.append(mx.nd.array(self.data[0][1]))
            #    l.append(mx.nd.array(self.label[0][1]))
            
            res = DataBatch(data=[mx.nd.array(self.data[0][1])], label=[mx.nd.array(self.label[0][1])], pad=self.getpad(), index=None)
            #if self.cursor % 100 == 0:
            #    print "cursor: " + str(self.cursor)
            return res
        else:
            raise StopIteration

def get_net_unet0():
    source = mx.sym.Variable("data")
    label = mx.sym.Variable("softmax_label")
    print_inferred_shape(source)

    kernel_size = (3, 3, 3)
    stride=(1, 1,1)
    pad_size = (1, 1, 1)
    filter_count = 32
    net =  mx.sym.Convolution(data=source, kernel=kernel_size, stride=stride, pad=pad_size, num_filter=filter_count)
    print_inferred_shape(net)
    #net = mx.sym.BatchNorm(net)
    net = mx.sym.Activation(net, act_type="relu")
    print_inferred_shape(net)
    
    
    net = mx.sym.Convolution(net, kernel=kernel_size, stride=stride, pad=pad_size, num_filter=filter_count*2)
    print_inferred_shape(net)
    #net = mx.sym.BatchNorm(net)
    net = mx.sym.Activation(net, act_type="relu")
    print_inferred_shape(net)
    
    net1=net
    
    net = mx.sym.Pooling(net, pool_type="max", kernel=(2, 2,2), stride=(2,2, 2))
    print_inferred_shape(net)
    
    net = mx.sym.Convolution(net, kernel=kernel_size, stride=stride, pad=pad_size, num_filter=filter_count*2)
    print_inferred_shape(net)
    #net = mx.sym.BatchNorm(net)
    net = mx.sym.Activation(net, act_type="relu")

    print_inferred_shape(net)

    
    net =  mx.sym.Convolution(net, kernel=kernel_size, stride=stride, pad=pad_size, num_filter=filter_count*4)
    print_inferred_shape(net)
    #net = mx.sym.BatchNorm(net)
    net = mx.sym.Activation(net, act_type="relu")
    print_inferred_shape(net)
    
    net2=net
    
    net = mx.sym.Pooling(net, pool_type="max", kernel=(2, 2,2), stride=(2,2, 2))
    print_inferred_shape(net)
    
    net = mx.sym.Convolution(net, kernel=kernel_size, stride=stride, pad=pad_size, num_filter=filter_count*4)
    print_inferred_shape(net)
    #net = mx.sym.BatchNorm(net)
    net = mx.sym.Activation(net, act_type="relu")
    print_inferred_shape(net)

    net = mx.sym.Convolution(net, kernel=kernel_size, stride=stride, pad=pad_size, num_filter=filter_count*8)
    print_inferred_shape(net)
    #net = mx.sym.BatchNorm(net)
    net = mx.sym.Activation(net, act_type="relu")
    #print "net3"
    print_inferred_shape(net)
    
    net3=net
    net = mx.sym.Pooling(net, pool_type="max", kernel=(2, 2,2), stride=(2,2, 2))
    print_inferred_shape(net)
    
    net = mx.sym.Convolution(net, kernel=kernel_size, stride=stride, pad=pad_size, num_filter=filter_count*8)
    print_inferred_shape(net)
    #net = mx.sym.BatchNorm(net)
    net = mx.sym.Activation(net, act_type="relu")
    print_inferred_shape(net)
    
    net = mx.sym.Convolution(net, kernel=kernel_size, stride=stride, pad=pad_size, num_filter=filter_count*16)
    print_inferred_shape(net)
    #net = mx.sym.BatchNorm(net)
    net = mx.sym.Activation(net, act_type="relu")
    print_inferred_shape(net)
    
    net = mx.sym.Deconvolution(net, kernel=(2, 2,2), pad=(0, 0,0), stride=(2,2, 2), num_filter=filter_count*16)
    print_inferred_shape(net)
    net = mx.sym.Activation(net, act_type="relu")

    print_inferred_shape(net)

    net = mx.sym.Concat(*[net3, net])

    print_inferred_shape(net)
    
    net = mx.sym.Convolution(net, kernel=kernel_size, stride=stride, pad=pad_size, num_filter=filter_count*8)
    print_inferred_shape(net)
    #net = mx.sym.BatchNorm(net)
    net = mx.sym.Activation(net, act_type="relu")
    print_inferred_shape(net)

    net = mx.sym.Convolution(net, kernel=kernel_size, stride=stride, pad=pad_size, num_filter=filter_count*8)
    print_inferred_shape(net)
    #net = mx.sym.BatchNorm(net)
    net = mx.sym.Activation(net, act_type="relu")
    print_inferred_shape(net)
    
    net = mx.sym.Deconvolution(net, kernel=(2, 2,2), pad=(0, 0,0), stride=(2,2, 2), num_filter=filter_count*8)
    print_inferred_shape(net)
    net = mx.sym.Activation(net, act_type="relu")
    print_inferred_shape(net)
    
    net = mx.sym.Concat(*[net2, net])
    print_inferred_shape(net)

    net = mx.sym.Convolution(net, kernel=kernel_size, stride=stride, pad=pad_size, num_filter=filter_count*4)
    print_inferred_shape(net)
    #net = mx.sym.BatchNorm(net)
    net = mx.sym.Activation(net, act_type="relu")
    print_inferred_shape(net)
    
    net = mx.sym.Convolution(net, kernel=kernel_size, stride=stride, pad=pad_size, num_filter=filter_count*4)
    print_inferred_shape(net)
    #net = mx.sym.BatchNorm(net)
    net = mx.sym.Activation(net, act_type="relu")
    print_inferred_shape(net)
    
    
    
    net = mx.sym.Deconvolution(net, kernel=(2, 2,2), pad=(0, 0,0), stride=(2,2, 2), num_filter=filter_count*4)
    print_inferred_shape(net)
    net = mx.sym.Activation(net, act_type="relu")
    print_inferred_shape(net)
    
    net = mx.sym.Concat(*[net1, net])
    print_inferred_shape(net)

    net = mx.sym.Convolution(net, kernel=kernel_size, stride=stride, pad=pad_size, num_filter=filter_count*2)
    print_inferred_shape(net)
    #net = mx.sym.BatchNorm(net)
    net = mx.sym.Activation(net, act_type="relu")
    print_inferred_shape(net)
    
    net = mx.sym.Convolution(net, kernel=kernel_size, stride=stride, pad=pad_size, num_filter=filter_count*2)
    print_inferred_shape(net)
    #net = mx.sym.BatchNorm(net)
    net = mx.sym.Activation(net, act_type="relu")
    print_inferred_shape(net)
        
    net = mx.sym.Convolution(net, kernel=kernel_size, stride=stride, pad=pad_size, num_filter=3)
    #net = mx.sym.BatchNorm(net)
    print_inferred_shape(net)
    net = mx.sym.Activation(net, act_type="sigmoid")
    #print_inferred_shape(net)
    
    net = mx.symbol.Flatten(net)
    
    print_inferred_shape(net)

    y = net
    
    #y=mx.symbol.LogisticRegressionOutput(data=net, name='softmax0')

    loss= mx.sym.MakeLoss(dice_coef_loss(label, y))

    pred_loss = mx.sym.Group([mx.sym.BlockGrad(y), loss])

    return pred_loss


def unet0():
    source = mx.sym.Variable("data")
    print_inferred_shape(source)

    kernel_size = (3, 3, 3)
    stride=(1, 1,1)
    pad_size = (1, 1, 1)
    filter_count = 32
    net =  mx.sym.Convolution(data=source, kernel=kernel_size, stride=stride, pad=pad_size, num_filter=filter_count)
    print_inferred_shape(net)
    #net = mx.sym.BatchNorm(net)
    net = mx.sym.Activation(net, act_type="relu")
    print_inferred_shape(net)
    
    
    net = mx.sym.Convolution(net, kernel=kernel_size, stride=stride, pad=pad_size, num_filter=filter_count*2)
    print_inferred_shape(net)
    #net = mx.sym.BatchNorm(net)
    net = mx.sym.Activation(net, act_type="relu")
    print_inferred_shape(net)
    
    net1=net
    
    net = mx.sym.Pooling(net, pool_type="max", kernel=(2, 2,2), stride=(2,2, 2))
    print_inferred_shape(net)
    
    net = mx.sym.Convolution(net, kernel=kernel_size, stride=stride, pad=pad_size, num_filter=filter_count*2)
    print_inferred_shape(net)
    #net = mx.sym.BatchNorm(net)
    net = mx.sym.Activation(net, act_type="relu")

    print_inferred_shape(net)

    
    net =  mx.sym.Convolution(net, kernel=kernel_size, stride=stride, pad=pad_size, num_filter=filter_count*4)
    print_inferred_shape(net)
    #net = mx.sym.BatchNorm(net)
    net = mx.sym.Activation(net, act_type="relu")
    print_inferred_shape(net)
    
    net2=net
    
    net = mx.sym.Pooling(net, pool_type="max", kernel=(2, 2,2), stride=(2,2, 2))
    print_inferred_shape(net)
    
    net = mx.sym.Convolution(net, kernel=kernel_size, stride=stride, pad=pad_size, num_filter=filter_count*4)
    print_inferred_shape(net)
    #net = mx.sym.BatchNorm(net)
    net = mx.sym.Activation(net, act_type="relu")
    print_inferred_shape(net)

    net = mx.sym.Convolution(net, kernel=kernel_size, stride=stride, pad=pad_size, num_filter=filter_count*8)
    print_inferred_shape(net)
    #net = mx.sym.BatchNorm(net)
    net = mx.sym.Activation(net, act_type="relu")
    #print "net3"
    print_inferred_shape(net)
    
    net3=net
    net = mx.sym.Pooling(net, pool_type="max", kernel=(2, 2,2), stride=(2,2, 2))
    print_inferred_shape(net)
    
    net = mx.sym.Convolution(net, kernel=kernel_size, stride=stride, pad=pad_size, num_filter=filter_count*8)
    print_inferred_shape(net)
    #net = mx.sym.BatchNorm(net)
    net = mx.sym.Activation(net, act_type="relu")
    print_inferred_shape(net)
    
    net = mx.sym.Convolution(net, kernel=kernel_size, stride=stride, pad=pad_size, num_filter=filter_count*16)
    print_inferred_shape(net)
    #net = mx.sym.BatchNorm(net)
    net = mx.sym.Activation(net, act_type="relu")
    print_inferred_shape(net)
    
    net = mx.sym.Deconvolution(net, kernel=(2, 2,2), pad=(0, 0,0), stride=(2,2, 2), num_filter=filter_count*16)
    print_inferred_shape(net)
    net = mx.sym.Activation(net, act_type="relu")

    print_inferred_shape(net)

    net = mx.sym.Concat(*[net3, net])

    print_inferred_shape(net)
    
    net = mx.sym.Convolution(net, kernel=kernel_size, stride=stride, pad=pad_size, num_filter=filter_count*8)
    print_inferred_shape(net)
    #net = mx.sym.BatchNorm(net)
    net = mx.sym.Activation(net, act_type="relu")
    print_inferred_shape(net)

    net = mx.sym.Convolution(net, kernel=kernel_size, stride=stride, pad=pad_size, num_filter=filter_count*8)
    print_inferred_shape(net)
    #net = mx.sym.BatchNorm(net)
    net = mx.sym.Activation(net, act_type="relu")
    print_inferred_shape(net)
    
    net = mx.sym.Deconvolution(net, kernel=(2, 2,2), pad=(0, 0,0), stride=(2,2, 2), num_filter=filter_count*8)
    print_inferred_shape(net)
    net = mx.sym.Activation(net, act_type="relu")
    print_inferred_shape(net)
    
    net = mx.sym.Concat(*[net2, net])
    print_inferred_shape(net)

    net = mx.sym.Convolution(net, kernel=kernel_size, stride=stride, pad=pad_size, num_filter=filter_count*4)
    print_inferred_shape(net)
    #net = mx.sym.BatchNorm(net)
    net = mx.sym.Activation(net, act_type="relu")
    print_inferred_shape(net)
    
    net = mx.sym.Convolution(net, kernel=kernel_size, stride=stride, pad=pad_size, num_filter=filter_count*4)
    print_inferred_shape(net)
    #net = mx.sym.BatchNorm(net)
    net = mx.sym.Activation(net, act_type="relu")
    print_inferred_shape(net)
    
    
    
    net = mx.sym.Deconvolution(net, kernel=(2, 2,2), pad=(0, 0,0), stride=(2,2, 2), num_filter=filter_count*4)
    print_inferred_shape(net)
    net = mx.sym.Activation(net, act_type="relu")
    print_inferred_shape(net)
    
    net = mx.sym.Concat(*[net1, net])
    print_inferred_shape(net)

    net = mx.sym.Convolution(net, kernel=kernel_size, stride=stride, pad=pad_size, num_filter=filter_count*2)
    print_inferred_shape(net)
    #net = mx.sym.BatchNorm(net)
    net = mx.sym.Activation(net, act_type="relu")
    print_inferred_shape(net)
    
    net = mx.sym.Convolution(net, kernel=kernel_size, stride=stride, pad=pad_size, num_filter=filter_count*2)
    print_inferred_shape(net)
    #net = mx.sym.BatchNorm(net)
    net = mx.sym.Activation(net, act_type="relu")
    print_inferred_shape(net)
        
    net = mx.sym.Convolution(net, kernel=kernel_size, stride=stride, pad=pad_size, num_filter=3)
    #net = mx.sym.BatchNorm(net)
    print_inferred_shape(net)
    net = mx.sym.Activation(net, act_type="sigmoid")
    #print_inferred_shape(net)
    
    net = mx.symbol.Flatten(net)
    
    print_inferred_shape(net)
    
    y=mx.symbol.LogisticRegressionOutput(data=net, name='softmax')


    return y


def get_net_317():
    source = mx.sym.Variable("data")
    label = mx.sym.Variable("softmax_label")
    #print_inferred_shape(source)

    kernel_size = (3, 3, 3)
    stride=(1, 1,1)
    pad_size = (1, 1, 1)
    filter_count = 32
    net =  mx.sym.Convolution(data=source, kernel=kernel_size, stride=stride, pad=pad_size, num_filter=filter_count)
    net = mx.sym.BatchNorm(net)
    net = mx.sym.Activation(net, act_type="relu")
    #print_inferred_shape(net)
    
    
    net = mx.sym.Convolution(net, kernel=kernel_size, stride=stride, pad=pad_size, num_filter=filter_count)
    net = mx.sym.BatchNorm(net)
    net = mx.sym.Activation(net, act_type="relu")
    #print_inferred_shape(net)
    
    net = mx.sym.Convolution(net, kernel=kernel_size, stride=stride, pad=pad_size, num_filter=filter_count)
    net = mx.sym.BatchNorm(net)
    net = mx.sym.Activation(net, act_type="relu")
    #print_inferred_shape(net)
    
    net1=net
    
    net = mx.sym.Pooling(net, pool_type="max", kernel=(2, 2,2), stride=(2,2, 2))
    #print_inferred_shape(net)
    
    net = mx.sym.Convolution(net, kernel=kernel_size, stride=stride, pad=pad_size, num_filter=filter_count)
    net = mx.sym.BatchNorm(net)
    net = mx.sym.Activation(net, act_type="relu")

    #print_inferred_shape(net)

    
    net =  mx.sym.Convolution(net, kernel=kernel_size, stride=stride, pad=pad_size, num_filter=filter_count)
    net = mx.sym.BatchNorm(net)
    net = mx.sym.Activation(net, act_type="relu")
    #print_inferred_shape(net)
    
    net = mx.sym.Convolution(net, kernel=kernel_size, stride=stride, pad=pad_size, num_filter=filter_count)
    net = mx.sym.BatchNorm(net)
    net = mx.sym.Activation(net, act_type="relu")
    #print_inferred_shape(net)
    
    net = mx.sym.Dropout(net,p=0.4)
    
    net = mx.sym.Deconvolution(net, kernel=(2, 2,2), pad=(0, 0,0), stride=(2,2, 2), num_filter=filter_count)
    net = mx.sym.Activation(net, act_type="relu")


    
    net = mx.sym.Concat(*[net1, net])
    #print_inferred_shape(net)

    net = mx.sym.Convolution(net, kernel=kernel_size, stride=stride, pad=pad_size, num_filter=filter_count)
    net = mx.sym.BatchNorm(net)
    net = mx.sym.Activation(net, act_type="relu")
    #print_inferred_shape(net)
    
    net = mx.sym.Convolution(net, kernel=kernel_size, stride=stride, pad=pad_size, num_filter=filter_count)
    net = mx.sym.BatchNorm(net)
    net = mx.sym.Activation(net, act_type="relu")
    #print_inferred_shape(net)
        
    net = mx.sym.Convolution(net, kernel=kernel_size, stride=stride, pad=pad_size, num_filter=1)
    net = mx.sym.BatchNorm(net,fix_gamma=False)
    net = mx.sym.Activation(net, act_type="sigmoid")
    #print_inferred_shape(net)
    
    y = mx.symbol.Flatten(net)
    
    loss= mx.sym.MakeLoss(dice_coef_loss(label, y))
    pred_loss = mx.sym.Group([mx.sym.BlockGrad(y), loss])

    return pred_loss

class FileIter(DataIter):
    def __init__(self, path,path_idx,
                 data_name="data",
                 label_name="softmax_label",
                 batch_size=1,
                 do_augment=False,
                 mean_image=.2815,
                 std_image = .2807,
                 do_shuffle = True):

        
        random.seed(313)
        self.ind2=None
        self.do_shuffle = do_shuffle
        self.epoch = 0
        self.mean_image = mean_image
        self.std_image = std_image
        
        super(FileIter, self).__init__()
        self.batch_size = batch_size
        self.do_augment=do_augment
        

        self.data_name = data_name
        self.label_name = label_name

        self.record = mx.recordio.MXIndexedRecordIO(path_idx, path, 'r')#mx.recordio.MXRecordIO(path, 'r')

        
        def readrecord(record):
            record.reset()
            num_data=0
            while True:
                item = record.read()
                num_data+=1
                if not item:
                    break
            return num_data-1
        
        
        self.num_data = readrecord(self.record)#len(open(self.flist_name, 'r').readlines())
        
        
        
        self.idx = self.shuffle_idx()
        self.cursor = -1
        self.cursor2 = -1
        self.ind = self.idx[0]
        self.record.reset()

        self.data, self.label = self._read()
        self.reset()
            
    def shuffle_idx(self):
        num_data = self.num_data//self.batch_size*self.batch_size
        idx = [i for i in range(num_data)]
        if self.do_shuffle:
            random.shuffle(idx)
        idx = np.array(idx)
        idx = idx.reshape(-1,self.batch_size)
        return idx
    
    def _read(self):
        """get two list, each list contains two elements: name and nd.array value"""
                
        data = {}
        label = {}

        dd = []
        ll = []
        
        if self.ind2 is None:
            ind = self.ind
        else:
            ind = self.ind2
            self.ind2=None
            
        for i in range(0, self.batch_size):
            
            item = self.record.read_idx(ind[i])            
            header, l = mx.recordio.unpack_img(item)
            
            d=header.label

            d=d.reshape((32,32,32))- self.mean_image
            d = d/self.std_image
            d = np.expand_dims(d, axis=0) 
            d = np.expand_dims(d, axis=0)
            

            l=l.reshape((32*32*32))
            l = np.expand_dims(l, axis=0)
            l=l.astype(float)

            dd.append(d)
            ll.append(l)

        d = np.vstack(dd)
        l = np.vstack(ll)
        data[self.data_name] = d
        label[self.label_name] = l
        res = list(data.items()), list(label.items())
        return res

    @property
    def provide_data(self):
        """The name and shape of data provided by this iterator"""
        res = [(k, tuple(list(v.shape[0:]))) for k, v in self.data]
        # print "data : " + str(res)
        return res

    @property
    def provide_label(self):
        """The name and shape of label provided by this iterator"""
        res = [(k, tuple(list(v.shape[0:]))) for k, v in self.label]
        return res
    

    def reset(self):
        self.cursor = -1
        self.cursor2 = -1
        self.record.reset()
        self.epoch += 1
        self.idx = self.shuffle_idx()
        


    def getpad(self):
        return 0

    def iter_next(self):
        self.cursor += self.batch_size
        self.cursor2 += 1
        num_data = self.num_data//self.batch_size*self.batch_size
            

        if self.cursor < self.num_data:
            self.ind = self.idx[self.cursor2]
            return True
        else:
            return False

    def eof(self):
        res = self.cursor >= self.num_data
        return res

    def next(self):
        """return one dict which contains "data" and "label" """
        if self.iter_next():
            self.data, self.label = self._read()
 
            res = DataBatch(data=[mx.nd.array(self.data[0][1])], label=[mx.nd.array(self.label[0][1])], pad=self.getpad(), index=None)

            return res
        else:
            raise StopIteration

class LRScheduler(object):
    """Base class of a learning rate scheduler.

    A scheduler returns a new learning rate based on the number of updates that have
    been performed.

    Parameters
    ----------
    base_lr : float, optional
        The initial learning rate.
    """
    def __init__(self, base_lr=0.01):
        self.base_lr = base_lr

    def __call__(self, num_update):
        """Return a new learning rate.

        The ``num_update`` is the upper bound of the number of updates applied to
        every weight.

        Assume the optimizer has updated *i*-th weight by *k_i* times, namely
        ``optimizer.update(i, weight_i)`` is called by *k_i* times. Then::

            num_update = max([k_i for all i])

        Parameters
        ----------
        num_update: int
            the maximal number of updates applied to a weight.
        """
        raise NotImplementedError("must override this")
class lr_find(LRScheduler):
    """Reduce the learning rate by a factor for every *n* steps.

    It returns a new learning rate by::

        base_lr * pow(factor, floor(num_update/step))

    Parameters
    ----------
    step : int
        Changes the learning rate for every n updates.
    factor : float, optional
        The factor to change the learning rate.
    stop_factor_lr : float, optional
        Stop updating the learning rate if it is less than this value.
        
    """

    def __init__(self, layer_opt_lr, nb, end_lr=10, linear=True):
        super(lr_find,self).__init__()

        self.linear = linear
        ratio = end_lr/layer_opt_lr
        self.lr_mult = (ratio/nb) if linear else ratio**(1/nb)
        self.iteration = 1
        self.losses=[]
        self.lrs=[]
        self.init_lrs=1e-5
        self.new_lr = self.init_lrs

    def on_train_begin(self):
        self.best=1e9
        
        
    def __call__(self,b):
        return self.new_lr



    def on_batch_end(self, loss):
        self.losses.append(loss)
        mult = self.lr_mult*self.iteration if self.linear else self.lr_mult**self.iteration
        self.iteration +=1
        self.new_lr = self.init_lrs * mult
        self.lrs.append(self.new_lr)
        return self.init_lrs * mult
        if math.isnan(loss) or loss>self.best*4:
            return True
        if (loss<self.best and self.iteration>10): self.best=loss

    def plot(self, n_skip=10):
        plt.ylabel("loss")
        plt.xlabel("learning rate (log scale)")
        plt.plot(self.lrs[n_skip:-5], self.losses[n_skip:-5])
        plt.xscale('log')
        
    def reset(self):
        self.iteration = 1
        self.losses=[]
        self.lrs=[]



def get_net_319():
    source = mx.sym.Variable("data")
    label = mx.sym.Variable("softmax_label")
    #print_inferred_shape(source)

    kernel_size = (3, 3, 3)
    stride=(1, 1,1)
    pad_size = (1, 1, 1)
    filter_count = 32
    net =  mx.sym.Convolution(data=source, kernel=kernel_size, stride=stride, pad=pad_size, num_filter=filter_count)
    net = mx.sym.BatchNorm(net)
    net = mx.sym.Activation(net, act_type="relu")
    #print_inferred_shape(net)
    
    
    net = mx.sym.Convolution(net, kernel=kernel_size, stride=stride, pad=pad_size, num_filter=filter_count)
    net = mx.sym.BatchNorm(net)
    net = mx.sym.Activation(net, act_type="relu")
    #print_inferred_shape(net)
    
    net = mx.sym.Convolution(net, kernel=kernel_size, stride=stride, pad=pad_size, num_filter=filter_count)
    net = mx.sym.BatchNorm(net)
    net = mx.sym.Activation(net, act_type="relu")
    #print_inferred_shape(net)
    
    net1=net
    
    net = mx.sym.Pooling(net, pool_type="max", kernel=(2, 2,2), stride=(2,2, 2))
    #print_inferred_shape(net)
    
    net = mx.sym.Convolution(net, kernel=kernel_size, stride=stride, pad=pad_size, num_filter=filter_count)
    net = mx.sym.BatchNorm(net)
    net = mx.sym.Activation(net, act_type="relu")

    #print_inferred_shape(net)

    
    net =  mx.sym.Convolution(net, kernel=kernel_size, stride=stride, pad=pad_size, num_filter=filter_count)
    net = mx.sym.BatchNorm(net)
    net = mx.sym.Activation(net, act_type="relu")
    #print_inferred_shape(net)
    
    net = mx.sym.Convolution(net, kernel=kernel_size, stride=stride, pad=pad_size, num_filter=filter_count)
    net = mx.sym.BatchNorm(net)
    net = mx.sym.Activation(net, act_type="relu")
    #print_inferred_shape(net)
    
    net = mx.sym.Dropout(net,p=0.6)
    
    net = mx.sym.Deconvolution(net, kernel=(2, 2,2), pad=(0, 0,0), stride=(2,2, 2), num_filter=filter_count)
    net = mx.sym.Activation(net, act_type="relu")


    
    net = mx.sym.Concat(*[net1, net])
    #print_inferred_shape(net)

    net = mx.sym.Convolution(net, kernel=kernel_size, stride=stride, pad=pad_size, num_filter=filter_count)
    net = mx.sym.BatchNorm(net)
    net = mx.sym.Activation(net, act_type="relu")
    #print_inferred_shape(net)
    
    net = mx.sym.Convolution(net, kernel=kernel_size, stride=stride, pad=pad_size, num_filter=filter_count)
    net = mx.sym.BatchNorm(net)
    net = mx.sym.Activation(net, act_type="relu")
    #print_inferred_shape(net)
        
    net = mx.sym.Convolution(net, kernel=kernel_size, stride=stride, pad=pad_size, num_filter=1)
    net = mx.sym.BatchNorm(net,fix_gamma=False)
    net = mx.sym.Activation(net, act_type="sigmoid")
    #print_inferred_shape(net)
    
    y = mx.symbol.Flatten(net)
    
    loss= mx.sym.MakeLoss(dice_coef_loss(label, y))
    pred_loss = mx.sym.Group([mx.sym.BlockGrad(y), loss])

    return pred_loss

