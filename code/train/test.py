import os
import pickle
import time
from os.path import join, dirname
import math
import cv2
import lmdb
import numpy as np
import torch
import torch.nn as nn

from data.utils import normalize, normalize_reverse
from model import Model
#from .metrics import psnr_calculate, ssim_calculate
from .utils import AverageMeter
from data import Crop, Flip, ToTensor, normalize

def alphablend(color, tmpColor, tempStrength):
    return color * (1-tempStrength) + tmpColor * tempStrength

def colorTransform(tmpK):
    tmpK = tmpK/100
    # first category: 6500K (RGB: 255,249,253)
    if tmpK == 65:
        return [255,249,253]
    # second category: 5500K (RGB: 255,236,224)
    elif tmpK == 55:
        return [255,236,224]
    # third category: 4500K (RGB: 255,219,186)
    elif tmpK == 45:
        return [255,219,186]
    # fourth category: 3500K (RGB: 255,196,137)
    elif tmpK == 35:
        return [255,196,137]
    # fifth category: 2500K (RGB: 255,161,72)
    elif tmpK == 25:
        return [255,161,72]
    else:
        print("invalid colortemp entry")
        return None

def colorTemperature(img, temperature, depth):
    tmpColor = np.array(colorTransform(temperature))

    img_hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)

    updatedColor6 = np.array(alphablend(img, tmpColor, 0.6), dtype='uint8')
    updatedColor = np.array(alphablend(img, tmpColor, 0.5), dtype='uint8')
    updatedColor1 = np.array(alphablend(img, tmpColor, 0.1), dtype='uint8')


    #change the background color back to the original
    print(img.shape)
    print(updatedColor.shape)
    print(np.amax(depth))
    print(np.amin(depth))

    (x_cord, y_cord) = np.where(depth < 1)
    for i, j in zip(x_cord, y_cord):
        updatedColor[i][j] = updatedColor1[i][j]

    (x_cord, y_cord) = np.where(depth < 0.5)
    for i, j in zip(x_cord, y_cord):
        updatedColor[i][j] = updatedColor6[i][j]

    (x_cord, y_cord) = np.where(depth == 0)
    for i,j in zip(x_cord, y_cord):
        updatedColor[i][j] = img[i][j]


    updatedColor_hls = cv2.cvtColor(updatedColor, cv2.COLOR_RGB2HLS)
    updatedColor_hls[:,:,1] = img_hls[:,:,1]
    updatedColor = cv2.cvtColor(updatedColor_hls, cv2.COLOR_HLS2RGB)

    return updatedColor

def img2double(img):
    img = img - np.min(img[:])
    img = img / np.max(img[:])

    return img

def DarkChannel(im,sz):
    b,g,r = cv2.split(im)
    dc = cv2.min(cv2.min(r,g),b);
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(sz,sz))
    dark = cv2.erode(dc,kernel)
    return dark
 
def AtmLight(im,dark):
    [h,w] = im.shape[:2]
    imsz = h*w
    numpx = int(max(math.floor(imsz/1000),1))
    darkvec = dark.reshape(imsz,1);
    imvec = im.reshape(imsz,3);
 
    indices = darkvec.argsort();
    indices = indices[imsz-numpx::]
 
    atmsum = np.zeros([1,3])
    for ind in range(1,numpx):
       atmsum = atmsum + imvec[indices[ind]]
 
    A = atmsum / numpx;
    return A
 
def TransmissionEstimate(im,A,sz):
    omega = 0.9;
    im3 = np.empty(im.shape,im.dtype);
 
    for ind in range(0,3):
        im3[:,:,ind] = im[:,:,ind]/A[0,ind]
 
    transmission = 1 - omega*DarkChannel(im3,sz);
    return transmission
 
def Guidedfilter(im,p,r,eps):
    mean_I = cv2.boxFilter(im,cv2.CV_64F,(r,r));
    mean_p = cv2.boxFilter(p, cv2.CV_64F,(r,r));
    mean_Ip = cv2.boxFilter(im*p,cv2.CV_64F,(r,r));
    cov_Ip = mean_Ip - mean_I*mean_p;
 
    mean_II = cv2.boxFilter(im*im,cv2.CV_64F,(r,r));
    var_I   = mean_II - mean_I*mean_I;
 
    a = cov_Ip/(var_I + eps);
    b = mean_p - a*mean_I;
 
    mean_a = cv2.boxFilter(a,cv2.CV_64F,(r,r));
    mean_b = cv2.boxFilter(b,cv2.CV_64F,(r,r));
 
    q = mean_a*im + mean_b;
    return q;
 
def TransmissionRefine(im,et):
    gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY);
    gray = np.float64(gray)/255;
    r = 60;
    eps = 0.0001;
    t = Guidedfilter(gray,et,r,eps);
 
    return t;
 
def Recover(im,t,A,tx = 0.1):
    res = np.empty(im.shape,im.dtype);
    t = cv2.max(t,tx);
 
    for ind in range(0,3):
        res[:,:,ind] = (im[:,:,ind]-A[0,ind])/t + A[0,ind]
 
    return res
 
def dehaze(src):  
    I = src.astype('float64')/255;
    dark = DarkChannel(I,15);
    A = AtmLight(I,dark);
    te = TransmissionEstimate(I,A,15);
    t = TransmissionRefine(src,te);
    J = Recover(I,t,A,0.7);
    arr = np.hstack((I, J))
    J = J*255
    return J

def img2double(img):
    img = img - np.min(img[:])
    img = img / np.max(img[:])

    return img

def load_depth(name_depth):
    d = np.load(name_depth, allow_pickle=True)
    d = d.item()
    ref_center_dis = d['ref_center_dis']
    depth= d['normalized_depth']
    #print('\t\t>>depth: ', np.shape(depth), type(depth), np.min(depth[:]), np.max(depth[:]))
    depth_ori = depth

    depth = img2double(depth)
    #print('\t\t>>depth: ', np.shape(depth), type(depth), np.min(depth[:]), np.max(depth[:]))

    H, W = np.shape(depth)
    depth = depth.reshape(H, W, 1)
    #print('\t\t>>depth: ', np.shape(depth), type(depth), np.min(depth[:]), np.max(depth[:]))

    #depth = np.array(depth * 255., dtype='uint8')
    depth = np.array(depth * 255)
    #print('\t\t>>depth: ', np.shape(depth), type(depth), np.min(depth[:]), np.max(depth[:]))

    return ref_center_dis, depth, depth_ori

def normalizeVector(v):
    length = np.sqrt(v[0]**2 + v[1]**2 + v[2]**2)
    v = v/length
    return v

def depth2normal(depth):
    #print('depth: ', np.shape(depth))
    depth = depth *10000
    h = np.shape(depth)[0]
    w = np.shape(depth)[1]

    normals = np.zeros((h, w, 3))
    for x in range(1, h-1):
        for y in range(1, w-1):
            dzdx = (float(depth[x+1, y]) - float(depth[x-1, y])) / 2.0 
            dzdy = (float(depth[x, y+1]) - float(depth[x, y-1])) / 2.0 
            d = (-dzdx, -dzdy, 1.0)
            n = normalizeVector(d)
            normals[x,y] = n * 0.5 + 0.5

    normals = np.array(normals)
    normals_255 = np.array(normals * 255)
    
    return normals, normals_255

def print_meta(input_data, tag):
    print('\t>>%s: '%(tag,), input_data.shape, type(input_data), np.max(input_data[:]), np.min(input_data[:]))

def test(para, logger):
    """
    test code
    """ 
    # load model with checkpoint
    if not para.test_only:
        para.test_checkpoint = join(logger.save_dir, 'model_best.pth.tar')
    if para.test_save_dir is None:
        para.test_save_dir = logger.save_dir

    model = Model(para).cuda()
    print(model)
    checkpoint_path = para.test_checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage.cuda())
    model = nn.DataParallel(model)
    model.load_state_dict(checkpoint['state_dict'])
    print('>>model: ', model)

    ds_name = para.dataset
    logger('{} results generating ...'.format(ds_name), prefix='\n') 

    #'track2' in ds_name
    if 'track1' in ds_name:
        ds_type = 'test'
        para.dataset_test = 'track1'
        _test_imgs(para, logger, model, ds_type)
    elif 'track2' in ds_name:
        ds_type = 'test'
        para.dataset_test = 'track2'
        _test_imgs(para, logger, model, ds_type)
    elif ds_name == 'BSD':
        ds_type = 'test'
        _test_torch(para, logger, model, ds_type)
    elif ds_name == 'gopro_ds_lmdb' or ds_name == 'reds_lmdb':
        ds_type = 'valid'
        _test_lmdb(para, logger, model, ds_type)
    else:
        raise NotImplementedError
 
def post_process(output, para, val_range):
    output = normalize_reverse(output, 
        centralize=para.centralize, 
        normalize=para.normalize,
        val_range=val_range)

    output = output.detach().cpu().numpy().transpose((1, 2, 0)).squeeze()
    output = np.clip(output, 0, val_range)
    output = output.astype(np.uint8) if para.data_format == 'RGB' else output.astype(np.uint16)
     
    return output
 
def _test_imgs(para, logger, model, ds_type):
    PSNR = AverageMeter()
    SSIM = AverageMeter()
    timer = AverageMeter()
    results_register = set()
    H, W = 1024, 1024
    val_range = 2.0 ** 8 - 1 if para.data_format == 'RGB' else 2.0 ** 16 - 1
    dataset_path = join(para.data_root, '{}/{}'.format(para.dataset_test, ds_type))
    # seqs = sorted(os.listdir(dataset_path)) 
    seqs = [os.path.join(dataset_path, 'Image%03d.png'%(k+345,)) for k in range(45)]
    seq_length = len(seqs)

    # make save directory
    _, ckpt_name = os.path.split(para.test_checkpoint)
    dir_name = '_'.join(('test', ckpt_name)) 
    # save_dir = join(para.test_save_dir, dir_name)
    save_dir = para.test_save_dir
    os.makedirs(save_dir, exist_ok=True)
 
    # 
    idx = 0
    for seq in seqs:
        idx += 1
        logger('test: image {} results generating ...'.format(seq))

        # read image
        name_img = seq
        name_depth = seq.replace('png', 'npy')

        img = cv2.imread(name_img)
        _, depth, depth_ori = load_depth(name_depth)
        depth_inverse = 255 - depth
        normal_01, normal= depth2normal(depth_ori) 
        input_data = np.concatenate((img, depth.astype('uint8'), depth_inverse.astype('uint8'), normal.astype('uint8')), axis=2)
        #print_meta(input_data, 'input_data')
 
        input_data = input_data.transpose(2, 0, 1)[np.newaxis, ...]
        #print_meta(input_data, 'transpose')

        model.eval()
        with torch.no_grad():
            input_seq = normalize(torch.from_numpy(input_data).float().cuda(), 
                centralize=para.centralize,
                normalize=para.normalize, 
                val_range=val_range)
            #print('\tinput: ', input_seq.shape)

            time_start = time.time()
            diff, output = model([input_seq, ])
            
            timer.update(time.time() - time_start, n=1)
            #print('\toutput: ', output.shape)
            #print('\tdiff: ', diff.shape)

            output = output.squeeze(dim=0)
            diff = diff.squeeze(dim=0)
            #print('\toutput: ', output.shape)
            #print('\tdiff: ', diff.shape)

            output = post_process(output, para, val_range)
            diff = post_process(diff, para, val_range)
            #print('\toutput: ', output.shape)
            
            _, img_name = os.path.split(name_img)
            output_img_path = join(save_dir, img_name)
#============================================================================================
            output = dehaze(output)
            #output = output.astype('float32')
            #output = cv2.cvtColor(output.astype('uint8'), cv2.COLOR_BGR2RGB)
            #output = colorTemperature(output, 4500, depth_ori)
            #output = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
            #output = colorTemperature(output, 4500)
            cv2.imwrite(output_img_path, output)
            #cv2.imwrite(output_img_path.replace('.png', 'in.png'), [input_seq, ]) 
            logger('done, save to {}'.format(output_img_path))


