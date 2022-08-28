import tensorflow as tf
import glob
import os
from scipy import misc
import numpy as np
from utills import *
import utills
import re
from PIL import Image
import random
import cv2

def readPFM(file):
    file = open(file, 'rb')

    header = file.readline().decode('utf-8').rstrip()

    if header == 'PF':
        color = True
    elif header == 'Pf':
        color = False
    else:
        raise Exception('Not a PFM file.')
    dim_match = re.match(r'^(\d+)\s(\d+)\s$', file.readline().decode('utf-8'))
    if dim_match:
        width, height = map(int, dim_match.groups())
    else:
        raise Exception('Malformed PFM header.')

    scale = float(file.readline().rstrip())
    if scale < 0:  # little-endian
        endian = '<'
        scale = -scale
    else:
        endian = '>'  # big-endian

    data = np.fromfile(file, endian + 'f')
    shape = (height, width, 3) if color else (height, width)

    data = np.reshape(data, shape)
    data = np.flipud(data)
    return data, scale

class DataLoaderDAV():
    def __init__(self, config):
        self.data_dir = "F:\SRdata/train_data\StereoData\DAVNet/train/"
        self.patch_size = config.label_size
        self.batch_size = config.batch_size
        self.shuffle_num = config.shuffle_num
        self.prefetch_num = config.prefetch_num
        self.map_parallel_num = config.map_parallel_num

    def get_generator(self):
        file = get_files(self.data_dir)
        imglft = []
        imglftbr = []
        imgrgt = []
        imgrgtbr = []
        imgds = []
        for i in range(len(file)):
            imglft.append(sorted(glob.glob(os.path.join(self.data_dir+'/'+file[i]+'/'+file[i]+'/image_left/','*.png'))))
            imglftbr.append(sorted(glob.glob(os.path.join(self.data_dir+'/'+file[i]+'/'+file[i]+'/image_left_blur_ga/','*.png'))))
            imgrgt.append(sorted(glob.glob(os.path.join(self.data_dir+'/'+file[i]+'/'+file[i]+'/image_right/','*.png'))))
            imgrgtbr.append(sorted(glob.glob(os.path.join(self.data_dir+'/'+file[i]+'/'+file[i]+'/image_right_blur_ga/','*.png'))))
            imgds.append(sorted(glob.glob(os.path.join(self.data_dir+'/'+file[i]+'/'+file[i]+'/disparity_left/','*.png'))))

        p = self.patch_size
        for i in range(len(imglft)):
            imgl = np.array(misc.imread(imglft[i]))
            imgr = np.array(misc.imread(imgrgt[i]))
            imglb = np.array(misc.imread(imglftbr[i]))
            imgrb = np.array(misc.imread(imgrgtbr[i]))
            for fd in range(6):
                if (fd == 0):
                    imgl1 = imgl
                    imglb1 = imglb
                    imgr1 = imgr
                    imgrb1 = imgrb
                elif (fd == 1):
                    imgl1 = np.rot90(imgl, 1)
                    imglb1 = np.rot90(imglb, 1)
                    imgr1 = np.rot90(imgr, 1)
                    imgrb1 = np.rot90(imgrb, 1)
                elif (fd == 2):
                    imgl1 = np.rot90(imgl, 2)
                    imglb1 = np.rot90(imglb, 2)
                    imgr1 = np.rot90(imgr, 2)
                    imgrb1 = np.rot90(imgrb, 2)
                elif (fd == 3):
                    imgl1 = np.rot90(imgl, 3)
                    imglb1 = np.rot90(imglb, 3)
                    imgr1 = np.rot90(imgr, 3)
                    imgrb1 = np.rot90(imgrb, 3)
                elif (fd == 4):
                    imgl1 = np.flip(imgl, 1)
                    imglb1 = np.flip(imglb, 1)
                    imgr1 = np.flip(imgr, 1)
                    imgrb1 = np.flip(imgrb, 1)
                else:
                    imgl1 = np.flip(imgl, 0)
                    imglb1 = np.flip(imglb, 0)
                    imgr1 = np.flip(imgr, 0)
                    imgrb1 = np.flip(imgrb, 0)

                for ds in [1, 2, 3, 4]:
                    imgl1 = misc.imresize(imgl1, 1 / ds, 'bicubic')
                    imglb1 = misc.imresize(imglb1, 1 / ds, 'bicubic')
                    imgr1 = misc.imresize(imgr1, 1 / ds, 'bicubic')
                    imgrb1 = misc.imresize(imgrb1, 1 / ds, 'bicubic')

                    H, W= imgl.shape
                    Ll = imglb1
                    Bl = imgl1
                    Lr = imgrb1
                    Br = imgr1
                    if H< self.patch_size or W< self.patch_size:
                        continue
                    hnum =H // p
                    wnum =W// p
                    for i in range(hnum):
                        for j in range(wnum):
                            h = p * i
                            w = p * j
                            ll = Ll[h: h + p, w:w + p]
                            lr = Lr[h: h + p, w:w + p]
                            bicl =Bl[h: h + p, w:w + p]#
                            bicr =Br[h: h + p, w:w + p]#
                            bicl = np.float32(bicl / 255.0)
                            bicr = np.float32(bicr / 255.0)
                            lr = np.float32(lr / 255.0)
                            ll = np.float32(ll / 255.0)
                            c1 = np.random.rand()
                            c2 = np.random.rand()
                            if c1 < 0.5:
                                bicr, bicl,lr, ll= bicr[::-1, :], bicl[::-1, :],lr[::-1, :], ll[::-1, :]
                            if c2 < 0.5:
                                bicr, bicl,lr, ll = bicr[:, ::-1], bicl[:, ::-1],lr[:, ::-1], ll[:, ::-1]
                            yield bicr, bicl,lr, ll

    def read_pngs(self):
        dataset = tf.data.Dataset.from_generator(self.get_generator, (tf.float32,tf.float32,tf.float32, tf.float32)) # 一个quality
        dataset = dataset.shuffle(self.shuffle_num).prefetch(self.prefetch_num)  #
        dataset = dataset.apply(tf.contrib.data.batch_and_drop_remainder(self.batch_size)).repeat()
        # dataset = dataset.shuffle(self.shuffle_num).prefetch(self.prefetch_num).batch(self.batch_size,drop_remainder=True).repeat()
        cr,cl,br,bl = dataset.make_one_shot_iterator().get_next() # 名字不能和generator中一样
        p = self.patch_size
        cl = tf.reshape(cl, [self.batch_size, p, p, 1])
        cr = tf.reshape(cr, [self.batch_size, p, p, 1])
        bl = tf.reshape(bl, [self.batch_size, p, p, 1])
        br = tf.reshape(br, [self.batch_size, p, p, 1])
        return cr,cl,br,bl

    def get_generatorS(self):
        file = get_files([self.data_dir])[:23]
        imglft = []
        imglftbr = []
        # imgrgt = []
        # imgrgtbr = []
        for i in range(len(file)):
            # print(file[i]+'/'+file[i][len(self.data_dir):]+'/image_left/')
            imglft = sorted(glob.glob(os.path.join(file[i]+'/'+file[i][len(self.data_dir):]+'/image_left/','*.png'))) + \
                     sorted(glob.glob(os.path.join(file[i]+'/'+file[i][len(self.data_dir):]+'/image_right/','*.png')))
            # imglft.append(sorted(glob.glob(os.path.join(self.data_dir+'/'+file[i]+'/'+file[i]+'/image_left/','*.png')))+sorted(glob.glob(os.path.join(self.data_dir+'/'+file[i]+'/'+file[i]+'/image_right/','*.png'))))
            # imglftbr.append(sorted(glob.glob(os.path.join(self.data_dir+'/'+file[i]+'/'+file[i]+'/image_left_blur_ga/','*.png')))).append(sorted(glob.glob(os.path.join(self.data_dir+'/'+file[i]+'/'+file[i]+'/image_right_blur_ga/','*.png'))))
            imglftbr = sorted(glob.glob(os.path.join(file[i]+'/'+file[i][len(self.data_dir):]+'/image_left_blur_ga/','*.png')))\
                            + sorted(glob.glob(os.path.join(file[i]+'/'+file[i][len(self.data_dir):]+'/image_right_blur_ga/','*.png')))

        p = self.patch_size
        for i in range(len(imglft)):
            imgl = np.array(misc.imread(imglft[i]))
            imgrb = np.array(misc.imread(imglftbr[i]))
            for fd in range(6):
                if (fd == 0):
                    imgl1 = imgl
                    imgrb1 = imgrb
                elif (fd == 1):
                    imgl1 = np.rot90(imgl, 1)
                    imgrb1 = np.rot90(imgrb, 1)
                elif (fd == 2):
                    imgl1 = np.rot90(imgl, 2)
                    imgrb1 = np.rot90(imgrb, 2)
                elif (fd == 3):
                    imgl1 = np.rot90(imgl, 3)
                    imgrb1 = np.rot90(imgrb, 3)
                elif (fd == 4):
                    imgl1 = np.flip(imgl, 1)
                    imgrb1 = np.flip(imgrb, 1)
                else:
                    imgl1 = np.flip(imgl, 0)
                    imgrb1 = np.flip(imgrb, 0)

                for ds in [1, 2, 3, 4]:
                    imgl1 = misc.imresize(imgl1, 1 / ds, 'bicubic')
                    imgrb1 = misc.imresize(imgrb1, 1 / ds, 'bicubic')

                    H, W,_= imgl.shape
                    Bl = imgl1
                    Lr = imgrb1
                    if H< self.patch_size or W< self.patch_size:
                        continue
                    hnum =H // p
                    wnum =W// p
                    for i in range(hnum):
                        for j in range(wnum):
                            h = p * i
                            w = p * j
                            l = Lr[h: h + p, w:w + p,:]
                            bic =Bl[h: h + p, w:w + p,:]#
                            if h+p >H or w+p >W:
                                print(h,w,imglft[i])
                            bic = np.float32(bic / 255.0)
                            l = np.float32(l / 255.0)
                            c1 = np.random.rand()
                            c2 = np.random.rand()
                            if c1 < 0.5:
                                bic, l= bic[::-1, :,:], l[::-1, :,:]
                            if c2 < 0.5:
                                bic,l= bic[:, ::-1,:], l[:, ::-1,:]
                            yield bic,l

    def read_pngsS(self):
        dataset = tf.data.Dataset.from_generator(self.get_generatorS, (tf.float32, tf.float32)) # 一个quality
        dataset = dataset.shuffle(self.shuffle_num).prefetch(self.prefetch_num)  #
        dataset = dataset.apply(tf.contrib.data.batch_and_drop_remainder(self.batch_size)).repeat()
        c,b = dataset.make_one_shot_iterator().get_next() # 名字不能和generator中一样
        p = self.patch_size
        c = tf.reshape(c, [self.batch_size, p, p, 3])
        b= tf.reshape(b, [self.batch_size, p, p, 3])
        return c,b

class DataLoader():
    def __init__(self, config):
        self.data_dirl = config.data_dirl
        self.data_dirr = config.data_dirr
        self.data_dirrw = config.data_dirrwarp
        self.patch_size = config.label_size
        self.scale = config.scale
        self.batch_size = config.batch_size
        self.shuffle_num = config.shuffle_num
        self.prefetch_num = config.prefetch_num
        self.map_parallel_num = config.map_parallel_num

    def get_generator(self):
        img_pathslb = sorted(glob.glob(os.path.join(self.data_dirl,'*.bmp')))
        img_pathslj = sorted(glob.glob(os.path.join(self.data_dirl,'*.jpg')))
        img_pathsl = img_pathslj + img_pathslb
        img_pathsrb = sorted(glob.glob(os.path.join(self.data_dirr,'*.bmp')))
        img_pathsrj = sorted(glob.glob(os.path.join(self.data_dirr, '*.jpg')))
        img_pathsr = img_pathsrj + img_pathsrb
        p = self.patch_size
        lp = p//self.scale
        for i in range(len(img_pathsl)):
            imgrgbl = misc.imread(img_pathsl[i])
            imgrgbr = misc.imread(img_pathsr[i])
            imgorl = rgb2y(imgrgbl)[:, :]
            imgorr = rgb2y(imgrgbr)[:, :]
            h,w = imgorl.shape
            h = h-h%self.scale
            w = w-w%self.scale
            imgorl = imgorl[:h,:w]
            imgorr = imgorr[:h,:w]
            for fd in range(6):
                if (fd == 0):
                    imgl = imgorl
                    imgr = imgorr
                elif (fd == 1):
                    imgl = np.rot90(imgorl, 1)
                    imgr = np.rot90(imgorr, 1)
                elif (fd == 2):
                    imgl = np.rot90(imgorl, 2)
                    imgr = np.rot90(imgorr, 2)
                elif (fd == 3):
                    imgl = np.rot90(imgorl, 3)
                    imgr = np.rot90(imgorr, 3)
                elif (fd == 4):
                    imgl = np.flip(imgorl, 1)
                    imgr = np.flip(imgorr, 1)
                else:
                    imgl = np.flip(imgorl, 0)
                    imgr = np.flip(imgorr, 0)
                for ds in [1, 2, 3, 4]:
                    imgl = misc.imresize(imgl, 1 / ds, 'bicubic')
                    imgr = misc.imresize(imgr, 1 / ds, 'bicubic')
                    H, W= imgl.shape
                    Ll = misc.imresize(imgl, 1 / self.scale, 'bicubic')
                    Bl = misc.imresize(Ll, [H,W], 'bicubic')
                    Lr = misc.imresize(imgr, 1 / self.scale, 'bicubic')
                    Br = misc.imresize(Lr,[H,W], 'bicubic')
                    if H< self.patch_size or W< self.patch_size:
                        continue
                    hnum =H // p
                    wnum =W// p
                    for i in range(hnum):  # 每张图像有几个patch
                        for j in range(wnum):
                            h = p * i
                            hl = lp*i
                            wl = lp*j
                            w = p * j
                            gtl = imgl[h: h + p, w:w + p]
                            gtr = imgr[h: h + p, w:w + p]
                            ll = Ll[hl: hl + lp, wl:wl + lp]#misc.imresize(gtl, 1 / self.scale, 'bicubic')#
                            lr = Lr[hl: hl + lp, wl:wl + lp]#misc.imresize(gtr, 1 / self.scale, 'bicubic')#
                            bicl =Bl[h: h + p, w:w + p]# misc.imresize(ll, [p,p], 'bicubic')#
                            bicr =Br[h: h + p, w:w + p]# misc.imresize(lr, [p,p], 'bicubic')#
                            gtl = np.float32(gtl / 255.0)
                            gtr = np.float32(gtr / 255.0)
                            bicl = np.float32(bicl / 255.0)
                            bicr = np.float32(bicr / 255.0)
                            lr = np.float32(lr / 255.0)
                            ll = np.float32(ll / 255.0)
                            c1 = np.random.rand()
                            c2 = np.random.rand()
                            if c1 < 0.5:
                                gtr, gtl,bicr, bicl,lr, ll= gtr[::-1, :], gtl[::-1, :],bicr[::-1, :], bicl[::-1, :],lr[::-1, :], ll[::-1, :]
                            if c2 < 0.5:
                                gtr, gtl,bicr, bicl,lr, ll = gtr[:, ::-1], gtl[:, ::-1],bicr[:, ::-1], bicl[:, ::-1],lr[:, ::-1], ll[:, ::-1]
                            yield gtr, gtl,bicr, bicl,lr, ll

    def read_pngs(self):
        dataset = tf.data.Dataset.from_generator(self.get_generator, (tf.float32,tf.float32,tf.float32,tf.float32,tf.float32, tf.float32)) # 一个quality
        dataset = dataset.shuffle(self.shuffle_num).prefetch(self.prefetch_num)  #
        dataset = dataset.apply(tf.contrib.data.batch_and_drop_remainder(self.batch_size)).repeat()
        # dataset = dataset.shuffle(self.shuffle_num).prefetch(self.prefetch_num).batch(self.batch_size,drop_remainder=True).repeat()
        cr,cl,br,bl,lrs,lls = dataset.make_one_shot_iterator().get_next() # 名字不能和generator中一样
        p = self.patch_size
        lp = p // self.scale
        lrs = tf.reshape(lrs, [self.batch_size, lp, lp, 1])
        lls = tf.reshape(lls, [self.batch_size, lp, lp, 1])
        cl = tf.reshape(cl, [self.batch_size, p, p, 1])
        cr = tf.reshape(cr, [self.batch_size, p, p, 1])
        bl = tf.reshape(bl, [self.batch_size, p, p, 1])
        br = tf.reshape(br, [self.batch_size, p, p, 1])
        return lrs,lls,cr,cl,br,bl

    def get_generatorRGB(self):
        img_pathsl = sorted(glob.glob(os.path.join(self.data_dirl,'*.bmp')))+ sorted(glob.glob(os.path.join(self.data_dirl,'*.jpg')))#sorted(glob.glob(os.path.join(self.data_dirl,'*.png')))#
        img_pathsr = sorted(glob.glob(os.path.join(self.data_dirr,'*.bmp'))) + sorted(glob.glob(os.path.join(self.data_dirr, '*.jpg')))#sorted(glob.glob(os.path.join(self.data_dirr,'*.png')))#
        p = self.patch_size
        lp = p//self.scale
        for i in range(len(img_pathsl)):
            imgorl = np.array(misc.imread(img_pathsl[i]))
            imgorr = np.array(misc.imread(img_pathsr[i]))
            h,w,_ = imgorl.shape
            h = h-h%self.scale
            w = w-w%self.scale
            imgorl = imgorl[:h,:w,:3]
            imgorr = imgorr[:h,:w,:3]
            for fd in range(6):
                if (fd == 0):
                    imgl = imgorl
                    imgr = imgorr
                elif (fd == 1):
                    imgl = np.rot90(imgorl, 1)
                    imgr = np.rot90(imgorr, 1)
                elif (fd == 2):
                    imgl = np.rot90(imgorl, 2)
                    imgr = np.rot90(imgorr, 2)
                elif (fd == 3):
                    imgl = np.rot90(imgorl, 3)
                    imgr = np.rot90(imgorr, 3)
                elif (fd == 4):
                    imgl = np.flip(imgorl, 1)
                    imgr = np.flip(imgorr, 1)
                else:
                    imgl = np.flip(imgorl, 0)
                    imgr = np.flip(imgorr, 0)
                for ds in [1, 2, 3, 4]: # ]:#
                    imgl0 = misc.imresize(imgl, 1 / ds, 'bicubic')
                    imgr0 = misc.imresize(imgr, 1 / ds, 'bicubic')
                    H, W,c= imgl0.shape
                    Ll = misc.imresize(imgl0, 1 / self.scale, 'bicubic')
                    Bl = misc.imresize(Ll, [H,W,c], 'bicubic')
                    Lr = misc.imresize(imgr0, 1 / self.scale, 'bicubic')
                    Br = misc.imresize(Lr,[H,W,c], 'bicubic')
                    if H< self.patch_size or W< self.patch_size:
                        continue
                    hnum =H // p
                    wnum =W// p
                    for i in range(hnum):  # 每张图像有几个patch
                        for j in range(wnum):
                            h = p * i
                            hl = lp*i
                            wl = lp*j
                            w = p * j
                            gtl = imgl0[h: h + p, w:w + p,:]
                            gtr = imgr0[h: h + p, w:w + p,:]
                            ll = Ll[hl: hl + lp, wl:wl + lp,:]
                            lr = Lr[hl: hl + lp, wl:wl + lp,:]
                            bicl =Bl[h: h + p, w:w + p,:]
                            bicr =Br[h: h + p, w:w + p,:]
                            gtl = np.float32(gtl / 255.0)
                            gtr = np.float32(gtr / 255.0)
                            bicl = np.float32(bicl / 255.0)
                            bicr = np.float32(bicr / 255.0)
                            lr = np.float32(lr / 255.0)
                            ll = np.float32(ll / 255.0)
                            c1 = np.random.rand()
                            c2 = np.random.rand()
                            if c1 < 0.5:
                                gtr, gtl,bicr, bicl,lr, ll= gtr[::-1, :,:], gtl[::-1, :,:],bicr[::-1,:, :], bicl[::-1, :,:],lr[::-1, :,:], ll[::-1, :,:]
                            if c2 < 0.5:
                                gtr, gtl,bicr, bicl,lr, ll = gtr[:, ::-1,:], gtl[:, ::-1,:],bicr[:, ::-1,:], bicl[:, ::-1,:],lr[:, ::-1,:], ll[:, ::-1,:]
                            yield gtr, gtl,bicr, bicl,lr, ll

    def read_pngsRGB(self):
        dataset = tf.data.Dataset.from_generator(self.get_generatorRGB, (tf.float32,tf.float32,tf.float32,tf.float32,tf.float32, tf.float32)) # 一个quality
        dataset = dataset.shuffle(self.shuffle_num).prefetch(self.prefetch_num)  #
        dataset = dataset.apply(tf.contrib.data.batch_and_drop_remainder(self.batch_size)).repeat()
        #dataset = dataset.shuffle(self.shuffle_num).prefetch(self.prefetch_num).batch(self.batch_size,drop_remainder=True).repeat()
        cr,cl,br,bl,lrs,lls = dataset.make_one_shot_iterator().get_next() # 名字不能和generator中一样
        p = self.patch_size
        lp = p // self.scale
        lrs = tf.reshape(lrs, [self.batch_size, lp, lp,3])
        lls = tf.reshape(lls, [self.batch_size, lp, lp,3])
        cl = tf.reshape(cl, [self.batch_size, p, p, 3])
        cr = tf.reshape(cr, [self.batch_size, p, p,3])
        bl = tf.reshape(bl, [self.batch_size, p, p,3])
        br = tf.reshape(br, [self.batch_size, p, p,3])
        return lrs,lls,cr,cl,br,bl

    def get_generatorRGB_LR(self): # left right exchange
        img_pathsl = sorted(glob.glob(os.path.join(self.data_dirl,'*.bmp')))+ sorted(glob.glob(os.path.join(self.data_dirl,'*.jpg')))+sorted(glob.glob(os.path.join(self.data_dirr, '*.bmp'))) + sorted(
            glob.glob(os.path.join(self.data_dirr, '*.jpg')))
        img_pathsr = sorted(glob.glob(os.path.join(self.data_dirr,'*.bmp'))) + sorted(glob.glob(os.path.join(self.data_dirr, '*.jpg')))+sorted(glob.glob(os.path.join(self.data_dirl, '*.bmp'))) + sorted(
            glob.glob(os.path.join(self.data_dirl, '*.jpg')))
        p = self.patch_size
        lp = p//self.scale
        for i in range(len(img_pathsl)):
            imgorl = np.array(misc.imread(img_pathsl[i]))
            imgorr = np.array(misc.imread(img_pathsr[i]))
            h, w, _ = imgorl.shape
            h = h - h % self.scale
            w = w - w % self.scale
            imgorl = imgorl[:h, :w, :]
            imgorr = imgorr[:h, :w, :]
            for fd in range(6):
                if (fd == 0):
                    imgl = imgorl
                    imgr = imgorr
                elif (fd == 1):
                    imgl = np.rot90(imgorl, 1)
                    imgr = np.rot90(imgorr, 1)
                elif (fd == 2):
                    imgl = np.rot90(imgorl, 2)
                    imgr = np.rot90(imgorr, 2)
                elif (fd == 3):
                    imgl = np.rot90(imgorl, 3)
                    imgr = np.rot90(imgorr, 3)
                elif (fd == 4):
                    imgl = np.flip(imgorl, 1)
                    imgr = np.flip(imgorr, 1)
                else:
                    imgl = np.flip(imgorl, 0)
                    imgr = np.flip(imgorr, 0)
                for ds in [1, 2, 3, 4]:
                    imgl = misc.imresize(imgl, 1 / ds, 'bicubic')
                    imgr = misc.imresize(imgr, 1 / ds, 'bicubic')
                    H, W, c = imgl.shape
                    Ll = misc.imresize(imgl, 1 / self.scale, 'bicubic')
                    Bl = misc.imresize(Ll, [H, W, c], 'bicubic')
                    Lr = misc.imresize(imgr, 1 / self.scale, 'bicubic')
                    Br = misc.imresize(Lr, [H, W, c], 'bicubic')
                    if H < self.patch_size or W < self.patch_size:
                        continue
                    hnum = H // p
                    wnum = W // p
                    for i in range(hnum):  # 每张图像有几个patch
                        for j in range(wnum):
                            h = p * i
                            hl = lp * i
                            wl = lp * j
                            w = p * j
                            gtl = imgl[h: h + p, w:w + p, :]
                            gtr = imgr[h: h + p, w:w + p, :]
                            ll = Ll[hl: hl + lp, wl:wl + lp, :]
                            lr = Lr[hl: hl + lp, wl:wl + lp, :]
                            bicl = Bl[h: h + p, w:w + p, :]
                            bicr = Br[h: h + p, w:w + p, :]
                            gtl = np.float32(gtl / 255.0)
                            gtr = np.float32(gtr / 255.0)
                            bicl = np.float32(bicl / 255.0)
                            bicr = np.float32(bicr / 255.0)
                            lr = np.float32(lr / 255.0)
                            ll = np.float32(ll / 255.0)
                            c1 = np.random.rand()
                            c2 = np.random.rand()
                            if c1 < 0.5:
                                gtr, gtl, bicr, bicl, lr, ll = gtr[::-1, :, :], gtl[::-1, :, :], bicr[::-1, :,:],\
                                                               bicl[::-1, :,:], lr[::-1, :,:], ll[::-1,:, :]
                            if c2 < 0.5:
                                gtr, gtl, bicr, bicl, lr, ll = gtr[:, ::-1, :], gtl[:, ::-1, :], bicr[:, ::-1,:], \
                                                               bicl[:, ::-1,:], lr[:, ::-1,:], ll[:,::-1,:]
                            yield gtr, gtl, bicr, bicl, lr, ll
    # left right change
    def read_pngsRGBLR(self):
        dataset = tf.data.Dataset.from_generator(self.get_generatorRGB_LR, (tf.float32,tf.float32,tf.float32,tf.float32,tf.float32, tf.float32)) # 一个quality
        dataset = dataset.shuffle(self.shuffle_num).prefetch(self.prefetch_num)  #
        dataset = dataset.apply(tf.contrib.data.batch_and_drop_remainder(self.batch_size)).repeat()
        #        dataset = dataset.shuffle(self.shuffle_num).prefetch(self.prefetch_num).batch(self.batch_size,drop_remainder=True).repeat()
        cr,cl,br,bl,lrs,lls = dataset.make_one_shot_iterator().get_next() # 名字不能和generator中一样
        p = self.patch_size
        lp = p // self.scale
        lrs = tf.reshape(lrs, [self.batch_size, lp, lp,3])
        lls = tf.reshape(lls, [self.batch_size, lp, lp,3])
        cl = tf.reshape(cl, [self.batch_size, p, p, 3])
        cr = tf.reshape(cr, [self.batch_size, p, p,3])
        bl = tf.reshape(bl, [self.batch_size, p, p,3])
        br = tf.reshape(br, [self.batch_size, p, p,3])
        return lrs,lls,cr,cl,br,bl

    def get_generatorof(self):
        img_pathsl = sorted(glob.glob(os.path.join(self.data_dirl,'*.bmp')))
        img_pathsr = sorted(glob.glob(os.path.join(self.data_dirr,'*.bmp')))
        img_pathsrw = sorted(glob.glob(os.path.join(self.data_dirrw,'*.bmp')))
        p = self.patch_size
        lp = p//self.scale
        for i in range(len(img_pathsl)):
            imgrgbl = misc.imread(img_pathsl[i])
            imgrgbr = misc.imread(img_pathsr[i])
            imgrgbrw = np.array(misc.imread(img_pathsrw[i]))
            imgorl = rgb2y(imgrgbl)[:, :]
            imgorr = rgb2y(imgrgbr)[:, :]
            h,w = imgorl.shape
            h0,w0 = imgrgbrw.shape
            if h != h0 or w != w0:
                print("mismatch!!!!!!!!!!!!!!!")
            h = h-h%self.scale
            w = w-w%self.scale
            imgorl = imgorl[:h,:w]
            imgorr = imgorr[:h,:w]
            imgorrw = imgrgbrw[:h,:w]
            for fd in range(6):
                if (fd == 0):
                    imgl = imgorl
                    imgr = imgorr
                    imgrw = imgorrw
                elif (fd == 1):
                    imgl = np.rot90(imgorl, 1)
                    imgr = np.rot90(imgorr, 1)
                    imgrw = np.rot90(imgorrw, 1)
                elif (fd == 2):
                    imgl = np.rot90(imgorl, 2)
                    imgr = np.rot90(imgorr, 2)
                    imgrw = np.rot90(imgorrw, 2)
                elif (fd == 3):
                    imgl = np.rot90(imgorl, 3)
                    imgr = np.rot90(imgorr, 3)
                    imgrw = np.rot90(imgorrw, 3)
                elif (fd == 4):
                    imgl = np.flip(imgorl, 1)
                    imgr = np.flip(imgorr, 1)
                    imgrw = np.flip(imgorrw, 1)
                else:
                    imgl = np.flip(imgorl, 0)
                    imgr = np.flip(imgorr, 0)
                    imgrw = np.flip(imgorrw, 0)
                for ds in [1, 2, 3, 4]:
                    imgl = misc.imresize(imgl, 1 / ds, 'bicubic')
                    imgr = misc.imresize(imgr, 1 / ds, 'bicubic')
                    imgrw = misc.imresize(imgrw, 1 / ds, 'bicubic')
                    H, W= imgl.shape
                    Ll = misc.imresize(imgl, 1 / self.scale, 'bicubic')
                    Bl = misc.imresize(Ll, [H,W], 'bicubic')
                    Lr = misc.imresize(imgr, 1 / self.scale, 'bicubic')
                    Br = misc.imresize(Lr,[H,W], 'bicubic')
                    Lrw = misc.imresize(imgrw, 1 / self.scale, 'bicubic')
                    Brw = misc.imresize(Lrw,[H,W], 'bicubic')
                    if H< self.patch_size or W< self.patch_size:
                        continue
                    hnum =H // p
                    wnum =W// p
                    for i in range(hnum):  # 每张图像有几个patch
                        for j in range(wnum):
                            h = p * i
                            hl = lp*i
                            wl = lp*j
                            w = p * j
                            gtl = imgl[h: h + p, w:w + p]
                            gtr = imgr[h: h + p, w:w + p]
                            ll = Ll[hl: hl + lp, wl:wl + lp]#misc.imresize(gtl, 1 / self.scale, 'bicubic')#
                            lr = Lr[hl: hl + lp, wl:wl + lp]#misc.imresize(gtr, 1 / self.scale, 'bicubic')#
                            lrw = Lrw[hl: hl + lp, wl:wl + lp]#misc.imresize(gtr, 1 / self.scale, 'bicubic')#
                            bicl =Bl[h: h + p, w:w + p]# misc.imresize(ll, [p,p], 'bicubic')#
                            bicr =Br[h: h + p, w:w + p]# misc.imresize(lr, [p,p], 'bicubic')#
                            bicrw =Brw[h: h + p, w:w + p]# misc.imresize(lr, [p,p], 'bicubic')#
                            gtl = np.float32(gtl / 255.0)
                            gtr = np.float32(gtr / 255.0)
                            bicl = np.float32(bicl / 255.0)
                            bicr = np.float32(bicr / 255.0)
                            bicrw = np.float32(bicrw / 255.0)
                            lr = np.float32(lr / 255.0)
                            lrw = np.float32(lrw / 255.0)
                            ll = np.float32(ll / 255.0)
                            c1 = np.random.rand()
                            c2 = np.random.rand()
                            if c1 < 0.5:
                                gtr, gtl,bicr,bicrw, bicl,lr,lrw, ll= gtr[::-1, :], gtl[::-1, :],bicr[::-1, :],bicrw[::-1, :], bicl[::-1, :],lr[::-1, :],lrw[::-1, :], ll[::-1, :]
                            if c2 < 0.5:
                                gtr, gtl,bicr,bicrw, bicl,lr,lrw, ll = gtr[:, ::-1], gtl[:, ::-1],bicr[:, ::-1],bicrw[:, ::-1], bicl[:, ::-1],lr[:, ::-1],lrw[:, ::-1], ll[:, ::-1]
                            yield gtr, gtl,bicr,bicrw, bicl,lr,lrw, ll
    # 读入现成的warp后的右图 及左右图的LR bic HR
    def read_pngsof(self):
        dataset = tf.data.Dataset.from_generator(self.get_generatorof, (tf.float32,tf.float32,tf.float32,tf.float32,tf.float32,tf.float32,tf.float32, tf.float32)) # 一个quality
        dataset = dataset.shuffle(self.shuffle_num).prefetch(self.prefetch_num)  #
        dataset = dataset.apply(tf.contrib.data.batch_and_drop_remainder(self.batch_size)).repeat()
        #dataset = dataset.shuffle(self.shuffle_num).prefetch(self.prefetch_num).batch(self.batch_size).repeat()
        cr,cl,br,brw,bl,lrs,lrws,lls = dataset.make_one_shot_iterator().get_next() # 名字不能和generator中一样
        p = self.patch_size
        lp = p // self.scale
        lrs = tf.reshape(lrs, [self.batch_size, lp, lp, 1])
        lrws = tf.reshape(lrws, [self.batch_size, lp, lp, 1])
        lls = tf.reshape(lls, [self.batch_size, lp, lp, 1])
        cl = tf.reshape(cl, [self.batch_size, p, p, 1])
        cr = tf.reshape(cr, [self.batch_size, p, p, 1])
        bl = tf.reshape(bl, [self.batch_size, p, p, 1])
        br = tf.reshape(br, [self.batch_size, p, p, 1])
        brw = tf.reshape(brw, [self.batch_size, p, p, 1])
        return lrs,lrws,lls,cr,cl,br,brw,bl

    def get_generatorS(self):
        img_paths = sorted(glob.glob(os.path.join(self.data_dirl, '*.bmp'))) + sorted(
            glob.glob(os.path.join(self.data_dirl, '*.jpg'))) \
                    + sorted(glob.glob(os.path.join(self.data_dirr, '*.bmp'))) + sorted(
            glob.glob(os.path.join(self.data_dirr, '*.jpg')))
        p = self.patch_size
        lp = p // self.scale
        for i in range(len(img_paths)):
            imgrgbl = misc.imread(img_paths[i])
            imgorl = utills.rgb2y(imgrgbl)[:, :]
            h, w = imgorl.shape
            h = h - h % self.scale
            w = w - w % self.scale
            imgorl = imgorl[:h, :w]
            for fd in range(6):
                if (fd == 0):
                    imgl = imgorl
                elif (fd == 1):
                    imgl = np.rot90(imgorl, 1)
                elif (fd == 2):
                    imgl = np.rot90(imgorl, 2)
                elif (fd == 3):
                    imgl = np.rot90(imgorl, 3)
                elif (fd == 4):
                    imgl = np.flip(imgorl, 1)
                else:
                    imgl = np.flip(imgorl, 0)
                for ds in [1, 2, 3, 4]:
                    imgl = misc.imresize(imgl, 1 / ds, 'bicubic')
                    H, W = imgl.shape
                    Ll = misc.imresize(imgl, 1 / self.scale, 'bicubic')
                    Bl = misc.imresize(Ll, [H, W], 'bicubic')
                    if H < self.patch_size or W < self.patch_size:
                        continue
                    hnum = H // p
                    wnum = W // p
                    for i in range(hnum):  # 每张图像有几个patch
                        for j in range(wnum):
                            h = p * i
                            hl = lp * i
                            wl = lp * j
                            w = p * j
                            gtl = imgl[h: h + p, w:w + p]
                            ll = Ll[hl: hl + lp, wl:wl + lp]  # misc.imresize(gtl, 1 / self.scale, 'bicubic')#
                            bicl = Bl[h: h + p, w:w + p]  # misc.imresize(ll, [p,p], 'bicubic')#
                            gtl = np.float32(gtl / 255.0)
                            bicl = np.float32(bicl / 255.0)
                            ll = np.float32(ll / 255.0)
                            c1 = np.random.rand()
                            c2 = np.random.rand()
                            if c1 < 0.5:
                                gtl, bicl, ll = gtl[::-1, :], bicl[::-1, :], ll[::-1, :]
                            if c2 < 0.5:
                                gtl, bicl, ll = gtl[:, ::-1], bicl[:, ::-1], ll[:, ::-1]
                            yield gtl, bicl, ll
    def get_generatorSrgb(self):
        img_paths = sorted(glob.glob(os.path.join(self.data_dirl, '*.bmp'))) + sorted(
            glob.glob(os.path.join(self.data_dirl, '*.jpg'))) \
                    + sorted(glob.glob(os.path.join(self.data_dirr, '*.bmp'))) + sorted(
            glob.glob(os.path.join(self.data_dirr, '*.jpg')))
        p = self.patch_size
        lp = p // self.scale
        for i in range(len(img_paths)):
            imgrgbl = misc.imread(img_paths[i])
            imgorl = np.array(imgrgbl)
            h, w,c = imgorl.shape
            h = h - h % self.scale
            w = w - w % self.scale
            imgorl = imgorl[:h, :w,:3]
            for fd in range(6):
                if (fd == 0):
                    imgl = imgorl
                elif (fd == 1):
                    imgl = np.rot90(imgorl, 1)
                elif (fd == 2):
                    imgl = np.rot90(imgorl, 2)
                elif (fd == 3):
                    imgl = np.rot90(imgorl, 3)
                elif (fd == 4):
                    imgl = np.flip(imgorl, 1)
                else:
                    imgl = np.flip(imgorl, 0)
                for ds in [1, 2, 3, 4]:
                    imgl = misc.imresize(imgl, 1 / ds, 'bicubic')
                    H, W,c = imgl.shape
                    Ll = misc.imresize(imgl, 1 / self.scale, 'bicubic')
                    Bl = misc.imresize(Ll, [H, W,c], 'bicubic')
                    if H < self.patch_size or W < self.patch_size:
                        continue
                    hnum = H // p
                    wnum = W // p
                    imgl = np.float32(imgl / 255.0)
                    Ll = np.float32(Ll / 255.0)
                    Bl = np.float32(Bl / 255.0)
                    for i in range(hnum):  # 每张图像有几个patch
                        for j in range(wnum):
                            h = p * i
                            hl = lp * i
                            wl = lp * j
                            w = p * j
                            gtl = imgl[h: h + p, w:w + p,:]
                            ll = Ll[hl: hl + lp, wl:wl + lp,:]
                            bicl = Bl[h: h + p, w:w + p,:]
                            c1 = np.random.rand()
                            c2 = np.random.rand()
                            if c1 < 0.5:
                                gtl, bicl, ll = gtl[::-1, :,:], bicl[::-1, :,:], ll[::-1, :,:]
                            if c2 < 0.5:
                                gtl, bicl, ll = gtl[:, ::-1,:], bicl[:, ::-1,:], ll[:, ::-1,:]
                            yield gtl, bicl, ll
    # read single image(左右图混合一起)
    def read_pngsS(self):
        dataset = tf.data.Dataset.from_generator(self.get_generatorS, (tf.float32,tf.float32, tf.float32)) # 一个quality
        dataset = dataset.shuffle(self.shuffle_num).prefetch(self.prefetch_num)  #
        dataset = dataset.apply(tf.contrib.data.batch_and_drop_remainder(self.batch_size)).repeat()
        #dataset = dataset.shuffle(self.shuffle_num).prefetch(self.prefetch_num).batch(self.batch_size).repeat()
        c,b, l = dataset.make_one_shot_iterator().get_next() # 名字不能和generator中一样
        p = self.patch_size
        lp = p // self.scale
        l = tf.reshape(l, [self.batch_size, lp, lp, 1])
        c = tf.reshape(c, [self.batch_size, p, p, 1])
        b = tf.reshape(b, [self.batch_size, p, p, 1])
        return c,b,l
    def read_pngsSrgb(self):
        dataset = tf.data.Dataset.from_generator(self.get_generatorSrgb, (tf.float32,tf.float32, tf.float32)) # 一个quality
        dataset = dataset.shuffle(self.shuffle_num).prefetch(self.prefetch_num)  #
        dataset = dataset.apply(tf.contrib.data.batch_and_drop_remainder(self.batch_size)).repeat()
        c,b, l = dataset.make_one_shot_iterator().get_next() # 名字不能和generator中一样
        p = self.patch_size
        lp = p // self.scale
        l = tf.reshape(l, [self.batch_size, lp, lp,3])
        c = tf.reshape(c, [self.batch_size, p, p,3])
        b = tf.reshape(b, [self.batch_size, p, p,3])
        return c,b,l

    def get_generatorOF(self):
        img_pathsl = sorted(glob.glob(os.path.join(self.data_dirl,'*.bmp')))+sorted(glob.glob(os.path.join(self.data_dirl,'*.jpg')))\
                    +sorted(glob.glob(os.path.join(self.data_dirr,'*.bmp')))+sorted(glob.glob(os.path.join(self.data_dirr, '*.jpg')))
        img_pathsr = sorted(glob.glob(os.path.join(self.data_dirr, '*.bmp'))) + sorted(
            glob.glob(os.path.join(self.data_dirr, '*.jpg'))) \
                     + sorted(glob.glob(os.path.join(self.data_dirl, '*.bmp'))) + sorted(
            glob.glob(os.path.join(self.data_dirl, '*.jpg')))

        p = self.patch_size
        lp = p//self.scale
        for i in range(len(img_pathsr)):
            imgrgbl = misc.imread(img_pathsl[i])
            imgrgbr = misc.imread(img_pathsr[i])
            imgorl = utills.rgb2y(imgrgbl)[:, :]
            imgorr = utills.rgb2y(imgrgbr)[:, :]
            h,w = imgorl.shape
            h = h-h%self.scale
            w = w-w%self.scale
            imgorl = imgorl[:h,:w]
            imgorr = imgorr[:h,:w]
            for fd in range(6):
                if (fd == 0):
                    imgl = imgorl
                    imgr = imgorr
                elif (fd == 1):
                    imgl = np.rot90(imgorl, 1)
                    imgr = np.rot90(imgorr, 1)
                elif (fd == 2):
                    imgl = np.rot90(imgorl, 2)
                    imgr = np.rot90(imgorr, 2)
                elif (fd == 3):
                    imgl = np.rot90(imgorl, 3)
                    imgr = np.rot90(imgorr, 3)
                elif (fd == 4):
                    imgl = np.flip(imgorl, 1)
                    imgr = np.flip(imgorr, 1)
                else:
                    imgl = np.flip(imgorl, 0)
                    imgr = np.flip(imgorr, 0)
                for ds in [1, 1.5, 2, 3, 4]:
                    imgl = misc.imresize(imgl, 1 / ds, 'bicubic')
                    imgr = misc.imresize(imgr, 1 / ds, 'bicubic')
                    H, W= imgl.shape
                    '''
                    Ll = misc.imresize(imgl, 1 / self.scale, 'bicubic')
                    Lr = misc.imresize(imgl, 1 / self.scale, 'bicubic')
                    Bl = misc.imresize(Ll, [H,W], 'bicubic')
                    Br = misc.imresize(Lr, [H,W], 'bicubic')
                    '''
                    if H< self.patch_size or W< self.patch_size:
                        continue
                    hnum =H // p
                    wnum =W// p
                    for i in range(hnum):  # 每张图像有几个patch
                        for j in range(wnum):
                            h = p * i
                            hl = lp*i
                            wl = lp*j
                            w = p * j
                            gtl = imgl[h: h + p, w:w + p]
                            gtr = imgr[h: h + p, w:w + p]
                            '''
                            ll = Ll[hl: hl + lp, wl:wl + lp]#misc.imresize(gtl, 1 / self.scale, 'bicubic')#
                            bicl =Bl[h: h + p, w:w + p]# misc.imresize(ll, [p,p], 'bicubic')#
                            bicl = np.float32(bicl / 255.0)
                            ll = np.float32(ll / 255.0)
                            '''
                            gtl = np.float32(gtl / 255.0)
                            gtr = np.float32(gtr / 255.0)
                            c1 = np.random.rand()
                            c2 = np.random.rand()
                            if c1 < 0.5:
                                gtl,gtr= gtl[::-1, :],gtr[::-1, :]
                            if c2 < 0.5:
                                gtl,gtr = gtl[:, ::-1],gtr[:, ::-1]
                            yield gtl,gtr
    # read 两个 image(左右图混合一起)
    def read_pngsOF(self):
        dataset = tf.data.Dataset.from_generator(self.get_generatorOF, (tf.float32,tf.float32)) # 一个quality
        dataset = dataset.shuffle(self.shuffle_num).prefetch(self.prefetch_num)  #
        dataset = dataset.apply(tf.contrib.data.batch_and_drop_remainder(self.batch_size)).repeat()
        cl,cr = dataset.make_one_shot_iterator().get_next() # 名字不能和generator中一样
        p = self.patch_size
        cl = tf.reshape(cl, [self.batch_size, p, p, 1])
        cr = tf.reshape(cr, [self.batch_size, p, p, 1])
        return cl,cr

    def get_generatorcvpr(self):
        img_pathsl = sorted(glob.glob(os.path.join(self.data_dirl,'*.bmp')))+ sorted(glob.glob(os.path.join(self.data_dirl,'*.jpg')))
        img_pathsr = sorted(glob.glob(os.path.join(self.data_dirr,'*.bmp')))+sorted(glob.glob(os.path.join(self.data_dirr, '*.jpg')))
        p = self.patch_size
        for i in range(len(img_pathsl)):
            imgrgbl = misc.imread(img_pathsl[i])
            imgrgbr = misc.imread(img_pathsr[i])
            imgorl = utills.rgb2ycbcr(imgrgbl)
            imgorr = utills.rgb2ycbcr(imgrgbr)
            h,w,_ = imgorl.shape
            h = h-h%self.scale
            w = w-w%self.scale
            imgorl = imgorl[:h,:w,:]
            imgorr = imgorr[:h,:w,:]
            for fd in range(0,6):
                if (fd == 0):
                    imgl = imgorl
                    imgr = imgorr
                elif (fd == 1):
                    imgl = np.rot90(imgorl, 1)
                    imgr = np.rot90(imgorr, 1)
                elif (fd == 2):
                    imgl = np.rot90(imgorl, 2)
                    imgr = np.rot90(imgorr, 2)
                elif (fd == 3):
                    imgl = np.rot90(imgorl, 3)
                    imgr = np.rot90(imgorr, 3)
                elif (fd == 4):
                    imgl = np.flip(imgorl, 1)
                    imgr = np.flip(imgorr, 1)
                else:
                    imgl = np.flip(imgorl, 0)
                    imgr = np.flip(imgorr, 0)
                for ds in [1, 2, 3, 4]:
                    imgl = misc.imresize(imgl, 1 / ds, 'bicubic')
                    imgr = misc.imresize(imgr, 1 / ds, 'bicubic')
                    H, W,c= imgl.shape
                    Ll = misc.imresize(imgl, 1 / self.scale, 'bicubic')
                    Bl0 = misc.imresize(Ll, [H,W,c], 'bicubic')
                    Lr = misc.imresize(imgr, 1 / self.scale, 'bicubic')
                    Br0 = misc.imresize(Lr,[H,W,c], 'bicubic')
                    imgl = imgl[:64,:,:]
                    imgr = imgr[:64,:,:]
                    Bl = Bl0[:64,:,:]
                    Br = np.zeros([H-64,W,64])
                    for nm in range(64):
                        Br[:,:,nm] = Br0[nm:H-nm,:,nm]
                    H1, W1,_ = imgl.shape
                    if H1< self.patch_size or W1< self.patch_size:
                        continue
                    hnum =H // p
                    wnum =W// p
                    for i in range(hnum):  # 每张图像有几个patch
                        for j in range(wnum):
                            h = p * i
                            w = p * j
                            gtly = imgl[h: h + p, w:w + p,0]
                            gtry = imgr[h: h + p, w:w + p,0]
                            gtlc = imgl[h: h + p, w:w + p,:]
                            gtrc = imgr[h: h + p, w:w + p,:]
                            bicly =Bl[h: h + p, w:w + p,0]
                            bicry =Br[h: h + p, w:w + p,:]
                            biclc = Bl[h: h + p, w:w + p,1:3]
                            bicrc = Br[h: h + p, w:w + p,1:3]
                            gtly = np.float32(gtly / 255.0)
                            gtry = np.float32(gtry / 255.0)
                            bicly = np.float32(bicly / 255.0)
                            bicry = np.float32(bicry / 255.0)
                            gtlc = np.float32(gtlc / 255.0)
                            gtrc = np.float32(gtrc / 255.0)
                            biclc = np.float32(biclc / 255.0)
                            bicrc = np.float32(bicrc / 255.0)
                            c1 = np.random.rand()
                            c2 = np.random.rand()
                            if c1 < 0.5:
                                gtry, gtly, bicry, bicly,gtrc, gtlc,bicrc, biclc = gtry[::-1, :], gtly[::-1, :],bicry[::-1, :,:], bicly[::-1, :],gtrc[::-1, :,:], gtlc[::-1, :,:],bicrc[::-1, :,:], biclc[::-1, :,:]
                            if c2 < 0.5:
                                gtry, gtly, bicry, bicly, gtrc, gtlc, bicrc, biclc = gtry[:, ::-1], gtly[:, ::-1],bicry[:, ::-1,:], bicly[:, ::-1], gtrc[:, ::-1,:], gtlc[:, ::-1,:],bicrc[:, ::-1,:], biclc[:, ::-1,:]
                            yield gtry, gtly, bicry, bicly,gtrc, gtlc,bicrc, biclc
    # Y RGB 分开读 CVPR18
    def read_pngcvpr(self):
        dataset = tf.data.Dataset.from_generator(self.get_generatorcvpr, (tf.float32,tf.float32,tf.float32,tf.float32,tf.float32, tf.float32,tf.float32, tf.float32))
        dataset = dataset.shuffle(self.shuffle_num).prefetch(self.prefetch_num)  #
        dataset = dataset.apply(tf.contrib.data.batch_and_drop_remainder(self.batch_size)).repeat()
        cry, cly, bry, bly,crc,clc,brc,blc = dataset.make_one_shot_iterator().get_next()
        p = self.patch_size
        cly = tf.reshape(cly, [self.batch_size, p, p, 1])
        cry = tf.reshape(cry, [self.batch_size, p, p, 1])
        bly = tf.reshape(bly, [self.batch_size, p, p, 1])
        bry = tf.reshape(bry, [self.batch_size, p, p, 64])
        clc = tf.reshape(clc, [self.batch_size, p, p, 3])
        crc = tf.reshape(crc, [self.batch_size, p, p,3])
        blc = tf.reshape(blc, [self.batch_size, p, p, 2])
        brc= tf.reshape(brc, [self.batch_size, p, p,2])
        return cry,cly,bry,bly,crc,clc,brc,blc
    def get_generatorcvpr1(self):
        img_pathsl = sorted(glob.glob(os.path.join(self.data_dirl,'*.bmp')))+ sorted(glob.glob(os.path.join(self.data_dirl,'*.jpg')))
        img_pathsr = sorted(glob.glob(os.path.join(self.data_dirr,'*.bmp')))+sorted(glob.glob(os.path.join(self.data_dirr, '*.jpg')))
        p = self.patch_size
        for i in range(len(img_pathsl)):
            imgrgbl = misc.imread(img_pathsl[i])
            imgrgbr = misc.imread(img_pathsr[i])
            imgorl = utills.rgb2ycbcr(imgrgbl)
            imgorr = utills.rgb2ycbcr(imgrgbr)
            h,w,_ = imgorl.shape
            h = h-h%self.scale
            w = w-w%self.scale
            imgorl = imgorl[:h,:w,:]
            imgorr = imgorr[:h,:w,:]
            for fd in range(0,6):
                if (fd == 0):
                    imgl = imgorl
                    imgr = imgorr
                elif (fd == 1):
                    imgl = np.rot90(imgorl, 1)
                    imgr = np.rot90(imgorr, 1)
                elif (fd == 2):
                    imgl = np.rot90(imgorl, 2)
                    imgr = np.rot90(imgorr, 2)
                elif (fd == 3):
                    imgl = np.rot90(imgorl, 3)
                    imgr = np.rot90(imgorr, 3)
                elif (fd == 4):
                    imgl = np.flip(imgorl, 1)
                    imgr = np.flip(imgorr, 1)
                else:
                    imgl = np.flip(imgorl, 0)
                    imgr = np.flip(imgorr, 0)
                for ds in [1, 2, 3, 4]:
                    imgl = misc.imresize(imgl, 1 / ds, 'bicubic')
                    imgr = misc.imresize(imgr, 1 / ds, 'bicubic')
                    H, W,c= imgl.shape
                    Ll = misc.imresize(imgl, 1 / self.scale, 'bicubic')
                    Bl0 = misc.imresize(Ll, [H,W,c], 'bicubic')
                    Lr = misc.imresize(imgr, 1 / self.scale, 'bicubic')
                    Br0 = misc.imresize(Lr,[H,W,c], 'bicubic')
                    imgl = imgl[:64,:,:]
                    imgr = imgr[:64,:,:]
                    Bl = Bl0[:64,:,:]
                    Br = np.zeros([H-64,W,64])
                    for nm in range(64):
                        Br[:,:,nm] = Br0[nm:H-nm,:,nm]
                    H1, W1,_ = imgl.shape
                    if H1< self.patch_size or W1< self.patch_size:
                        continue
                    hnum =H // p
                    wnum =W// p
                    for i in range(hnum):  # 每张图像有几个patch
                        for j in range(wnum):
                            h = p * i
                            w = p * j
                            gtly = imgl[h: h + p, w:w + p,0]
                            gtry = imgr[h: h + p, w:w + p,0]
                            gtlc = imgl[h: h + p, w:w + p,:]
                            gtrc = imgr[h: h + p, w:w + p,:]
                            bicly =Bl[h: h + p, w:w + p,0]
                            bicry =Br[h: h + p, w:w + p,:]
                            biclc = Bl[h: h + p, w:w + p,1:3]
                            bicrc = Br[h: h + p, w:w + p,1:3]
                            gtly = np.float32(gtly / 255.0)
                            gtry = np.float32(gtry / 255.0)
                            bicly = np.float32(bicly / 255.0)
                            bicry = np.float32(bicry / 255.0)
                            gtlc = np.float32(gtlc / 255.0)
                            gtrc = np.float32(gtrc / 255.0)
                            biclc = np.float32(biclc / 255.0)
                            bicrc = np.float32(bicrc / 255.0)
                            c1 = np.random.rand()
                            c2 = np.random.rand()
                            if c1 < 0.5:
                                gtry, gtly, bicry, bicly,gtrc, gtlc,bicrc, biclc = gtry[::-1, :], gtly[::-1, :],bicry[::-1, :,:], bicly[::-1, :],gtrc[::-1, :,:], gtlc[::-1, :,:],bicrc[::-1, :,:], biclc[::-1, :,:]
                            if c2 < 0.5:
                                gtry, gtly, bicry, bicly, gtrc, gtlc, bicrc, biclc = gtry[:, ::-1], gtly[:, ::-1],bicry[:, ::-1,:], bicly[:, ::-1], gtrc[:, ::-1,:], gtlc[:, ::-1,:],bicrc[:, ::-1,:], biclc[:, ::-1,:]
                            yield gtry, gtly, bicry, bicly,gtrc, gtlc,bicrc, biclc
    # Y RGB 分开读 CVPR18
    def read_pngcvpr1(self):
        dataset = tf.data.Dataset.from_generator(self.get_generatorcvpr1, (tf.float32,tf.float32,tf.float32,tf.float32,tf.float32, tf.float32,tf.float32, tf.float32))
        dataset = dataset.shuffle(self.shuffle_num).prefetch(self.prefetch_num)  #
        dataset = dataset.apply(tf.contrib.data.batch_and_drop_remainder(self.batch_size)).repeat()
        cry, cly, bry, bly,crc,clc,brc,blc = dataset.make_one_shot_iterator().get_next()
        p = self.patch_size
        cly = tf.reshape(cly, [self.batch_size, p, p, 1])
        cry = tf.reshape(cry, [self.batch_size, p, p, 1])
        bly = tf.reshape(bly, [self.batch_size, p, p, 1])
        bry = tf.reshape(bry, [self.batch_size, p, p, 64])
        clc = tf.reshape(clc, [self.batch_size, p, p, 3])
        crc = tf.reshape(crc, [self.batch_size, p, p,3])
        blc = tf.reshape(blc, [self.batch_size, p, p, 2])
        brc= tf.reshape(brc, [self.batch_size, p, p,2])
        return cry,cly,bry,bly,crc,clc,brc,blc

    # =========================== Noise ==========================
    def get_generatorRGBN(self):
        img_pathsl = sorted(glob.glob(os.path.join(self.data_dirl,'*.bmp')))+ sorted(glob.glob(os.path.join(self.data_dirl,'*.jpg')))#sorted(glob.glob(os.path.join(self.data_dirl,'*.png')))#
        img_pathsr = sorted(glob.glob(os.path.join(self.data_dirr,'*.bmp'))) + sorted(glob.glob(os.path.join(self.data_dirr, '*.jpg')))#sorted(glob.glob(os.path.join(self.data_dirr,'*.png')))#
        p = self.patch_size
        for i in range(len(img_pathsl)):
            imgorl = np.array(misc.imread(img_pathsl[i]))
            imgorr = np.array(misc.imread(img_pathsr[i]))
            h,w,_ = imgorl.shape
            imgorl = imgorl[:h,:w,:3]
            imgorr = imgorr[:h,:w,:3]
            for fd in range(6):
                if (fd == 0):
                    imgl = imgorl
                    imgr = imgorr
                elif (fd == 1):
                    imgl = np.rot90(imgorl, 1)
                    imgr = np.rot90(imgorr, 1)
                elif (fd == 2):
                    imgl = np.rot90(imgorl, 2)
                    imgr = np.rot90(imgorr, 2)
                elif (fd == 3):
                    imgl = np.rot90(imgorl, 3)
                    imgr = np.rot90(imgorr, 3)
                elif (fd == 4):
                    imgl = np.flip(imgorl, 1)
                    imgr = np.flip(imgorr, 1)
                else:
                    imgl = np.flip(imgorl, 0)
                    imgr = np.flip(imgorr, 0)
                for ds in [1, 2, 3, 4]: # ]:#
                    imgl0 = misc.imresize(imgl, 1 / ds, 'bicubic')
                    imgr0 = misc.imresize(imgr, 1 / ds, 'bicubic')
                    H, W,c= imgl0.shape

                    nlevel = random.randint(0, 30) # 10 #
                    noisemapyl = (nlevel) * np.random.randn(H, W, c)
                    noisemapyr = noisemapyl #(nlevel) * np.random.randn(H, W, c) #
                    imgl = np.float32(imgl0 / 255 + noisemapyl / 255)
                    imgr = np.float32(imgr0 / 255 + noisemapyr / 255)

                    Ll = imgl
                    Lr = imgr
                    if H< self.patch_size or W< self.patch_size:
                        continue
                    hnum =H // p
                    wnum =W// p
                    for i in range(hnum):  # 每张图像有几个patch
                        for j in range(wnum):
                            h = p * i
                            w = p * j
                            gtl = imgl0[h: h + p, w:w + p,:]
                            gtr = imgr0[h: h + p, w:w + p,:]
                            bicl =Ll[h: h + p, w:w + p,:]
                            bicr =Lr[h: h + p, w:w + p,:]
                            gtl = np.float32(gtl / 255.0)
                            gtr = np.float32(gtr / 255.0)
                            c1 = np.random.rand()
                            c2 = np.random.rand()
                            if c1 < 0.5:
                                gtr, gtl,bicr, bicl= gtr[::-1, :,:], gtl[::-1, :,:],bicr[::-1,:, :], bicl[::-1, :,:]
                            if c2 < 0.5:
                                gtr, gtl,bicr, bicl = gtr[:, ::-1,:], gtl[:, ::-1,:],bicr[:, ::-1,:], bicl[:, ::-1,:]
                            yield gtr, gtl,bicr, bicl
    def read_pngsRGBN(self):
        dataset = tf.data.Dataset.from_generator(self.get_generatorRGBN, (tf.float32,tf.float32,tf.float32,tf.float32)) # 一个quality
        dataset = dataset.shuffle(self.shuffle_num).prefetch(self.prefetch_num)  #
        dataset = dataset.apply(tf.contrib.data.batch_and_drop_remainder(self.batch_size)).repeat()
        #dataset = dataset.shuffle(self.shuffle_num).prefetch(self.prefetch_num).batch(self.batch_size,drop_remainder=True).repeat()
        cr,cl,nr,nl = dataset.make_one_shot_iterator().get_next() # 名字不能和generator中一样
        p = self.patch_size
        cl = tf.reshape(cl, [self.batch_size, p, p, 3])
        cr = tf.reshape(cr, [self.batch_size, p, p,3])
        nl = tf.reshape(nl, [self.batch_size, p, p,3])
        nr = tf.reshape(nr, [self.batch_size, p, p,3])
        return cr,cl,nr,nl
     # read single image(左右图混合一起)
    def get_generatorSrgbN(self):
        img_paths = sorted(glob.glob(os.path.join(self.data_dirl, '*.bmp'))) + sorted(
            glob.glob(os.path.join(self.data_dirl, '*.jpg'))) \
                    + sorted(glob.glob(os.path.join(self.data_dirr, '*.bmp'))) + sorted(
            glob.glob(os.path.join(self.data_dirr, '*.jpg')))
        p = self.patch_size
        lp = p // self.scale
        for i in range(len(img_paths)):
            imgrgbl = misc.imread(img_paths[i])
            imgorl = np.array(imgrgbl)
            h, w, c = imgorl.shape
            h = h - h % self.scale
            w = w - w % self.scale
            imgorl = imgorl[:h, :w, :3]
            for fd in range(6):
                if (fd == 0):
                    imgl = imgorl
                elif (fd == 1):
                    imgl = np.rot90(imgorl, 1)
                elif (fd == 2):
                    imgl = np.rot90(imgorl, 2)
                elif (fd == 3):
                    imgl = np.rot90(imgorl, 3)
                elif (fd == 4):
                    imgl = np.flip(imgorl, 1)
                else:
                    imgl = np.flip(imgorl, 0)
                for ds in [1, 2, 3, 4]:
                    imgl = misc.imresize(imgl, 1 / ds, 'bicubic')
                    H, W, c = imgl.shape

                    nlevel = random.randint(0, 30)
                    noisemapyl = (nlevel) * np.random.randn(H, W, c)
                    imgln = np.float32(imgl / 255 + noisemapyl / 255)
                    Ll = imgln
                    if H < self.patch_size or W < self.patch_size:
                        continue
                    hnum = H // p
                    wnum = W // p
                    imgl = np.float32(imgl / 255.0)
                    Ll = Ll
                    for i in range(hnum):  # 每张图像有几个patch
                        for j in range(wnum):
                            h = p * i
                            w = p * j
                            gtl = imgl[h: h + p, w:w + p, :]
                            nl = Ll[h: h + p, w:w + p, :]
                            c1 = np.random.rand()
                            c2 = np.random.rand()
                            if c1 < 0.5:
                                gtl, nl = gtl[::-1, :, :], nl[::-1, :, :]
                            if c2 < 0.5:
                                gtl, nl = gtl[:, ::-1, :], nl[:, ::-1, :]
                            yield gtl, nl
    def read_pngsSrgbN(self):
        dataset = tf.data.Dataset.from_generator(self.get_generatorSrgbN, (tf.float32, tf.float32)) # 一个quality
        dataset = dataset.shuffle(self.shuffle_num).prefetch(self.prefetch_num)  #
        dataset = dataset.apply(tf.contrib.data.batch_and_drop_remainder(self.batch_size)).repeat()
        c,n = dataset.make_one_shot_iterator().get_next() # 名字不能和generator中一样
        p = self.patch_size
        lp = p // self.scale
        c = tf.reshape(c, [self.batch_size, p, p,3])
        n = tf.reshape(n, [self.batch_size, p, p,3])
        return c,n
    # =========================== Blur ==========================
    def gaussian_kernel_2d_opencv(self, sigma=0.0):
        kx = cv2.getGaussianKernel(15, sigma)
        ky = cv2.getGaussianKernel(15, sigma)
        return np.multiply(kx, np.transpose(ky))
    def get_generatorRGBb(self):
        img_pathsl = sorted(glob.glob(os.path.join(self.data_dirl,'*.bmp')))+ sorted(glob.glob(os.path.join(self.data_dirl,'*.jpg')))#sorted(glob.glob(os.path.join(self.data_dirl,'*.png')))#
        img_pathsr = sorted(glob.glob(os.path.join(self.data_dirr,'*.bmp'))) + sorted(glob.glob(os.path.join(self.data_dirr, '*.jpg')))#sorted(glob.glob(os.path.join(self.data_dirr,'*.png')))#
        p = self.patch_size
        for i in range(len(img_pathsl)):
            imgorl = np.array(misc.imread(img_pathsl[i]))
            imgorr = np.array(misc.imread(img_pathsr[i]))
            h,w,_ = imgorl.shape
            imgorl = imgorl[:h,:w,:3]
            imgorr = imgorr[:h,:w,:3]
            for fd in range(6):
                if (fd == 0):
                    imgl = imgorl
                    imgr = imgorr
                elif (fd == 1):
                    imgl = np.rot90(imgorl, 1)
                    imgr = np.rot90(imgorr, 1)
                elif (fd == 2):
                    imgl = np.rot90(imgorl, 2)
                    imgr = np.rot90(imgorr, 2)
                elif (fd == 3):
                    imgl = np.rot90(imgorl, 3)
                    imgr = np.rot90(imgorr, 3)
                elif (fd == 4):
                    imgl = np.flip(imgorl, 1)
                    imgr = np.flip(imgorr, 1)
                else:
                    imgl = np.flip(imgorl, 0)
                    imgr = np.flip(imgorr, 0)
                for ds in [1, 2, 3, 4]: # ]:#
                    imgl0 = misc.imresize(imgl, 1 / ds, 'bicubic')
                    imgr0 = misc.imresize(imgr, 1 / ds, 'bicubic')
                    H, W,c= imgl0.shape

                    a = random.randint(2, 40) / 10
                    kernel = self.gaussian_kernel_2d_opencv(a)
                    imblurl = cv2.filter2D(imgl0, -1, kernel)
                    Ll = np.float32(imblurl / 255.0)
                    imblurr = cv2.filter2D(imgr0, -1, kernel)
                    Lr = np.float32(imblurr / 255.0)

                    if H< self.patch_size or W< self.patch_size:
                        continue
                    hnum =H // p
                    wnum =W// p
                    for i in range(hnum):  # 每张图像有几个patch
                        for j in range(wnum):
                            h = p * i
                            w = p * j
                            gtl = imgl0[h: h + p, w:w + p,:]
                            gtr = imgr0[h: h + p, w:w + p,:]
                            bicl =Ll[h: h + p, w:w + p,:]
                            bicr =Lr[h: h + p, w:w + p,:]
                            gtl = np.float32(gtl / 255.0)
                            gtr = np.float32(gtr / 255.0)
                            c1 = np.random.rand()
                            c2 = np.random.rand()
                            if c1 < 0.5:
                                gtr, gtl,bicr, bicl= gtr[::-1, :,:], gtl[::-1, :,:],bicr[::-1,:, :], bicl[::-1, :,:]
                            if c2 < 0.5:
                                gtr, gtl,bicr, bicl = gtr[:, ::-1,:], gtl[:, ::-1,:],bicr[:, ::-1,:], bicl[:, ::-1,:]
                            yield gtr, gtl,bicr, bicl
    def read_pngsRGBb(self):
        dataset = tf.data.Dataset.from_generator(self.get_generatorRGBb, (tf.float32,tf.float32,tf.float32,tf.float32)) # 一个quality
        dataset = dataset.shuffle(self.shuffle_num).prefetch(self.prefetch_num)  #
        dataset = dataset.apply(tf.contrib.data.batch_and_drop_remainder(self.batch_size)).repeat()
        #dataset = dataset.shuffle(self.shuffle_num).prefetch(self.prefetch_num).batch(self.batch_size,drop_remainder=True).repeat()
        cr,cl,nr,nl = dataset.make_one_shot_iterator().get_next() # 名字不能和generator中一样
        p = self.patch_size
        cl = tf.reshape(cl, [self.batch_size, p, p, 3])
        cr = tf.reshape(cr, [self.batch_size, p, p,3])
        nl = tf.reshape(nl, [self.batch_size, p, p,3])
        nr = tf.reshape(nr, [self.batch_size, p, p,3])
        return cr,cl,nr,nl
     # read single image(左右图混合一起)
    def get_generatorSrgbb(self):
        img_paths = sorted(glob.glob(os.path.join(self.data_dirl, '*.bmp'))) + sorted(
            glob.glob(os.path.join(self.data_dirl, '*.jpg'))) \
                    + sorted(glob.glob(os.path.join(self.data_dirr, '*.bmp'))) + sorted(
            glob.glob(os.path.join(self.data_dirr, '*.jpg')))
        p = self.patch_size
        for i in range(len(img_paths)):
            imgrgbl = misc.imread(img_paths[i])
            imgorl = np.array(imgrgbl)
            h, w, c = imgorl.shape
            h = h - h % self.scale
            w = w - w % self.scale
            imgorl = imgorl[:h, :w, :3]
            for fd in range(6):
                if (fd == 0):
                    imgl = imgorl
                elif (fd == 1):
                    imgl = np.rot90(imgorl, 1)
                elif (fd == 2):
                    imgl = np.rot90(imgorl, 2)
                elif (fd == 3):
                    imgl = np.rot90(imgorl, 3)
                elif (fd == 4):
                    imgl = np.flip(imgorl, 1)
                else:
                    imgl = np.flip(imgorl, 0)
                for ds in [1, 2, 3, 4]:
                    imgl = misc.imresize(imgl, 1 / ds, 'bicubic')
                    H, W, c = imgl.shape

                    a = random.randint(2, 40) / 10
                    kernel = self.gaussian_kernel_2d_opencv(a)
                    imblur = cv2.filter2D(imgl, -1, kernel)
                    Ll = np.float32(imblur/255.0)

                    if H < self.patch_size or W < self.patch_size:
                        continue
                    hnum = H // p
                    wnum = W // p
                    imgl = np.float32(imgl / 255.0)
                    Ll = Ll
                    for i in range(hnum):  # 每张图像有几个patch
                        for j in range(wnum):
                            h = p * i
                            w = p * j
                            gtl = imgl[h: h + p, w:w + p, :]
                            nl = Ll[h: h + p, w:w + p, :]
                            c1 = np.random.rand()
                            c2 = np.random.rand()
                            if c1 < 0.5:
                                gtl, nl = gtl[::-1, :, :], nl[::-1, :, :]
                            if c2 < 0.5:
                                gtl, nl = gtl[:, ::-1, :], nl[:, ::-1, :]
                            yield gtl, nl
    def read_pngsSrgbb(self):
        dataset = tf.data.Dataset.from_generator(self.get_generatorSrgbb, (tf.float32, tf.float32)) # 一个quality
        dataset = dataset.shuffle(self.shuffle_num).prefetch(self.prefetch_num)  #
        dataset = dataset.apply(tf.contrib.data.batch_and_drop_remainder(self.batch_size)).repeat()
        c,n = dataset.make_one_shot_iterator().get_next() # 名字不能和generator中一样
        p = self.patch_size
        lp = p // self.scale
        c = tf.reshape(c, [self.batch_size, p, p,3])
        n = tf.reshape(n, [self.batch_size, p, p,3])
        return c,n

class DataLoaderTsukubaY():
    def __init__(self, config):
        self.data_dirl1 = config.data_dirl
        self.data_dirr1 = config.data_dirr
        #self.data_dirl1 = "./Middlebury/train/halfsize/left/"
        #self.data_dirr1 = "./Middlebury/train/halfsize/right/"
        self.data_dirl2 = "./Middlebury/train/daylight\left/"#
        self.data_dirr2 = "./Middlebury/train/daylight/right/"#
        self.patch_size = config.label_size
        self.scale = config.scale
        self.batch_size = config.batch_size
        self.shuffle_num = config.shuffle_num
        self.prefetch_num = config.prefetch_num
        self.map_parallel_num = config.map_parallel_num

    def get_generator(self):
        img_pathsl1 = sorted(glob.glob(os.path.join(self.data_dirl1,'*.bmp')))+sorted(glob.glob(os.path.join(self.data_dirl1,'*.jpg')))
        img_pathsr1 = sorted(glob.glob(os.path.join(self.data_dirr1,'*.bmp')))+ sorted(glob.glob(os.path.join(self.data_dirr1, '*.jpg')))
        img_pathsr2 = sorted(glob.glob(os.path.join(self.data_dirr2, '*.png')))
        img_pathsl2 = sorted(glob.glob(os.path.join(self.data_dirl2, '*.png')))
        img_pathsr = img_pathsr1 + img_pathsr2
        img_pathsl = img_pathsl1 + img_pathsl2
        p = self.patch_size
        lp = p//self.scale
        for i in range(len(img_pathsl)):
            imgrgbl = np.array(misc.imread(img_pathsl[i]))[:, :, 0:3]
            imgrgbr = np.array(misc.imread(img_pathsr[i]))[:, :, 0:3]
            imgorl = utills.rgb2y(imgrgbl)[:, :]
            imgorr = utills.rgb2y(imgrgbr)[:, :]
            h,w = imgorl.shape
            h = h-h%self.scale
            w = w-w%self.scale
            imgorl = imgorl[:h,:w]
            imgorr = imgorr[:h,:w]
            for fd in range(6):
                if (fd == 0):
                    imgl = imgorl
                    imgr = imgorr
                elif (fd == 1):
                    imgl = np.rot90(imgorl, 1)
                    imgr = np.rot90(imgorr, 1)
                elif (fd == 2):
                    imgl = np.rot90(imgorl, 2)
                    imgr = np.rot90(imgorr, 2)
                elif (fd == 3):
                    imgl = np.rot90(imgorl, 3)
                    imgr = np.rot90(imgorr, 3)
                elif (fd == 4):
                    imgl = np.flip(imgorl, 1)
                    imgr = np.flip(imgorr, 1)
                else:
                    imgl = np.flip(imgorl, 0)
                    imgr = np.flip(imgorr, 0)
                for ds in [1, 2, 3, 4]:
                    imgl = misc.imresize(imgl, 1 / ds, 'bicubic')
                    imgr = misc.imresize(imgr, 1 / ds, 'bicubic')
                    H, W= imgl.shape
                    Ll = misc.imresize(imgl, 1 / self.scale, 'bicubic')
                    Bl = misc.imresize(Ll, [H,W], 'bicubic')
                    Lr = misc.imresize(imgr, 1 / self.scale, 'bicubic')
                    Br = misc.imresize(Lr,[H,W], 'bicubic')
                    if H< self.patch_size or W< self.patch_size:
                        continue
                    hnum =H // p
                    wnum =W// p
                    for i in range(hnum+1):  # 每张图像有几个patch
                        for j in range(wnum+1):
                            h = p * i
                            hl = lp*i
                            wl = lp*j
                            w = p * j
                            gtl = imgl[h: h + p, w:w + p]
                            gtr = imgr[h: h + p, w:w + p]
                            ll = Ll[hl: hl + lp, wl:wl + lp]
                            lr = Lr[hl: hl + lp, wl:wl + lp]
                            bicl =Bl[h: h + p, w:w + p]
                            bicr =Br[h: h + p, w:w + p]
                            if i==hnum and j !=wnum:
                                gtl = imgl[h-p: h, w:w + p]
                                gtr = imgr[h-p: h, w:w + p]
                                ll = Ll[hl-lp: hl, wl:wl + lp]  # misc.imresize(gtl, 1 / self.scale, 'bicubic')#
                                lr = Lr[hl-lp: hl, wl:wl + lp]  # misc.imresize(gtr, 1 / self.scale, 'bicubic')#
                                bicl = Bl[h-p: h, w:w + p]  # misc.imresize(ll, [p,p], 'bicubic')#
                                bicr = Br[h-p: h, w:w + p]  # misc.imresize(lr, [p,p], 'bicubic')#
                            elif i!=hnum and j ==wnum:
                                gtl = imgl[h: h + p, w-p:w]
                                gtr = imgr[h: h + p, w-p:w]
                                ll = Ll[hl: hl + lp, wl-lp:wl]
                                lr = Lr[hl: hl + lp, wl-lp:wl]
                                bicl = Bl[h: h + p, w-p:w]
                                bicr = Br[h: h + p, w-p:w]
                            elif i == hnum and j == wnum:
                                gtl = imgl[h-p: h, w - p:w]
                                gtr = imgr[h-p: h, w - p:w]
                                ll = Ll[hl-lp: hl, wl - lp:wl]
                                lr = Lr[hl-lp: hl, wl - lp:wl]
                                bicl = Bl[h-p: h, w - p:w]
                                bicr = Br[h-p: h, w - p:w]
                            gtl = np.float32(gtl / 255.0)
                            gtr = np.float32(gtr / 255.0)
                            bicl = np.float32(bicl / 255.0)
                            bicr = np.float32(bicr / 255.0)
                            lr = np.float32(lr / 255.0)
                            ll = np.float32(ll / 255.0)
                            c1 = np.random.rand()
                            c2 = np.random.rand()
                            if c1 < 0.5:
                                gtr, gtl,bicr, bicl,lr, ll= gtr[::-1, :], gtl[::-1, :],bicr[::-1, :], bicl[::-1, :],lr[::-1, :], ll[::-1, :]
                            if c2 < 0.5:
                                gtr, gtl,bicr, bicl,lr, ll = gtr[:, ::-1], gtl[:, ::-1],bicr[:, ::-1], bicl[:, ::-1],lr[:, ::-1], ll[:, ::-1]
                            yield gtr, gtl,bicr, bicl,lr, ll

    def read_pngs(self):
        dataset = tf.data.Dataset.from_generator(self.get_generator, (tf.float32,tf.float32,tf.float32,tf.float32,tf.float32, tf.float32)) # 一个quality
        dataset = dataset.shuffle(self.shuffle_num).prefetch(self.prefetch_num)  #
        dataset = dataset.apply(tf.contrib.data.batch_and_drop_remainder(self.batch_size)).repeat()
        cr,cl,br,bl,lrs,lls = dataset.make_one_shot_iterator().get_next() # 名字不能和generator中一样
        p = self.patch_size
        lp = p // self.scale
        lrs = tf.reshape(lrs, [self.batch_size, lp, lp, 1])
        lls = tf.reshape(lls, [self.batch_size, lp, lp, 1])
        cl = tf.reshape(cl, [self.batch_size, p, p, 1])
        cr = tf.reshape(cr, [self.batch_size, p, p, 1])
        bl = tf.reshape(bl, [self.batch_size, p, p, 1])
        br = tf.reshape(br, [self.batch_size, p, p, 1])
        return lrs,lls,cr,cl,br,bl
class DataLoaderTsukuba():
    def __init__(self, config):
        self.data_dirl1 = "./Middlebury/train/halfsize/left/"
        self.data_dirr1 = "./Middlebury/train/halfsize/right/"
        self.data_dirl2 = "./Middlebury/train/daylight\left/"
        self.data_dirr2 = "./Middlebury/train/daylight/right/"#
        self.patch_size = config.label_size
        self.scale = config.scale
        self.batch_size = config.batch_size
        self.shuffle_num = config.shuffle_num
        self.prefetch_num = config.prefetch_num
        self.map_parallel_num = config.map_parallel_num

    def get_generator(self):
        img_pathsl1 = sorted(glob.glob(os.path.join(self.data_dirl1,'*.bmp')))+sorted(glob.glob(os.path.join(self.data_dirl1,'*.jpg')))
        img_pathsr1 = sorted(glob.glob(os.path.join(self.data_dirr1,'*.bmp')))+ sorted(glob.glob(os.path.join(self.data_dirr1, '*.jpg')))
        img_pathsr2 = sorted(glob.glob(os.path.join(self.data_dirr2, '*.png')))
        img_pathsl2 = sorted(glob.glob(os.path.join(self.data_dirl2, '*.png')))
        img_pathsr = img_pathsr1 + img_pathsr2
        img_pathsl = img_pathsl1 + img_pathsl2
        p = self.patch_size
        lp = p//self.scale
        for i in range(len(img_pathsl)):
            imgrgbl = np.array(misc.imread(img_pathsl[i]))[:, :, 0:3]
            imgrgbr = np.array(misc.imread(img_pathsr[i]))[:, :, 0:3]
            imgorl = imgrgbl
            imgorr = imgrgbr
            h,w,c = imgorl.shape
            h = h-h%self.scale
            w = w-w%self.scale
            imgorl = imgorl[:h,:w,:]
            imgorr = imgorr[:h,:w,:]
            for fd in range(6):
                if (fd == 0):
                    imgl = imgorl
                    imgr = imgorr
                elif (fd == 1):
                    imgl = np.rot90(imgorl, 1)
                    imgr = np.rot90(imgorr, 1)
                elif (fd == 2):
                    imgl = np.rot90(imgorl, 2)
                    imgr = np.rot90(imgorr, 2)
                elif (fd == 3):
                    imgl = np.rot90(imgorl, 3)
                    imgr = np.rot90(imgorr, 3)
                elif (fd == 4):
                    imgl = np.flip(imgorl, 1)
                    imgr = np.flip(imgorr, 1)
                else:
                    imgl = np.flip(imgorl, 0)
                    imgr = np.flip(imgorr, 0)
                for ds in [1, 2, 3, 4]:
                    imgl = misc.imresize(imgl, 1 / ds, 'bicubic')
                    imgr = misc.imresize(imgr, 1 / ds, 'bicubic')
                    H, W,c= imgl.shape
                    Ll = misc.imresize(imgl, 1 / self.scale, 'bicubic')
                    Bl = misc.imresize(Ll, [H,W,c], 'bicubic')
                    Lr = misc.imresize(imgr, 1 / self.scale, 'bicubic')
                    Br = misc.imresize(Lr,[H,W,c], 'bicubic')
                    if H< self.patch_size or W< self.patch_size:
                        continue
                    hnum =H // p
                    wnum =W// p
                    for i in range(hnum+1):  # 每张图像有几个patch
                        for j in range(wnum+1):
                            h = p * i
                            hl = lp*i
                            wl = lp*j
                            w = p * j
                            gtl = imgl[h: h + p, w:w + p,:]
                            gtr = imgr[h: h + p, w:w + p,:]
                            ll = Ll[hl: hl + lp, wl:wl + lp,:]
                            lr = Lr[hl: hl + lp, wl:wl + lp,:]
                            bicl =Bl[h: h + p, w:w + p,:]
                            bicr =Br[h: h + p, w:w + p,:]
                            if i==hnum and j !=wnum:
                                gtl = imgl[h-p: h, w:w + p,:]
                                gtr = imgr[h-p: h, w:w + p,:]
                                ll = Ll[hl-lp: hl, wl:wl + lp,:]  # misc.imresize(gtl, 1 / self.scale, 'bicubic')#
                                lr = Lr[hl-lp: hl, wl:wl + lp,:]  # misc.imresize(gtr, 1 / self.scale, 'bicubic')#
                                bicl = Bl[h-p: h, w:w + p,:]  # misc.imresize(ll, [p,p], 'bicubic')#
                                bicr = Br[h-p: h, w:w + p,:]  # misc.imresize(lr, [p,p], 'bicubic')#
                            elif i!=hnum and j ==wnum:
                                gtl = imgl[h: h + p, w-p:w,:]
                                gtr = imgr[h: h + p, w-p:w,:]
                                ll = Ll[hl: hl + lp, wl-lp:wl,:]
                                lr = Lr[hl: hl + lp, wl-lp:wl,:]
                                bicl = Bl[h: h + p, w-p:w,:]
                                bicr = Br[h: h + p, w-p:w,:]
                            elif i == hnum and j == wnum:
                                gtl = imgl[h-p: h, w - p:w,:]
                                gtr = imgr[h-p: h, w - p:w,:]
                                ll = Ll[hl-lp: hl, wl - lp:wl,:]
                                lr = Lr[hl-lp: hl, wl - lp:wl,:]
                                bicl = Bl[h-p: h, w - p:w,:]
                                bicr = Br[h-p: h, w - p:w,:]
                            gtl = np.float32(gtl / 255.0)
                            gtr = np.float32(gtr / 255.0)
                            bicl = np.float32(bicl / 255.0)
                            bicr = np.float32(bicr / 255.0)
                            lr = np.float32(lr / 255.0)
                            ll = np.float32(ll / 255.0)
                            c1 = np.random.rand()
                            c2 = np.random.rand()
                            if c1 < 0.5:
                                gtr, gtl,bicr, bicl,lr, ll= gtr[::-1, :,:], gtl[::-1, :,:],bicr[::-1, :,:], bicl[::-1, :,:],lr[::-1, :,:], ll[::-1, :,:]
                            if c2 < 0.5:
                                gtr, gtl,bicr, bicl,lr, ll = gtr[:, ::-1,:], gtl[:, ::-1,:],bicr[:, ::-1,:], bicl[:, ::-1,:],lr[:, ::-1,:], ll[:, ::-1,:]
                            yield gtr, gtl,bicr, bicl,lr, ll

    def read_pngs(self):
        dataset = tf.data.Dataset.from_generator(self.get_generator, (tf.float32,tf.float32,tf.float32,tf.float32,tf.float32, tf.float32)) # 一个quality
        dataset = dataset.shuffle(self.shuffle_num).prefetch(self.prefetch_num)  #
        dataset = dataset.apply(tf.contrib.data.batch_and_drop_remainder(self.batch_size)).repeat()
        cr,cl,br,bl,lrs,lls = dataset.make_one_shot_iterator().get_next() # 名字不能和generator中一样
        p = self.patch_size
        lp = p // self.scale
        lrs = tf.reshape(lrs, [self.batch_size, lp, lp,3])
        lls = tf.reshape(lls, [self.batch_size, lp, lp,3])
        cl = tf.reshape(cl, [self.batch_size, p, p,3])
        cr = tf.reshape(cr, [self.batch_size, p, p,3])
        bl = tf.reshape(bl, [self.batch_size, p, p,3])
        br = tf.reshape(br, [self.batch_size, p, p,3])
        return lrs,lls,cr,cl,br,bl
class DataLoaderdisp():
    def __init__(self,l,r,d,config,md = 0):
        self.data_dirl = l
        self.data_dirr = r
        self.data_dird = d
        self.patch_size = config.label_size
        self.scale = config.scale
        self.batch_size = config.batch_size
        self.shuffle_num = config.shuffle_num
        self.prefetch_num = config.prefetch_num
        self.map_parallel_num = config.map_parallel_num
        self.md = md
        self.lp = config.data_dirl
        self.rp = config.data_dirr
        self.dp = config.data_dird

    def get_generatorRGB(self):
        img_pathsl = self.data_dirl
        img_pathsr = self.data_dirr
        img_pathsd = self.data_dird
        p = self.patch_size
        lp = p//self.scale
        if self.md ==0:
            for i in range(len(img_pathsl)):
                imgorl = np.array(misc.imread(img_pathsl[i]))[:, :, 0:3]
                imgorr = np.array(misc.imread(img_pathsr[i]))[:, :, 0:3]
                imgord, _ = readPFM(img_pathsd[i])
                h, w, c = imgorl.shape
                h = h - h % self.scale
                w = w - w % self.scale
                imgd0 = imgord[:h, :w]
                imgl0 = imgorl[:h, :w, :3]
                imgr0 = imgorr[:h, :w, :3]
                for ds in [1, 2, 3, 4]:
                    imgl = misc.imresize(imgl0, 1 / ds, 'bicubic')
                    imgr = misc.imresize(imgr0, 1 / ds, 'bicubic')
                    H, W, c = imgl.shape

                    imgd = np.resize(imgd0, [H, W]) # * (1 / ds)  # 放到原图大小
                    Hd, Wd = imgd.shape
                    zl = np.zeros([H // self.scale, W // self.scale, 2], dtype=np.float32)
                    zh = np.zeros([H, W, 2], dtype=np.float32)
                    zb = np.zeros([H, W, 2], dtype=np.float32)
                    if H != Hd or Wd != W:
                        print(img_pathsl[i])
                        print(img_pathsd[i])
                        print(H, Hd, W, Wd)
                        print(i, ds, '?????????????????????????????????')
                    Ld = np.resize(imgd, [H // self.scale, W // self.scale])# * (1 / self.scale)  #
                    Bd = np.resize(imgd, [H, W])# * self.scale
                    zb[:, :, 0] = Bd[:, :]
                    zl[:, :, 0] = Ld[:, :]
                    zh[:, :, 0] = imgd[:, :]
                    Ll = misc.imresize(imgl, 1 / self.scale, 'bicubic')
                    Bl = misc.imresize(Ll, [H, W, c], 'bicubic')
                    Lr = misc.imresize(imgr, 1 / self.scale, 'bicubic')
                    Br = misc.imresize(Lr, [H, W, c], 'bicubic')
                    if H < self.patch_size or W < self.patch_size:
                        continue
                    hnum = H // p
                    wnum = W // p
                    for i in range(hnum):  # 每张图像有几个patch
                        for j in range(wnum):
                            h = p * i
                            hl = lp * i
                            wl = lp * j
                            w = p * j
                            gtd = np.float32(imgd[h: h + p, w:w + p])
                            gtdf = np.float32(zh[h: h + p, w:w + p, :])# flow disp
                            gtl = np.float32(imgl[h: h + p, w:w + p, :] / 255.0)
                            gtr = np.float32(imgr[h: h + p, w:w + p, :] / 255.0)

                            ld = np.float32(Ld[hl: hl + lp, wl:wl + lp])
                            ldf = np.float32(zl[hl: hl + lp, wl:wl + lp, :])
                            ll = np.float32(Ll[hl: hl + lp, wl:wl + lp, :] / 255.0)
                            lr = np.float32(Lr[hl: hl + lp, wl:wl + lp, :] / 255.0)

                            bicd = np.float32(Bd[h: h + p, w:w + p])
                            bicdf = np.float32(zb[h: h + p, w:w + p, :])
                            bicl = np.float32(Bl[h: h + p, w:w + p, :] / 255.0)
                            bicr = np.float32(Br[h: h + p, w:w + p, :] / 255.0)
                            c1 = np.random.rand()
                            c2 = np.random.rand()
                            if c1 < 0.5:
                                gtd, ld, bicd = gtd[::-1, :], ld[::-1, :], bicd[::-1, :]
                                gtdf, ldf, bicdf = gtdf[::-1, :, :], ldf[::-1, :, :], bicdf[::-1, :, :]
                                gtr, gtl, bicr, bicl, lr, ll = gtr[::-1, :, :], gtl[::-1, :, :], bicr[::-1, :, :], bicl[
                                                                                                                   ::-1,
                                                                                                                   :,
                                                                                                                   :], lr[
                                                                                                                       ::-1,
                                                                                                                       :,
                                                                                                                       :], ll[
                                                                                                                           ::-1,
                                                                                                                           :,
                                                                                                                           :]
                            if c2 < 0.5:
                                gtd, ld, bicd = gtd[:, ::-1], ld[:, ::-1], bicd[:, ::-1]
                                gtdf, ldf, bicdf = gtdf[:, ::-1, :], ldf[:, ::-1, :], bicdf[:, ::-1, :]
                                gtr, gtl, bicr, bicl, lr, ll = gtr[:, ::-1, :], gtl[:, ::-1, :], bicr[:, ::-1, :], bicl[
                                                                                                                   :,
                                                                                                                   ::-1,
                                                                                                                   :], lr[
                                                                                                                       :,
                                                                                                                       ::-1,
                                                                                                                       :], ll[
                                                                                                                           :,
                                                                                                                           ::-1,
                                                                                                                           :]
                            yield gtr, gtl, bicr, bicl, lr, ll, gtd, ld, bicd, gtdf, ldf, bicdf
        else: # disp 归一化？
            for i in range(len(img_pathsr)):
                name = img_pathsr[i][len(self.rp):-4]
                # print(name)
                imgorl = np.array(misc.imread(img_pathsl[i]))[:, :, 0:3]
                imgorr = np.array(misc.imread(img_pathsr[i]))[:, :, 0:3]
                if self.dp+name+'_left.png' in img_pathsd:
                    imgord = np.array(Image.open(self.dp + name + '_left.png'), np.float32)  #
                # elif self.dp+name+'.pfm' in img_pathsd:
                #     imgord = readPFM(self.dp + name + '.pfm')[0]/255.0
                else:
                    imgord = np.array(Image.open(self.dp + name + '_left.jpg'), np.float32)  #
                # print(img_pathsl[i])
                # print(self.dp + name + '.png')  # img_pathsd[i])
                h, w, c = imgorl.shape
                hd, wd = imgord.shape
                if h != hd or wd != w:
                    print(img_pathsl[i])
                    print(self.dp+name+'.png') #img_pathsd[i])
                    print(h, hd, w, wd)
                    h = min(h,hd)
                    w = min(w,wd)
                h = h - h % self.scale
                w = w - w % self.scale
                imgd0 = imgord[:h, :w]
                imgl0 = imgorl[:h, :w, :3]
                imgr0 = imgorr[:h, :w, :3]
                for ds in [1]: #[1, 2, 3, 4]:
                    imgl = misc.imresize(imgl0, 1 / ds, 'bicubic')
                    imgr = misc.imresize(imgr0, 1 / ds, 'bicubic')
                    H, W, c = imgl.shape

                    imgd = np.resize(imgd0, [H, W])# * (1 / ds)  # 放到原图大小
                    Hd, Wd = imgd.shape
                    zl = np.zeros([H // self.scale, W // self.scale, 2], dtype=np.float32)
                    zh = np.zeros([H, W, 2], dtype=np.float32)
                    zb = np.zeros([H, W, 2], dtype=np.float32)
                    if H != Hd or Wd != W:
                        print(img_pathsl[i])
                        print(img_pathsd[i])
                        print(H, Hd, W, Wd)
                        print(i, ds, '?????????????????????????????????')
                    Ld = np.resize(imgd, [H // self.scale, W // self.scale])# * (1 / self.scale)
                    Bd = np.resize(imgd, [H, W]) #* self.scale
                    zb[:, :, 0] = Bd[:, :]
                    zl[:, :, 0] = Ld[:, :]
                    zh[:, :, 0] = imgd[:, :]
                    Ll = misc.imresize(imgl, 1 / self.scale, 'bicubic')
                    Bl = misc.imresize(Ll, [H, W, c], 'bicubic')
                    Lr = misc.imresize(imgr, 1 / self.scale, 'bicubic')
                    Br = misc.imresize(Lr, [H, W, c], 'bicubic')
                    if H < self.patch_size or W < self.patch_size:
                        continue
                    hnum = H // p
                    wnum = W // p
                    for i in range(hnum):  # 每张图像有几个patch
                        for j in range(wnum):
                            h = p * i
                            hl = lp * i
                            wl = lp * j
                            w = p * j
                            gtd = np.float32(zh[h: h + p, w:w + p, :] / 255.0)
                            gtl = np.float32(imgl[h: h + p, w:w + p, :] / 255.0)
                            gtr = np.float32(imgr[h: h + p, w:w + p, :] / 255.0)

                            ld = np.float32(zl[hl: hl + lp, wl:wl + lp, :] / 255.0)
                            ll = np.float32(Ll[hl: hl + lp, wl:wl + lp, :] / 255.0)
                            lr = np.float32(Lr[hl: hl + lp, wl:wl + lp, :] / 255.0)

                            bicd = np.float32(zb[h: h + p, w:w + p, :] / 255.0)
                            bicl = np.float32(Bl[h: h + p, w:w + p, :] / 255.0)
                            bicr = np.float32(Br[h: h + p, w:w + p, :] / 255.0)
                            c1 = np.random.rand()
                            c2 = np.random.rand()
                            if c1 < 0.5:
                                gtd, ld, bicd = gtd[::-1, :, :], ld[::-1, :, :], bicd[::-1, :, :]
                                gtr, gtl, bicr, bicl, lr, ll = gtr[::-1, :, :], gtl[::-1, :, :], bicr[::-1, :, :], bicl[
                                                                                                                   ::-1,
                                                                                                                   :,
                                                                                                                   :], lr[
                                                                                                                       ::-1,
                                                                                                                       :,
                                                                                                                       :], ll[
                                                                                                                           ::-1,
                                                                                                                           :,
                                                                                                                           :]
                            if c2 < 0.5:
                                gtd, ld, bicd = gtd[:, ::-1, :], ld[:, ::-1, :], bicd[:, ::-1, :]
                                gtr, gtl, bicr, bicl, lr, ll = gtr[:, ::-1, :], gtl[:, ::-1, :], bicr[:, ::-1, :], bicl[
                                                                                                                   :,
                                                                                                                   ::-1,
                                                                                                                   :], lr[
                                                                                                                       :,
                                                                                                                       ::-1,
                                                                                                                       :], ll[
                                                                                                                           :,
                                                                                                                           ::-1,
                                                                                                                           :]
                            yield gtr, gtl, bicr, bicl, lr, ll, gtd, ld, bicd

    def read_pngsRGB(self):
        dataset = tf.data.Dataset.from_generator(self.get_generatorRGB, (tf.float32,tf.float32,tf.float32,tf.float32,tf.float32,tf.float32,tf.float32,tf.float32,tf.float32,tf.float32,tf.float32, tf.float32)) # 一个quality
        dataset = dataset.shuffle(self.shuffle_num).prefetch(self.prefetch_num)  #
        dataset = dataset.apply(tf.contrib.data.batch_and_drop_remainder(self.batch_size)).repeat()
        #dataset = dataset.shuffle(self.shuffle_num).prefetch(self.prefetch_num).batch(self.batch_size,drop_remainder=True).repeat()
        cr,cl,br,bl,lrs,lls,dg,dl,db,dgf,dlf,dbf = dataset.make_one_shot_iterator().get_next() # 名字不能和generator中一样
        p = self.patch_size
        lp = p // self.scale#
        lrs = tf.reshape(lrs, [self.batch_size, lp, lp,3])
        lls = tf.reshape(lls, [self.batch_size, lp, lp,3])
        ldsf = tf.reshape(dlf, [self.batch_size, lp, lp,2])
        lds = tf.reshape(dl, [self.batch_size, lp, lp])
        cd = tf.reshape(dg, [self.batch_size, p, p])
        cdf = tf.reshape(dgf, [self.batch_size, p, p, 2])
        cl = tf.reshape(cl, [self.batch_size, p, p, 3])
        cr = tf.reshape(cr, [self.batch_size, p, p,3])
        bdf = tf.reshape(dbf, [self.batch_size, p, p,2])
        bd = tf.reshape(db, [self.batch_size, p, p])
        bl = tf.reshape(bl, [self.batch_size, p, p,3])
        br = tf.reshape(br, [self.batch_size, p, p,3])
        return lrs,lls,cr,cl,br,bl,lds,cd,bd
