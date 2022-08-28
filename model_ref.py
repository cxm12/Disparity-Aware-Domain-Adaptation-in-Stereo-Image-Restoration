from opts2 import *

# 基于 SRNTT RefSR 的模型 多尺度多参考的超分  用CNN代替swarp
fm_n = 64

FLAGS = tf.app.flags.FLAGS
dropout_keep_prob1 = tf.placeholder(tf.float32)

init_value2 = []
print('kkkk')


def ResBlock(h0, name, f_size=fm_n, kernel_size=3):
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        h1 = relu(conv2d(h0, f_size, kernel_size, kernel_size, name='conv1', init_value=init_value2))
        h2 = conv2d(h1, f_size, kernel_size, kernel_size, name='conv2', init_value=init_value2)
        h4 = tf.add(h2, h0)
        return h4
    
    
# =============================== local Net =========================#
def SR_localbic(l, b, fsize=128, scale=4, small=True, Y=0):  # 20 layer
    with tf.variable_scope("pret_ed", reuse=tf.AUTO_REUSE) as scope:
        x = conv2d(l, fsize, 3, 3, name='in')
        # Store the output of the first convolution to add later
        conv_1 = x
        for i in range(6):
            x = ResBlock(x, name='res' + str(i), f_size=fsize)
        x = conv2d(x, fsize, 3, 3, name='conv1')
        x += conv_1
        
        xim = lrelu(conv2d(x, fsize, 3, 3, name='convim'))
        
        xf = lrelu(conv2d(x, fsize, 3, 3, name='convf1'))
        xf1 = lrelu(conv2d(xf, fsize, 3, 3, name='convf2'))
        
        f = concat([conv_1, xim, xf1], 3)
        f = conv2d(f, fsize, 3, 3, name='convout0')
        f = f + conv_1
        # Upsample output of the convolution
        x = deconv_anysize(f, 'deconv1', fsize, ksize=8, stride=scale)
        if Y == 0:
            output = conv2d(x, 3, 3, 3, name='convout')
        else:
            output = conv2d(x, 1, 3, 3, name='convout')
        if b != None:
            output = output + b
        if small:
            return xf1, output  # small linVSR feature
        else:
            return x, output  # big feature


def SR_ctxLeftRight(sl, sr, fl, fr, f_size=128, s=4, small=False):  # 19
    with tf.variable_scope("context", reuse=tf.AUTO_REUSE) as scope:
        if small:  # spacetodepth  time??  PSNR??s`
            sl1 = tf.space_to_depth(sl, s, 'left', "NHWC")  # scale=4 blockSize=2 # 2的倍数
            sr1 = tf.space_to_depth(sr, s, 'right', "NHWC")
        else:  # left right image 交叉传递feature
            sl1 = sl
            sr1 = sr
        xl = lrelu(conv2d(sl1, f_size, 3, 3, name='in'))
        xr = lrelu(conv2d(sr1, f_size, 3, 3, name='inr'))
        cat1 = concat([xl, xr], 3)
        
        xl = lrelu(conv2d(cat1, f_size, 3, 3, name='l1'))
        xl = ResBlock(xl, name='res1', f_size=f_size)
        
        xr = lrelu(conv2d(cat1, f_size, 3, 3, name='r1'))
        xr = ResBlock(xr, name='resr1', f_size=f_size)
        cat2 = concat([xl, xr], 3)
        
        xl = lrelu(conv2d(cat2, f_size, 3, 3, name='l2'))
        xl = ResBlock(xl, name='res2', f_size=f_size)
        
        xr = lrelu(conv2d(cat2, f_size, 3, 3, name='r2'))
        xr = ResBlock(xr, name='resr2', f_size=f_size)
        cat3 = concat([xl, xr], 3)
        x = lrelu(conv2d(cat3, f_size, 3, 3, name='catconv'))
        
        if small:  # 级联image+feature
            im = conv2d(sl1, f_size, 3, 3, name='imin')
            imr = conv2d(sr1, f_size, 3, 3, name='iminr')
            im = ResBlock(im, name='imres1', f_size=f_size)
            imr = ResBlock(imr, name='imresr', f_size=f_size)
            fl = conv2d(concat([im, x, fl], 3), f_size, 3, 3, name='fin')
            fr = conv2d(concat([imr, x, fr], 3), f_size, 3, 3, name='finr')
            fl = ResBlock(fl, name='fres1', f_size=f_size)
            fr = ResBlock(fr, name='fresr', f_size=f_size)
            xl = im + fl
            xl = deconv_anysize(xl, 'deconv1', f_size, ksize=8, stride=s)
            xl = conv2d(xl, 3, 3, 3, name='convout')
            output = xl + sl
            xr = imr + fr
            xr = deconv_anysize(xr, 'deconv1r', f_size, ksize=8, stride=s)
            xr = conv2d(xr, 3, 3, 3, name='convoutr')
            outputr = xr + sr
        else:
            # im = conv2d(concat([sl,sr],3), f_size, 3, 3, name='imin')
            im = conv2d(sl1, f_size, 3, 3, name='imin')
            im = ResBlock(im, name='imres1', f_size=f_size)
            # f = conv2d(concat([x,fl,fr],3), f_size, 3, 3, name='fin')
            f = conv2d(concat([im, x, fl], 3), f_size, 3, 3, name='fin')
            f = ResBlock(f, name='fres1', f_size=f_size)
            x = im + f
            x = conv2d(x, 3, 3, 3, name='convout')
            output = x + sl
            imr = conv2d(sr1, f_size, 3, 3, name='iminr')
            imr = ResBlock(imr, name='imresr', f_size=f_size)
            f = conv2d(concat([imr, x, fr], 3), f_size, 3, 3, name='finr')  # x被left处理了！！！
            f = ResBlock(f, name='fresr', f_size=f_size)
            x = imr + f
            x = conv2d(x, 3, 3, 3, name='convoutr')
            outputr = x + sr
        return output, outputr


def SR_ctxLeft(sl, sr, fl, f_size=128, small=False, s=4):  # 22
    with tf.variable_scope("context", reuse=tf.AUTO_REUSE) as scope:
        # left right image 交叉传递feature
        if small == True:  # 利用smallfeature
            xl = lrelu(conv2d(tf.space_to_depth(sl, s, 'left', "NHWC"), f_size, 3, 3, name='in'))
            xr = lrelu(conv2d(tf.space_to_depth(sr, s, 'right', "NHWC"), f_size, 3, 3, name='inr'))
        else:
            xl = lrelu(conv2d(sl, f_size, 3, 3, name='in'))
            xr = lrelu(conv2d(sr, f_size, 3, 3, name='inr'))
        cat1 = concat([xl, xr], 3)
        
        xl = lrelu(conv2d(cat1, f_size, 3, 3, name='l1'))
        xl = ResBlock(xl, name='res1', f_size=f_size)
        
        xr = lrelu(conv2d(cat1, f_size, 3, 3, name='r1'))
        xr = ResBlock(xr, name='resr1', f_size=f_size)
        cat2 = concat([xl, xr], 3)
        
        xl = lrelu(conv2d(cat2, f_size, 3, 3, name='l2'))
        xl = ResBlock(xl, name='res2', f_size=f_size)
        
        xr = lrelu(conv2d(cat2, f_size, 3, 3, name='r2'))
        xr = ResBlock(xr, name='resr2', f_size=f_size)
        x = lrelu(conv2d(concat([xl, xr], 3), f_size, 3, 3, name='catconv'))
        
        # 级联image+feature
        if small == True:  # 利用smallfeature
            im = conv2d(tf.space_to_depth(sl, s, 'left', "NHWC"), f_size, 3, 3, name='imin')
        else:
            im = conv2d(sl, f_size, 3, 3, name='imin')  # im = conv2d(concat([sl,sr],3), f_size, 3, 3, name='imin')
        im = ResBlock(im, name='imres1', f_size=f_size)
        f = conv2d(concat([im, x, fl], 3), f_size, 3, 3,
                   name='fin')  # f = conv2d(concat([x,fl,fr],3), f_size, 3, 3, name='fin')
        f = ResBlock(f, name='fres1', f_size=f_size)
        x = im + f
        if small == True:  # 利用smallfeature
            x = deconv_anysize(x, 'deconv1', f_size, ksize=8, stride=s)
            x = conv2d(x, 3, 3, 3, name='convout')
        else:
            x = conv2d(x, 3, 3, 3, name='convout')
        output = x + sl
        return output
