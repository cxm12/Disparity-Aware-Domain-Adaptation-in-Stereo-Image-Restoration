from PIL import Image
from scipy import misc
from skimage import measure
from utills import *
from model_ref import *

scale = 4
method = 'local_Le'
# 'SR_localEDSR'# 'SR_localEDSR_kt'#'EDSR' #'local_ds_Le'#'PWC_ftlocal'#'local_Lesmall'#'local_cross_ctxcross'# 'local_cross'#'localctx1'#'SR_EDSR2im'#'local_ms_ctxB'#
modelpath = './model/s' + str(scale) + '/' + method + '/'  # /pretrain GEU1/’ #Ablation_study/
model = 259000  # 67000# 165000#
endmodel_id = model + 1  # 5001#13001#
name = ['md']
# 'ABC', 'tsukuba', 'kitti2012', 'kitti2015'  'A' ##'tsukuba' #'A'# 'flying'  # 'lr'#'driving' #'flying' #'monkaa'  #
path2 = "F:/SRdata/testsets/depth/C/RGB/"  # './Middlebury/test/'+name+'/left/'
path1 = 'D:/U_copy\StereoSR\IQA\StereoSR/gt/' + name[0] + '/left/'
path3 = './Middlebury/test/quartersize/right_refl2r/'
savepath = './result/s' + str(scale) + '/' + method + '/' + name[0] + 'right/'  # 'left/'#
os.makedirs(savepath, exist_ok=True)


# 'SR_context' # no optical flow
def testSR_context(num=0):
    filenames = get_filenames([path1])
    filenamesr = get_filenames([path2])
    num = len(filenames)
    print(filenames)
    print(filenamesr)
    inp_lr = tf.placeholder(tf.float32, [1, 32, 32, 3])
    inp_b = tf.placeholder(tf.float32, [1, scale * 32, scale * 32, 3])
    
    fl, sl = SR_localbic(inp_lr, inp_b, 128, scale, small=False)
    res = SR_ctxLeft(sl, sl, fl, s=scale)  # # res = SR_ctxLeftRight(sl, sl, fl, fl)
    
    max = 0
    id2 = 0
    mean_psnr2 = 0
    mean_ssim2 = 0
    with tf.Session() as sess:
        for id in range(model, endmodel_id, 1000):
            model_id = id
            mean_psnr = 0
            mean_psnr2 = 0
            mean_ssim = 0
            mean_ssim2 = 0
            sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver(var_list=tf.trainable_variables(), max_to_keep=10000)
            module_file = modelpath + 'model.ckpt-' + str(model_id)
            saver.restore(sess, module_file)
            for fi in range(0, num):
                rgb = np.array(Image.open(filenames[fi]))
                cur = rgb[:, :, 0:3]
                fname = filenames[fi]
                fname = fname[len(path1):]
                rgb2 = np.array(Image.open(filenamesr[fi]))  # np.array(Image.open(path2 + fname[:-8] + 'right.bmp'))#
                cur2 = rgb2[:, :, 0:3]
                h0, w0, c0 = cur.shape
                h = h0 - h0 % scale
                w = w0 - w0 % scale
                cur = cur[:h, :w, :]
                cur2 = cur2[:h, :w, :]
                cur_lr = misc.imresize(cur, 1 / scale, 'bicubic')
                cur_lr2 = np.float32(misc.imresize(cur2, 1 / scale, 'bicubic')) / 255.0
                cur_bic = np.float32(misc.imresize(cur_lr, [h, w, c0], 'bicubic')) / 255.0
                cur_bic2 = np.float32(misc.imresize(cur_lr2, [h, w, c0], 'bicubic')) / 255.0
                cur_lr = np.float32(cur_lr) / 255.0
                
                cur_lr12 = np.rot90(cur_lr, 1)
                cur_lr22 = np.rot90(cur_lr2, 1)
                cur_bic12 = np.rot90(cur_bic, 1)
                cur_bic22 = np.rot90(cur_bic2, 1)
                
                cur_lr12 = np.reshape(cur_lr12, [1, int(w / scale), int(h / scale), 3])
                cur_lr22 = np.reshape(cur_lr22, [1, int(w / scale), int(h / scale), 3])
                cur_bic12 = np.reshape(cur_bic12, [1, w, h, 3])
                cur_bic22 = np.reshape(cur_bic22, [1, w, h, 3])
                cur_lr = np.reshape(cur_lr, [1, int(h / scale), int(w / scale), 3])
                cur_lr2 = np.reshape(cur_lr2, [1, int(h / scale), int(w / scale), 3])
                cur_bic = np.reshape(cur_bic, [1, h, w, 3])
                cur_bic2 = np.reshape(cur_bic2, [1, h, w, 3])
                fl, sl = SR_localbic(tf.convert_to_tensor(cur_lr), tf.convert_to_tensor(cur_bic), 128, scale,
                                     small=False)
                fr, sr = SR_localbic(tf.convert_to_tensor(cur_lr2), tf.convert_to_tensor(cur_bic2), 128, scale,
                                     small=False)
                out = SR_ctxLeft(sl, sr, fl, s=scale)  # out,outr = SR_ctxLeftRight(sl, sr, fl, fr)
                fl, sl = SR_localbic(tf.convert_to_tensor(cur_lr12), tf.convert_to_tensor(cur_bic12), 128, scale,
                                     small=False)
                fr, sr = SR_localbic(tf.convert_to_tensor(cur_lr22), tf.convert_to_tensor(cur_bic22), 128, scale,
                                     small=False)
                out2 = SR_ctxLeft(sl, sr, fl, s=scale)
                
               
                result = sess.run(out, feed_dict={dropout_keep_prob1: 1.0})
                result2 = sess.run(out2, feed_dict={dropout_keep_prob1: 1.0})
                result = np.reshape(result, [h, w, c0])
                result2 = np.reshape(result2, [w, h, c0])
                result2 = np.rot90(result2, -1)
                result = (result + result2) / 2  #
                
                cur_bic = np.reshape(cur_bic, [h, w, c0])
                if (len(rgb.shape) == 3):
                    Image.fromarray(img_to_uint8(np.round(np.maximum(0, np.minimum(1, result)) * 255))).save(
                        savepath + fname)
                else:
                    result_im = Image.fromarray(np.uint8(np.round(np.maximum(0, np.minimum(1, result)) * 255)))
                    result_im.save(savepath + fname)
                c = np.float32(rgb2y(cur[scale:-scale, scale:-scale, :]))
                sr = rgb2y(
                    np.uint8(np.round(np.maximum(0, np.minimum(1, result[scale:-scale, scale:-scale, :])) * 255)))
                psnr1, _ = psnr(c, sr)
                ssim = measure.compare_ssim(c, np.float32(sr), data_range=255)
                print('SR im:', psnr1, ssim)
                mean_ssim += ssim
                mean_psnr += psnr1
                bic = rgb2y(np.uint8(np.round(np.maximum(0, np.minimum(1, cur_bic[scale:-scale, scale:-scale])) * 255)))
                psnr1, _ = psnr(c, bic)
                ssim = measure.compare_ssim(c, np.float32(bic), data_range=255)
                mean_ssim2 += ssim
                mean_psnr2 += psnr1
            mean_psnr /= num  # len(filenames)
            mean_ssim /= num  # len(filenames)
            print(model_id, 'SR', mean_psnr, mean_ssim)
            if (mean_psnr > max):
                max = mean_psnr
                id2 = model_id
                maxss = mean_ssim
            mean_psnr2 /= num  # len(filenames)
            mean_ssim2 /= num  # len(filenames)
            print(id2, 'maxSR', max, maxss)
        print('Bic', mean_ssim2, mean_psnr2, id2, 'maxSR', max, maxss)


# *8
def testSR_context8():  #
    filenames = get_filenames([path1])
    filenamesr = get_filenames([path2])
    num = len(filenames)
    inp_b = tf.placeholder(tf.float32, [1, scale * 32, scale * 32, 3])
    inp_lr = tf.placeholder(tf.float32, [1, 32, 32, 3])
    
    fl, sl = SR_localbic(inp_lr, inp_b, 128, scale=scale, small=False)
    out = SR_ctxLeft(sl, sl, fl, small=False, s=scale)  # out, outr = SR_ctxLeftRight(sl, sl, fl, fl)#
    max = 0
    id2 = 0
    for id in range(model, endmodel_id, 1000):
        model_id = id
        mean_psnr = 0
        mean_psnr2 = 0
        mean_ssim = 0
        mean_ssim2 = 0
        with tf.Session() as sess:
            for fi in range(num):
                rgb = np.array(Image.open(filenames[fi]))
                cur = rgb[:, :, 0:3]
                fname = filenames[fi]
                fname = fname[len(path1):]
                rgb2 = np.array(Image.open(filenamesr[fi]))  # np.array(Image.open(path2 + fname[:-8] + 'right.bmp'))  #
                cur2 = rgb2[:, :, 0:3]
                h0, w0, c0 = cur.shape
                h = h0 - h0 % (scale * scale)
                w = w0 - w0 % (scale * scale)
                cur = cur[:h, :w, :]
                cur2 = cur2[:h, :w, :]
                
                cur_lr = misc.imresize(cur, 1 / scale, 'bicubic')
                cur_lr2 = misc.imresize(cur2, 1 / scale, 'bicubic')
                cur_bic = np.float32(misc.imresize(cur_lr, [h, w, c0], 'bicubic')) / 255.0
                cur_bic2 = np.float32(misc.imresize(cur_lr2, [h, w, c0], 'bicubic')) / 255.0
                cur_lr = np.float32(cur_lr) / 255.0
                cur_lr2 = np.float32(cur_lr2) / 255.0
                # '''
                cur_lr12 = np.rot90(cur_lr, 1)
                cur_lr22 = np.rot90(cur_lr2, 1)
                
                cur_lr13 = np.flip(cur_lr, 1)
                cur_lr23 = np.flip(cur_lr2, 1)
                
                cur_lr14 = np.rot90(cur_lr13, 1)
                cur_lr24 = np.rot90(cur_lr23, 1)
                
                cur_lr15 = np.rot90(cur_lr, 2)
                cur_lr25 = np.rot90(cur_lr2, 2)
                
                cur_lr16 = np.rot90(cur_lr, 3)
                cur_lr26 = np.rot90(cur_lr2, 3)
                
                cur_lr17 = np.rot90(cur_lr13, 2)
                cur_lr27 = np.rot90(cur_lr23, 2)
                
                cur_lr18 = np.rot90(cur_lr13, 3)
                cur_lr28 = np.rot90(cur_lr23, 3)
                
                cur_bic12 = np.rot90(cur_bic, 1)
                cur_bic22 = np.rot90(cur_bic2, 1)
                
                cur_bic13 = np.flip(cur_bic, 1)
                cur_bic23 = np.flip(cur_bic2, 1)
                
                cur_bic14 = np.rot90(cur_bic13, 1)
                cur_bic24 = np.rot90(cur_bic23, 1)
                
                cur_bic15 = np.rot90(cur_bic, 2)
                cur_bic25 = np.rot90(cur_bic2, 2)
                
                cur_bic16 = np.rot90(cur_bic, 3)
                cur_bic26 = np.rot90(cur_bic2, 3)
                
                cur_bic17 = np.rot90(cur_bic13, 2)
                cur_bic27 = np.rot90(cur_bic23, 2)
                
                cur_bic18 = np.rot90(cur_bic13, 3)
                cur_bic28 = np.rot90(cur_bic23, 3)
                
                cur_lr12 = np.reshape(cur_lr12, [1, int(w / scale), int(h / scale), 3])
                cur_lr22 = np.reshape(cur_lr22, [1, int(w / scale), int(h / scale), 3])
                cur_lr13 = np.reshape(cur_lr13, [1, int(h / scale), int(w / scale), 3])
                cur_lr23 = np.reshape(cur_lr23, [1, int(h / scale), int(w / scale), 3])
                cur_lr14 = np.reshape(cur_lr14, [1, int(w / scale), int(h / scale), 3])
                cur_lr24 = np.reshape(cur_lr24, [1, int(w / scale), int(h / scale), 3])
                cur_lr16 = np.reshape(cur_lr16, [1, int(w / scale), int(h / scale), 3])
                cur_lr26 = np.reshape(cur_lr26, [1, int(w / scale), int(h / scale), 3])
                cur_lr18 = np.reshape(cur_lr18, [1, int(w / scale), int(h / scale), 3])
                cur_lr28 = np.reshape(cur_lr28, [1, int(w / scale), int(h / scale), 3])
                cur_lr15 = np.reshape(cur_lr15, [1, int(h / scale), int(w / scale), 3])
                cur_lr25 = np.reshape(cur_lr25, [1, int(h / scale), int(w / scale), 3])
                cur_lr17 = np.reshape(cur_lr17, [1, int(h / scale), int(w / scale), 3])
                cur_lr27 = np.reshape(cur_lr27, [1, int(h / scale), int(w / scale), 3])  # '''
                cur_lr = np.reshape(cur_lr, [1, int(h / scale), int(w / scale), 3])
                cur_lr2 = np.reshape(cur_lr2, [1, int(h / scale), int(w / scale), 3])
                
                cur_bic12 = np.reshape(cur_bic12, [1, w, h, 3])
                cur_bic22 = np.reshape(cur_bic22, [1, w, h, 3])
                cur_bic13 = np.reshape(cur_bic13, [1, h, w, 3])
                cur_bic23 = np.reshape(cur_bic23, [1, h, w, 3])
                cur_bic14 = np.reshape(cur_bic14, [1, w, h, 3])
                cur_bic24 = np.reshape(cur_bic24, [1, w, h, 3])
                cur_bic16 = np.reshape(cur_bic16, [1, w, h, 3])
                cur_bic26 = np.reshape(cur_bic26, [1, w, h, 3])
                cur_bic18 = np.reshape(cur_bic18, [1, w, h, 3])
                cur_bic28 = np.reshape(cur_bic28, [1, w, h, 3])
                cur_bic15 = np.reshape(cur_bic15, [1, h, w, 3])
                cur_bic25 = np.reshape(cur_bic25, [1, h, w, 3])
                cur_bic17 = np.reshape(cur_bic17, [1, h, w, 3])
                cur_bic27 = np.reshape(cur_bic27, [1, h, w, 3])
                cur_bic = np.reshape(cur_bic, [1, h, w, 3])
                cur_bic2 = np.reshape(cur_bic2, [1, h, w, 3])
                
                fl, sl = SR_localbic(tf.convert_to_tensor(cur_lr), tf.convert_to_tensor(cur_bic), 128, scale=scale,
                                     small=False)
                fr, sr = SR_localbic(tf.convert_to_tensor(cur_lr2), tf.convert_to_tensor(cur_bic2), 128, scale=scale,
                                     small=False)
                out = SR_ctxLeft(sl, sr, fl, small=False, s=scale)  # out, outr = SR_ctxLeftRight(sl, sr, fl, fr) #
                fl, sl = SR_localbic(tf.convert_to_tensor(cur_lr12), tf.convert_to_tensor(cur_bic12), 128, scale=scale,
                                     small=False)
                fr, sr = SR_localbic(tf.convert_to_tensor(cur_lr22), tf.convert_to_tensor(cur_bic22), 128, scale=scale,
                                     small=False)
                # out2, outr2 = SR_ctxLeftRight(sl, sr, fl,fr)
                out2 = SR_ctxLeft(sl, sr, fl, small=False, s=scale)  #
                fl, sl = SR_localbic(tf.convert_to_tensor(cur_lr13), tf.convert_to_tensor(cur_bic13), 128, scale=scale,
                                     small=False)
                fr, sr = SR_localbic(tf.convert_to_tensor(cur_lr23), tf.convert_to_tensor(cur_bic23), 128, scale=scale,
                                     small=False)
                out3 = SR_ctxLeft(sl, sr, fl, small=False, s=scale)  # , outr3 = SR_ctxLeftRight(sl, sr, fl, fr)#
                fl, sl = SR_localbic(tf.convert_to_tensor(cur_lr14), tf.convert_to_tensor(cur_bic14), 128, scale=scale,
                                     small=False)
                fr, sr = SR_localbic(tf.convert_to_tensor(cur_lr24), tf.convert_to_tensor(cur_bic24), 128, scale=scale,
                                     small=False)
                out4 = SR_ctxLeft(sl, sr, fl, small=False, s=scale)  # , outr4 = SR_ctxLeftRight(sl, sr, fl, fr)#
                fl, sl = SR_localbic(tf.convert_to_tensor(cur_lr15), tf.convert_to_tensor(cur_bic15), 128, scale=scale,
                                     small=False)
                fr, sr = SR_localbic(tf.convert_to_tensor(cur_lr25), tf.convert_to_tensor(cur_bic25), 128, scale=scale,
                                     small=False)
                out5 = SR_ctxLeft(sl, sr, fl, small=False, s=scale)  # , outr5 = SR_ctxLeftRight(sl, sr, fl, fr)#
                fl, sl = SR_localbic(tf.convert_to_tensor(cur_lr16), tf.convert_to_tensor(cur_bic16), 128, scale=scale,
                                     small=False)
                fr, sr = SR_localbic(tf.convert_to_tensor(cur_lr26), tf.convert_to_tensor(cur_bic26), 128, scale=scale,
                                     small=False)
                out6 = SR_ctxLeft(sl, sr, fl, small=False, s=scale)  # , outr6 = SR_ctxLeftRight(sl, sr, fl, fr)#
                fl, sl = SR_localbic(tf.convert_to_tensor(cur_lr17), tf.convert_to_tensor(cur_bic17), 128, scale=scale,
                                     small=False)
                fr, sr = SR_localbic(tf.convert_to_tensor(cur_lr27), tf.convert_to_tensor(cur_bic27), 128, scale=scale,
                                     small=False)
                out7 = SR_ctxLeft(sl, sr, fl, small=False, s=scale)  # , outr7 = SR_ctxLeftRight(sl, sr, fl, fr)#
                fl, sl = SR_localbic(tf.convert_to_tensor(cur_lr18), tf.convert_to_tensor(cur_bic18), 128, scale=scale,
                                     small=False)
                fr, sr = SR_localbic(tf.convert_to_tensor(cur_lr28), tf.convert_to_tensor(cur_bic28), 128, scale=scale,
                                     small=False)
                out8 = SR_ctxLeft(sl, sr, fl, small=False, s=scale)  # , outr8 = SR_ctxLeftRight(sl, sr, fl, fr)#
                sess.run(tf.global_variables_initializer())
                saver = tf.train.Saver()
                module_file = modelpath + 'model.ckpt-' + str(model_id)
                saver.restore(sess, module_file)
                result, result2 = sess.run([out, out2], feed_dict={dropout_keep_prob1: 1.0})
                result3, result4 = sess.run([out3, out4], feed_dict={dropout_keep_prob1: 1.0})
                result5, result6 = sess.run([out5, out6], feed_dict={dropout_keep_prob1: 1.0})
                result7, result8 = sess.run([out7, out8], feed_dict={dropout_keep_prob1: 1.0})
                
                result = np.reshape(result, [h, w, c0])
                result2 = np.reshape(result2, [w, h, c0])
                result2 = np.rot90(result2, -1)
                result5 = np.reshape(result5, [h, w, c0])
                result5 = np.rot90(result5, -2)
                result6 = np.reshape(result6, [w, h, c0])
                result6 = np.rot90(result6, -3)
                result3 = np.reshape(result3, [h, w, c0])
                result3 = np.flip(result3, 1)
                result4 = np.reshape(result4, [w, h, c0])
                result4 = np.flip(np.rot90(result4, -1), 1)
                result7 = np.reshape(result7, [h, w, c0])
                result7 = np.flip(np.rot90(result7, -2), 1)
                result8 = np.reshape(result8, [w, h, c0])
                result8 = np.flip(np.rot90(result8, -3), 1)
                result = (result + result2 + result3 + result4 + result5 + result6 + result7 + result8) / 8  #
                cur_bic = np.reshape(cur_bic, [h, w, c0])
                if (len(rgb.shape) == 3):
                    Image.fromarray(img_to_uint8(np.round(np.maximum(0, np.minimum(1, result)) * 255))).save(
                        savepath + fname)
                c = np.float32(rgb2y(cur[scale:-scale, scale:-scale, :]))
                sr = rgb2y(
                    np.uint8(np.round(np.maximum(0, np.minimum(1, result[scale:-scale, scale:-scale, :])) * 255)))
                bic = rgb2y(np.uint8(np.round(np.maximum(0, np.minimum(1, cur_bic[scale:-scale, scale:-scale])) * 255)))
                
                psnr1, _ = psnr(c, sr)
                ssim = measure.compare_ssim(c, np.float32(sr),
                                            data_range=255)  # measure.compare_ssim(c / 255.0, sr / 255.0, multichannel=True)  #
                print('SR im:', psnr1, ssim)
                mean_ssim += ssim
                mean_psnr += psnr1
                psnr1, _ = psnr(c, bic)
                ssim = measure.compare_ssim(c / 255.0, bic / 255.0, data_range=255)  # , multichannel=True)  #
                mean_ssim2 += ssim
                mean_psnr2 += psnr1
        mean_psnr /= num  # len(filenames)
        mean_ssim /= num  # len(filenames)
        print(model_id, 'SR', mean_psnr, mean_ssim)
        if (mean_psnr > max):
            max = mean_psnr
            id2 = model_id
            maxss = mean_ssim
        mean_psnr2 /= num  # len(filenames)
        mean_ssim2 /= num  # len(filenames)
        print('Bic', mean_ssim2, mean_psnr2, id2, 'maxSR', max, maxss)


# ================ 同时超分左右图 ========================#
def testSR_context2im8(start='0', finish='5'):  # 同时超分左右图
    filenames = get_filenames([path1])
    inp_lr = tf.placeholder(tf.float32, [1, 32, 32, 3])
    inp_b = tf.placeholder(tf.float32, [1, scale * 32, scale * 32, 3])
    
    # local_LeRitwo
    fl, sl = SR_localbic(inp_lr, inp_b, 128, small=False)
    res = SR_ctxLeftRight(sl, sl, fl, fl)  #
    max = 0
    id2 = 0
    for id in range(model, endmodel_id, 1000):
        model_id = id
        mean_psnr = 0
        mean_psnr2 = 0
        mean_ssim = 0
        mean_ssim2 = 0
        mean_psnrr = 0
        mean_psnrr2 = 0
        mean_ssimr = 0
        mean_ssimr2 = 0
        for fi in range(int(start), int(finish)):
            rgb = np.array(Image.open(filenames[fi]))
            cur = rgb[:, :, 0:3]
            fname = filenames[fi]
            fname = fname[len(path1):]
            rgb2 = np.array(Image.open(path2 + fname[:-8] + 'right.bmp'))
            cur2 = rgb2[:, :, 0:3]
            h0, w0, c0 = cur.shape
            h = h0 - h0 % scale
            w = w0 - w0 % scale
            cur = cur[:h, :w, :]
            cur2 = cur2[:h, :w, :]
            cur_lr = misc.imresize(cur, 1 / scale, 'bicubic')
            cur_lr2 = misc.imresize(cur2, 1 / scale, 'bicubic')
            cur_bic = np.float32(misc.imresize(cur_lr, [h, w, c0], 'bicubic')) / 255.0
            cur_bic2 = np.float32(misc.imresize(cur_lr2, [h, w, c0], 'bicubic')) / 255.0
            cur_lr = np.float32(cur_lr) / 255.0
            cur_lr2 = np.float32(cur_lr2) / 255.0
            
            cur_lr12 = np.rot90(cur_lr, 1)
            cur_lr22 = np.rot90(cur_lr2, 1)
            cur_lr13 = np.flip(cur_lr, 1)
            cur_lr23 = np.flip(cur_lr2, 1)
            cur_lr14 = np.rot90(cur_lr13, 1)
            cur_lr24 = np.rot90(cur_lr23, 1)
            cur_lr15 = np.rot90(cur_lr, 2)
            cur_lr25 = np.rot90(cur_lr2, 2)
            cur_lr16 = np.rot90(cur_lr, 3)
            cur_lr26 = np.rot90(cur_lr2, 3)
            cur_lr17 = np.rot90(cur_lr13, 2)
            cur_lr27 = np.rot90(cur_lr23, 2)
            cur_lr18 = np.rot90(cur_lr13, 3)
            cur_lr28 = np.rot90(cur_lr23, 3)
            cur_bic12 = np.rot90(cur_bic, 1)
            cur_bic22 = np.rot90(cur_bic2, 1)
            cur_bic13 = np.flip(cur_bic, 1)
            cur_bic23 = np.flip(cur_bic2, 1)
            cur_bic14 = np.rot90(cur_bic13, 1)
            cur_bic24 = np.rot90(cur_bic23, 1)
            cur_bic15 = np.rot90(cur_bic, 2)
            cur_bic25 = np.rot90(cur_bic2, 2)
            cur_bic16 = np.rot90(cur_bic, 3)
            cur_bic26 = np.rot90(cur_bic2, 3)
            cur_bic17 = np.rot90(cur_bic13, 2)
            cur_bic27 = np.rot90(cur_bic23, 2)
            cur_bic18 = np.rot90(cur_bic13, 3)
            cur_bic28 = np.rot90(cur_bic23, 3)
            cur_lr12 = np.reshape(cur_lr12, [1, int(w / scale), int(h / scale), 3])
            cur_lr22 = np.reshape(cur_lr22, [1, int(w / scale), int(h / scale), 3])
            cur_lr13 = np.reshape(cur_lr13, [1, int(h / scale), int(w / scale), 3])
            cur_lr23 = np.reshape(cur_lr23, [1, int(h / scale), int(w / scale), 3])
            cur_lr14 = np.reshape(cur_lr14, [1, int(w / scale), int(h / scale), 3])
            cur_lr24 = np.reshape(cur_lr24, [1, int(w / scale), int(h / scale), 3])
            cur_lr16 = np.reshape(cur_lr16, [1, int(w / scale), int(h / scale), 3])
            cur_lr26 = np.reshape(cur_lr26, [1, int(w / scale), int(h / scale), 3])
            cur_lr18 = np.reshape(cur_lr18, [1, int(w / scale), int(h / scale), 3])
            cur_lr28 = np.reshape(cur_lr28, [1, int(w / scale), int(h / scale), 3])
            cur_lr15 = np.reshape(cur_lr15, [1, int(h / scale), int(w / scale), 3])
            cur_lr25 = np.reshape(cur_lr25, [1, int(h / scale), int(w / scale), 3])
            cur_lr17 = np.reshape(cur_lr17, [1, int(h / scale), int(w / scale), 3])
            cur_lr27 = np.reshape(cur_lr27, [1, int(h / scale), int(w / scale), 3])  # '''
            cur_lr = np.reshape(cur_lr, [1, int(h / scale), int(w / scale), 3])
            cur_lr2 = np.reshape(cur_lr2, [1, int(h / scale), int(w / scale), 3])
            cur_bic12 = np.reshape(cur_bic12, [1, w, h, 3])
            cur_bic22 = np.reshape(cur_bic22, [1, w, h, 3])
            cur_bic13 = np.reshape(cur_bic13, [1, h, w, 3])
            cur_bic23 = np.reshape(cur_bic23, [1, h, w, 3])
            cur_bic14 = np.reshape(cur_bic14, [1, w, h, 3])
            cur_bic24 = np.reshape(cur_bic24, [1, w, h, 3])
            cur_bic16 = np.reshape(cur_bic16, [1, w, h, 3])
            cur_bic26 = np.reshape(cur_bic26, [1, w, h, 3])
            cur_bic18 = np.reshape(cur_bic18, [1, w, h, 3])
            cur_bic28 = np.reshape(cur_bic28, [1, w, h, 3])
            cur_bic15 = np.reshape(cur_bic15, [1, h, w, 3])
            cur_bic25 = np.reshape(cur_bic25, [1, h, w, 3])
            cur_bic17 = np.reshape(cur_bic17, [1, h, w, 3])
            cur_bic27 = np.reshape(cur_bic27, [1, h, w, 3])
            cur_bic = np.reshape(cur_bic, [1, h, w, 3])
            cur_bic2 = np.reshape(cur_bic2, [1, h, w, 3])
            with tf.Session() as sess:
                with tf.variable_scope('', reuse=True):
                    ''' # SR_ctx2im
                    fl, sl = SR_local(tf.convert_to_tensor(cur_lr), 128, reuse=False)
                    fr, sr = SR_local(tf.convert_to_tensor(cur_lr2), 128, reuse=True)
                    out,outr = SR_ctx2im(sl, sr, fl, fr)
                    fl, sl = SR_local(tf.convert_to_tensor(cur_lr12), 128, reuse=False)
                    fr, sr = SR_local(tf.convert_to_tensor(cur_lr22), 128, reuse=True)
                    _, out2 = SR_ctx2im(sl, sr, fl, fr)
                    fl, sl = SR_local(tf.convert_to_tensor(cur_lr13), 128, reuse=False)
                    fr, sr = SR_local(tf.convert_to_tensor(cur_lr23), 128, reuse=True)
                    _, out3 = SR_ctx2im(sl, sr, fl, fr)
                    fl, sl = SR_local(tf.convert_to_tensor(cur_lr14), 128, reuse=False)
                    fr, sr = SR_local(tf.convert_to_tensor(cur_lr24), 128, reuse=True)
                    _, out4 = SR_ctx2im(sl, sr, fl, fr)
                    fl, sl = SR_local(tf.convert_to_tensor(cur_lr15), 128, reuse=False)
                    fr, sr = SR_local(tf.convert_to_tensor(cur_lr25), 128, reuse=True)
                    _, out5 = SR_ctx2im(sl, sr, fl, fr)
                    fl, sl = SR_local(tf.convert_to_tensor(cur_lr16), 128, reuse=False)
                    fr, sr = SR_local(tf.convert_to_tensor(cur_lr26), 128, reuse=True)
                    _, out6 = SR_ctx2im(sl, sr, fl, fr)
                    fl, sl = SR_local(tf.convert_to_tensor(cur_lr17), 128, reuse=False)
                    fr, sr = SR_local(tf.convert_to_tensor(cur_lr27), 128, reuse=True)
                    _, out7 = SR_ctx2im(sl, sr, fl, fr)
                    fl, sl = SR_local(tf.convert_to_tensor(cur_lr18), 128, reuse=False)
                    fr, sr = SR_local(tf.convert_to_tensor(cur_lr28), 128, reuse=True)
                    _, out8 = SR_ctx2im(sl, sr, fl, fr)'''
                    # SREDSR2im
                    # sl, fl = EDSR(tf.convert_to_tensor(cur_lr), scale=scale, reuse=False)
                    # sr, fr = EDSR(tf.convert_to_tensor(cur_lr2), scale=scale, reuse=True)
                    # out, outr = SR_ctx2im(sl, sr, fl, fr, s=scale)
                    # sl, fl = EDSR(tf.convert_to_tensor(cur_lr12), scale=scale, reuse=False)
                    # sr, fr = EDSR(tf.convert_to_tensor(cur_lr22), scale=scale, reuse=True)
                    # outl2, out2 = SR_ctx2im(sl, sr, fl, fr, s=scale)
                    # sl, fl = EDSR(tf.convert_to_tensor(cur_lr13), scale=scale, reuse=False)
                    # sr, fr = EDSR(tf.convert_to_tensor(cur_lr23), scale=scale, reuse=True)
                    # outl3, out3 = SR_ctx2im(sl, sr, fl, fr, s=scale)
                    # sl, fl = EDSR(tf.convert_to_tensor(cur_lr14), scale=scale, reuse=False)
                    # sr, fr = EDSR(tf.convert_to_tensor(cur_lr24), scale=scale, reuse=True)
                    # outl4, out4 = SR_ctx2im(sl, sr, fl, fr, s=scale)
                    # sl, fl = EDSR(tf.convert_to_tensor(cur_lr15), scale=scale, reuse=False)
                    # sr, fr = EDSR(tf.convert_to_tensor(cur_lr25), scale=scale, reuse=True)
                    # outl5, out5 = SR_ctx2im(sl, sr, fl, fr, s=scale)
                    # sl, fl = EDSR(tf.convert_to_tensor(cur_lr16), scale=scale, reuse=False)
                    # sr, fr = EDSR(tf.convert_to_tensor(cur_lr26), scale=scale, reuse=True)
                    # outl6, out6 = SR_ctx2im(sl, sr, fl, fr, s=scale)
                    # sl, fl = EDSR(tf.convert_to_tensor(cur_lr17), scale=scale, reuse=False)
                    # sr, fr = EDSR(tf.convert_to_tensor(cur_lr27), scale=scale, reuse=True)
                    # outl7, out7 = SR_ctx2im(sl, sr, fl, fr, s=scale)
                    # sl, fl = EDSR(tf.convert_to_tensor(cur_lr18), scale=scale, reuse=False)
                    # sr, fr = EDSR(tf.convert_to_tensor(cur_lr28), scale=scale, reuse=True)
                    # outl8, out8 = SR_ctx2im(sl, sr, fl, fr, s=scale)  # SR_imfed'''
                    # local_LeRitwo
                    fl, sl = SR_localbic(tf.convert_to_tensor(cur_lr), tf.convert_to_tensor(cur_bic), 128, small=False)
                    fr, sr = SR_localbic(tf.convert_to_tensor(cur_lr2), tf.convert_to_tensor(cur_bic2), 128,
                                         small=False)
                    outl, outr = SR_ctxLeftRight(sl, sr, fl, fr)  #
                    fl, sl = SR_localbic(tf.convert_to_tensor(cur_lr12), tf.convert_to_tensor(cur_bic12), 128,
                                         small=False)
                    fr, sr = SR_localbic(tf.convert_to_tensor(cur_lr22), tf.convert_to_tensor(cur_bic22), 128,
                                         small=False)
                    outl2, outr2 = SR_ctxLeftRight(sl, sr, fl, fr)  #
                    fl, sl = SR_localbic(tf.convert_to_tensor(cur_lr13), tf.convert_to_tensor(cur_bic13), 128,
                                         small=False)
                    fr, sr = SR_localbic(tf.convert_to_tensor(cur_lr23), tf.convert_to_tensor(cur_bic23), 128,
                                         small=False)
                    outl3, outr3 = SR_ctxLeftRight(sl, sr, fl, fr)  #
                    fl, sl = SR_localbic(tf.convert_to_tensor(cur_lr14), tf.convert_to_tensor(cur_bic14), 128,
                                         small=False)
                    fr, sr = SR_localbic(tf.convert_to_tensor(cur_lr24), tf.convert_to_tensor(cur_bic24), 128,
                                         small=False)
                    outl4, outr4 = SR_ctxLeftRight(sl, sr, fl, fr)  #
                    fl, sl = SR_localbic(tf.convert_to_tensor(cur_lr15), tf.convert_to_tensor(cur_bic15), 128,
                                         small=False)
                    fr, sr = SR_localbic(tf.convert_to_tensor(cur_lr25), tf.convert_to_tensor(cur_bic25), 128,
                                         small=False)
                    outl5, outr5 = SR_ctxLeftRight(sl, sr, fl, fr)  #
                    fl, sl = SR_localbic(tf.convert_to_tensor(cur_lr16), tf.convert_to_tensor(cur_bic16), 128,
                                         small=False)
                    fr, sr = SR_localbic(tf.convert_to_tensor(cur_lr26), tf.convert_to_tensor(cur_bic26), 128,
                                         small=False)
                    outl6, outr6 = SR_ctxLeftRight(sl, sr, fl, fr)  #
                    fl, sl = SR_localbic(tf.convert_to_tensor(cur_lr17), tf.convert_to_tensor(cur_bic17), 128,
                                         small=False)
                    fr, sr = SR_localbic(tf.convert_to_tensor(cur_lr27), tf.convert_to_tensor(cur_bic27), 128,
                                         small=False)
                    outl7, outr7 = SR_ctxLeftRight(sl, sr, fl, fr)  #
                    fl, sl = SR_localbic(tf.convert_to_tensor(cur_lr18), tf.convert_to_tensor(cur_bic18), 128,
                                         small=False)
                    fr, sr = SR_localbic(tf.convert_to_tensor(cur_lr28), tf.convert_to_tensor(cur_bic28), 128,
                                         small=False)
                    outl8, outr8 = SR_ctxLeftRight(sl, sr, fl, fr)  #
                    sess.run(tf.global_variables_initializer())
                    saver = tf.train.Saver(var_list=tf.trainable_variables(), max_to_keep=10000)
                    for index in range(model_id, model_id - 1, -1):
                        module_file = modelpath + 'model.ckpt-' + str(index)
                        saver.restore(sess, module_file)
                        # resultl,resultl2,resultl3,resultl4 = sess.run([outl,outl2,outl3,outl4], feed_dict={dropout_keep_prob1: 1.0})
                        # resultl5,resultl6,resultl7,resultl8 = sess.run([outl5,outl6,outl7,outl8], feed_dict={dropout_keep_prob1: 1.0})
                        resultr, result2 = sess.run([outr, outr2], feed_dict={dropout_keep_prob1: 1.0})
                        result3, result4 = sess.run([outr3, outr4], feed_dict={dropout_keep_prob1: 1.0})
                        result5, result6 = sess.run([outr5, outr6], feed_dict={dropout_keep_prob1: 1.0})
                        result7, result8 = sess.run([outr7, outr8], feed_dict={dropout_keep_prob1: 1.0})
                    # resultl = np.reshape(resultl, [h, w, c0])
                    # resultl2 = np.reshape(resultl2, [w, h, c0])
                    # resultl2 = np.rot90(resultl2, -1)
                    # resultl5 = np.reshape(resultl5, [h, w, c0])
                    # resultl5 = np.rot90(resultl5, -2)
                    # resultl6 = np.reshape(resultl6, [w, h, c0])
                    # resultl6 = np.rot90(resultl6, -3)
                    # resultl3 = np.reshape(resultl3, [h, w, c0])
                    # resultl3 = np.flip(resultl3, 1)
                    # resultl4 = np.reshape(resultl4, [w, h, c0])
                    # resultl4 = np.flip(np.rot90(resultl4, -1), 1)
                    # resultl7 = np.reshape(resultl7, [h, w, c0])
                    # resultl7 = np.flip(np.rot90(resultl7, -2), 1)
                    # resultl8 = np.reshape(resultl8, [w, h, c0])
                    # resultl8 = np.flip(np.rot90(resultl8, -3), 1)
                    # resultl = (resultl + resultl2 + resultl3 + resultl4 + resultl5 + resultl6 + resultl7 + resultl8) / 8  #
                    # right:
                    resultr = np.reshape(resultr, [h, w, c0])
                    result2 = np.reshape(result2, [w, h, c0])
                    result2 = np.rot90(result2, -1)
                    result5 = np.reshape(result5, [h, w, c0])
                    result5 = np.rot90(result5, -2)
                    result6 = np.reshape(result6, [w, h, c0])
                    result6 = np.rot90(result6, -3)
                    result3 = np.reshape(result3, [h, w, c0])
                    result3 = np.flip(result3, 1)
                    result4 = np.reshape(result4, [w, h, c0])
                    result4 = np.flip(np.rot90(result4, -1), 1)
                    result7 = np.reshape(result7, [h, w, c0])
                    result7 = np.flip(np.rot90(result7, -2), 1)
                    result8 = np.reshape(result8, [w, h, c0])
                    result8 = np.flip(np.rot90(result8, -3), 1)
                    resultr = (resultr + result2 + result3 + result4 + result5 + result6 + result7 + result8) / 8  #
            cur_bic = np.reshape(cur_bic, [h, w, c0])
            cur_bicr = np.reshape(cur_bic2, [h, w, c0])
            if (len(rgb.shape) == 3):
                # Image.fromarray(img_to_uint8(np.round(np.maximum(0, np.minimum(1, resultl)) * 255))).save(
                #     savepath + fname)
                Image.fromarray(img_to_uint8(np.round(np.maximum(0, np.minimum(1, resultr)) * 255))).save(
                    savepath + 'right_' + fname)
            else:
                # Image.fromarray(np.uint8(np.round(np.maximum(0, np.minimum(1, resultl)) * 255))).save(savepath + fname)
                Image.fromarray(np.uint8(np.round(np.maximum(0, np.minimum(1, resultr)) * 255))).save(
                    savepath + 'right_' + fname)
            cy = np.float32(rgb2y(cur[scale:-scale, scale:-scale, :]))
            cry = np.float32(rgb2y(cur2[scale:-scale, scale:-scale, :]))
            # sry = rgb2y(np.uint8(np.round(np.maximum(0, np.minimum(1, resultl[scale:-scale, scale:-scale, :])) * 255)))
            srry = rgb2y(np.uint8(np.round(np.maximum(0, np.minimum(1, resultr[scale:-scale, scale:-scale, :])) * 255)))
            c = np.float32(cur[scale:-scale, scale:-scale, :])
            cr = np.float32(cur2[scale:-scale, scale:-scale, :])
            # sr = np.round(np.maximum(0, np.minimum(1, resultl[scale:-scale, scale:-scale, :])) * 255)
            srr = np.round(np.maximum(0, np.minimum(1, resultr[scale:-scale, scale:-scale, :])) * 255)
            
            # psnr1, _ = psnr(c, sr)
            psnrr1, _ = psnr(cr, srr)
            # ssim = measure.compare_ssim(c, np.float32(sr), data_range=255,multichannel=True)
            ssimr = measure.compare_ssim(cr, np.float32(srr), data_range=255, multichannel=True)
            print('right SR im:', psnrr1, ssimr)  # ('SR im:', psnr1, ssim)#
            # mean_ssim += ssim
            # mean_psnr += psnr1
            mean_ssimr += ssimr
            mean_psnrr += psnrr1
            bicy = rgb2y(np.uint8(np.round(np.maximum(0, np.minimum(1, cur_bic[scale:-scale, scale:-scale])) * 255)))
            bicry = rgb2y(np.uint8(np.round(np.maximum(0, np.minimum(1, cur_bicr[scale:-scale, scale:-scale])) * 255)))
            bic = np.uint8(np.round(np.maximum(0, np.minimum(1, cur_bic[scale:-scale, scale:-scale])) * 255))
            bicr = np.uint8(np.round(np.maximum(0, np.minimum(1, cur_bicr[scale:-scale, scale:-scale])) * 255))
            psnr1, _ = psnr(c, bic)
            psnrr1, _ = psnr(cr, bicr)
            ssim = measure.compare_ssim(c, np.float32(bic), data_range=255, multichannel=True)
            ssimr = measure.compare_ssim(cr, np.float32(bicr), data_range=255, multichannel=True)
            mean_ssim2 += ssim
            mean_psnr2 += psnr1
            mean_ssimr2 += ssimr
            mean_psnrr2 += psnrr1
        print('right SR', mean_psnrr / len(filenames), mean_ssimr / len(filenames))
        print(model_id, 'left SR', mean_psnr / len(filenames), mean_ssim / len(filenames))  #
        if (mean_psnrr / len(filenames) > max):
            max = mean_psnrr / len(filenames)
            id2 = model_id
            maxss = mean_ssimr / len(filenames)
        print('Bic', mean_ssim2 / len(filenames), mean_psnr2 / len(filenames),
              'right Bic', mean_ssimr2 / len(filenames), mean_psnrr2 / len(filenames), id2, 'max right SR', max, maxss)


if __name__ == '__main__':
    for nm in name:
        path1 = 'D:/U_copy\StereoSR\IQA\StereoSR/gt/' + nm + '/right/'
        path2 = 'D:/U_copy\StereoSR\IQA\StereoSR/gt/' + nm + '/left/'
        savepath = './result/s' + str(scale) + '/' + method + '/' + nm + 'right/'  # 'left/'#
        if not os.path.exists(savepath):
            os.makedirs(savepath)
        testSR_context8()
