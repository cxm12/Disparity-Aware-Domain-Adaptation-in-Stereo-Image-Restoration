import tensorflow as tf

try:
    image_summary = tf.image_summary
    scalar_summary = tf.scalar_summary
    histogram_summary = tf.histogram_summary
    merge_summary = tf.merge_summary
    SummaryWriter = tf.train.SummaryWriter
except:
    image_summary = tf.summary.image
    scalar_summary = tf.summary.scalar
    histogram_summary = tf.summary.histogram
    merge_summary = tf.summary.merge
    SummaryWriter = tf.summary.FileWriter

if "concat_v2" in dir(tf):
    def concat(tensors, axis, *args, **kwargs):
        return tf.concat_v2(tensors, axis, *args, **kwargs)
else:
    def concat(tensors, axis, *args, **kwargs):
        return tf.concat(tensors, axis, *args, **kwargs)


def conv_cond_concat(x, y):
    """Concatenate conditioning vector on feature map axis."""
    x_shapes = x.get_shape()
    y_shapes = y.get_shape()
    return concat([
        x, y * tf.ones([x_shapes[0], x_shapes[1], x_shapes[2], y_shapes[3]])], 3)


def _weight_decay(var, wd=0.0001):
    weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
    tf.add_to_collection('losses', weight_decay)


def conv2d(input_, output_dim,
           k_h=3, k_w=3, d_h=1, d_w=1, stddev=0.02,
           name="conv2d", init_value=[], trainable=True, hasBias=True):
    # sc=scale_kernel*math.sqrt(2/(k_h*k_w*(input_.get_shape()[-1].value)* output_dim))
    
    with tf.variable_scope(name):
        # w=tf.Variable(sc*tf.random_normal([k_h, k_w, input_.get_shape()[-1].value, output_dim]),name='w')
        if (not (init_value)):
            w = tf.get_variable('w', [k_h, k_w, input_.get_shape()[-1], output_dim], trainable=trainable,
                                initializer=tf.contrib.layers.xavier_initializer())
        else:
            vname = tf.get_variable_scope()
            w = tf.Variable(init_value[vname.name + '/w:0'], name='w', trainable=trainable)
        _weight_decay(w)
        conv = tf.nn.conv2d(input_, w, strides=[1, d_h, d_w, 1], padding='SAME')
        
        if hasBias:
            biases = tf.get_variable('biases', [output_dim], initializer=tf.constant_initializer(0.0))
            conv = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape())
        
        return conv


def conv3d(input_, output_dim,
           k_d=2, k_h=3, k_w=3, d_d=1, d_h=1, d_w=1,
           name="conv2d", init_value=[], trainable=True):
    with tf.variable_scope(name):
        if (not (init_value)):
            w = tf.get_variable('w', [k_d, k_h, k_w, input_.get_shape()[-1], output_dim], trainable=trainable,
                                initializer=tf.contrib.layers.xavier_initializer())
        else:
            vname = tf.get_variable_scope()
            w = tf.Variable(init_value[vname.name + '/w:0'], name='w', trainable=trainable)
        _weight_decay(w)
        conv = tf.nn.conv3d(input_, w, [1, d_d, d_h, d_w, 1], padding='SAME', dilations=[1, 1, 1, 1, 1], name='r1')
        return conv


def depthwise_conv2d(input_, channel_multiplier,
                     k_h=3, k_w=3, d_h=1, d_w=1, stddev=0.02,
                     name="conv2d", init_value=[], trainable=True, hasBias=True):
    # sc=scale_kernel*math.sqrt(2/(k_h*k_w*(input_.get_shape()[-1].value)* output_dim))
    
    with tf.variable_scope(name):
        # w=tf.Variable(sc*tf.random_normal([k_h, k_w, input_.get_shape()[-1].value, output_dim]),name='w')
        if (not (init_value)):
            w = tf.get_variable('w', [k_h, k_w, input_.get_shape()[-1], channel_multiplier], trainable=trainable,
                                initializer=tf.contrib.layers.xavier_initializer())
        else:
            vname = tf.get_variable_scope()
            w = tf.Variable(init_value[vname.name + '/w:0'], name='w', trainable=trainable)
        _weight_decay(w)
        conv = tf.nn.depthwise_conv2d(input_, w, strides=[1, d_h, d_w, 1], padding='SAME')
        
        if hasBias:
            biases = tf.get_variable('biases', [channel_multiplier * (input_.get_shape()[-1].value)],
                                     initializer=tf.constant_initializer(0.0))
            conv = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape())
        
        return conv


def dilated_conv2d(input_, output_dim,
                   k_h=3, k_w=3, rate=2, stddev=0.02,
                   name="conv2d", init_value=[], trainable=True, hasBias=True):
    # sc=scale_kernel*math.sqrt(2/(k_h*k_w*(input_.get_shape()[-1].value)* output_dim))
    
    with tf.variable_scope(name):
        # w=tf.Variable(sc*tf.random_normal([k_h, k_w, input_.get_shape()[-1].value, output_dim]),name='w')
        if (not (init_value)):
            w = tf.get_variable('w', [k_h, k_w, input_.get_shape()[-1], output_dim], trainable=trainable,
                                initializer=tf.contrib.layers.xavier_initializer())
        else:
            vname = tf.get_variable_scope()
            w = tf.Variable(init_value[vname.name + '/w:0'], name='w', trainable=trainable)
        _weight_decay(w)
        # conv = tf.nn.conv2d(input_, w, strides=[1, d_h, d_w, 1], padding='SAME')
        conv = tf.nn.atrous_conv2d(input_, w, rate, 'SAME')
        
        if hasBias:
            biases = tf.get_variable('biases', [output_dim], initializer=tf.constant_initializer(0.0))
            conv = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape())
        
        return conv


def deconv2d(input_, output_shape,
             k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02,
             name="deconv2d", with_w=False):
    with tf.variable_scope(name):
        # filter : [height, width, output_channels, in_channels]
        w = tf.get_variable('w', [k_h, k_w, output_shape[-1], input_.get_shape()[-1]],
                            initializer=tf.random_normal_initializer(stddev=stddev))
        _weight_decay(w)
        try:
            deconv = tf.nn.conv2d_transpose(input_, w, output_shape=output_shape,
                                            strides=[1, d_h, d_w, 1])
        
        # Support for verisons of TensorFlow before 0.7.0
        except AttributeError:
            deconv = tf.nn.deconv2d(input_, w, output_shape=output_shape,
                                    strides=[1, d_h, d_w, 1])
        
        biases = tf.get_variable('biases', [output_shape[-1]], initializer=tf.constant_initializer(0.0))
        deconv = tf.reshape(tf.nn.bias_add(deconv, biases), deconv.get_shape())
        
        if with_w:
            return deconv, w, biases
        else:
            return deconv


def mydeconv2d(input_, out_dim, outputshape, k_h=6, k_w=6, d_h=2, d_w=2, stddev=0.02,
               name="deconv2d", with_w=False, init_value=[], trainable=True):
    # sc = scale_kernel*math.sqrt(2 / (k_h * k_w * (input_.get_shape()[-1].value) * out_dim))
    with tf.variable_scope(name):
        # filter : [height, width, output_channels, in_channels]
        # w=tf.Variable(sc*tf.random_normal([k_h, k_w, out_dim, input_.get_shape()[-1].value]),name='w')
        if (not (init_value)):
            w = tf.get_variable('w', [k_h, k_w, out_dim, input_.get_shape()[-1]], trainable=trainable,
                                initializer=tf.contrib.layers.xavier_initializer())
        else:
            vname = tf.get_variable_scope()
            w = tf.Variable(init_value[vname.name + '/w:0'], name='w', trainable=trainable)
        _weight_decay(w)
        try:
            deconv = tf.nn.conv2d_transpose(input_, w, strides=[1, d_h, d_w, 1], output_shape=outputshape)
        
        # Support for verisons of TensorFlow before 0.7.0
        except AttributeError:
            deconv = tf.nn.deconv2d(input_, w, strides=[1, d_h, d_w, 1])
        
        biases = tf.get_variable('biases', [out_dim], initializer=tf.constant_initializer(0.0))
        deconv = tf.reshape(tf.nn.bias_add(deconv, biases), deconv.get_shape())
        
        if with_w:
            return deconv, w, biases
        else:
            return deconv


def lrelu(x, leak=0.2, name="lrelu"):
    return tf.maximum(x, leak * x)


def PReLU(_x, name):
    _alpha = tf.get_variable(name, shape=_x.get_shape()[-1], dtype=_x.dtype, initializer=tf.constant_initializer(0.2))
    return tf.maximum(0.0, _x) + _alpha * tf.minimum(0.0, _x)


def relu(x):
    return tf.maximum(x, 0)


def boxfilter(tensor, r, name):
    with tf.variable_scope(name):
        shape = tensor.get_shape()
        n = shape[0].value
        h = shape[1].value
        w = shape[2].value
        fn = shape[3].value
        tCum = tf.cumsum(tensor, 1)
        tDst = tf.Variable(tf.ones([n, h, w, fn]), False)
        # tDst=tDst[:,0:r+1,:,:].assign(tCum[:,r:2*r+1,:,:])
        op1 = tDst[:, 0:r + 1, :, :].assign(tCum[:, r:2 * r + 1, :, :])
        op2 = tDst[:, r + 1:h - r, :, :].assign(tf.subtract(tCum[:, 2 * r + 1:h, :, :], tCum[:, 0:h - 2 * r - 1, :, :]))
        op3 = tDst[:, h - r:h, :, :].assign(
            tf.subtract(tf.reshape(tCum[:, h - 1, :, :], [n, 1, w, fn]), tCum[:, h - 2 * r - 1:h - r - 1, :, :]))
        with tf.control_dependencies([tf.group(op1, op2, op3)]):
            tCum = tf.cumsum(tDst, 2)
            op4 = tDst[:, :, 0:r + 1, :].assign(tCum[:, :, r:2 * r + 1, :])
            op5 = tDst[:, :, r + 1:w - r, :].assign(
                tf.subtract(tCum[:, :, 2 * r + 1:w, :], tCum[:, :, 0:w - 2 * r - 1, :]))
            op6 = tDst[:, :, w - r:w, :].assign(
                tf.subtract(tf.reshape(tCum[:, :, w - 1, :], [n, h, 1, fn]), tCum[:, :, w - 2 * r - 1:w - r - 1, :]))
            with tf.control_dependencies([tf.group(op4, op5, op6)]):
                tf.no_op()
                return op6


def guidedFilter(I, p, name, r=5, eps=1000.0):
    with tf.variable_scope(name):
        shape = I.get_shape()
        h = shape[1].value
        w = shape[2].value
        N = boxfilter(tf.ones(shape), r, 'g1')
        mean_I = tf.divide(boxfilter(I, r, 'g2'), N)
        mean_p = tf.divide(boxfilter(p, r, 'g3'), N)
        mean_Ip = tf.divide(boxfilter(tf.multiply(I, p), r, 'g4'), N)
        cov_Ip = mean_Ip - tf.multiply(mean_I, mean_p)
        mean_II = tf.divide(boxfilter(tf.multiply(I, I), r, 'g5'), N)
        var_I = tf.add(eps, tf.subtract(mean_II, tf.multiply(mean_I, mean_I)))
        a = tf.divide(cov_Ip, var_I)
        b = tf.subtract(mean_p, tf.multiply(a, mean_I))
        mean_a = tf.divide(boxfilter(a, r, 'g6'), N)
        mean_b = tf.divide(boxfilter(b, r, 'g7'), N)
        
        q = tf.add(mean_b, tf.multiply(mean_a, I))
        return q


def deconv_anysize(h, name, f_size=64, ksize=6, stride=2):
    with tf.variable_scope(name):
        inputs_shape = h.get_shape()
        outputs_shape = [inputs_shape[0].value, inputs_shape[1].value * stride, inputs_shape[2].value * stride, f_size]
        h_deconv = mydeconv2d(h, f_size, outputs_shape, ksize, ksize, stride, stride, name="deconv1")
        return h_deconv


trainable = True


def _phase_shift(I, r):
    bsize, a, b, c = I.get_shape().as_list()
    bsize = tf.shape(I)[0]
    X = tf.reshape(I, (bsize, a, b, r, r))
    X = tf.transpose(X, (0, 1, 2, 4, 3))
    X = tf.split(X, a, 1)
    X = tf.concat([tf.squeeze(x, axis=1) for x in X], 2)  # bsize, b, a*r, r
    X = tf.split(X, b, 1)
    X = tf.concat([tf.squeeze(x, axis=1) for x in X], 2)
    return tf.reshape(X, (bsize, a * r, b * r, 1))


def PS(X, r, color=False):
    if color:
        Xc = tf.split(X, 3, 3)
        X = tf.concat([_phase_shift(x, r) for x in Xc], 3)
    else:
        X = _phase_shift(X, r)
    return X
