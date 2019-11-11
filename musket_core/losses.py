import tensorflow as tf
from keras import backend as K
import numpy as np
import keras
import tqdm
# credits: https://www.kaggle.com/guglielmocamporese/macro-f1-score-keras


def keras_loss(func):
    keras.utils.get_custom_objects()[func.__name__]=func
    return func

def macro_f1(y_true, y_pred):
    # y_pred = K.round(y_pred)
    # y_pred = K.cast(K.greater(K.clip(y_pred, 0, 1), THRESHOLD), K.floatx())
    tp = K.sum(K.cast(y_true * y_pred, 'float'), axis=0)
    tn = K.sum(K.cast((1 - y_true) * (1 - y_pred), 'float'), axis=0)
    fp = K.sum(K.cast((1 - y_true) * y_pred, 'float'), axis=0)
    fn = K.sum(K.cast(y_true * (1 - y_pred), 'float'), axis=0)

    p = tp / (tp + fp + K.epsilon())
    r = tp / (tp + fn + K.epsilon())
    r = tp / (tp + fn + K.epsilon())

    f1 = 2 * p * r / (p + r + K.epsilon())
    f1 = tf.where(tf.is_nan(f1), tf.zeros_like(f1), f1)
    return K.mean(f1)

def matthews_correlation(y_true, y_pred):
    '''Calculates the Matthews correlation coefficient measure for quality
    of binary classification problems.
    '''
    y_pred_pos = K.round(K.clip(y_pred, 0, 1))
    y_pred_neg = 1 - y_pred_pos

    y_pos = K.round(K.clip(y_true, 0, 1))
    y_neg = 1 - y_pos

    tp = K.sum(y_pos * y_pred_pos)
    tn = K.sum(y_neg * y_pred_neg)

    fp = K.sum(y_neg * y_pred_pos)
    fn = K.sum(y_pos * y_pred_neg)

    numerator = (tp * tn - fp * fn)
    denominator = K.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))

    return numerator / (denominator + K.epsilon())

def l2_loss(y_true, y_pred):
    diff = y_true - y_pred

    return K.sum(diff * diff)

def f1_loss(y_true, y_pred):
    tp = K.sum(K.cast(y_true * y_pred, 'float'), axis=0)
    tn = K.sum(K.cast((1 - y_true) * (1 - y_pred), 'float'), axis=0)
    fp = K.sum(K.cast((1 - y_true) * y_pred, 'float'), axis=0)
    fn = K.sum(K.cast(y_true * (1 - y_pred), 'float'), axis=0)

    p = tp / (tp + fp + K.epsilon())
    r = tp / (tp + fn + K.epsilon())

    f1 = 2 * p * r / (p + r + K.epsilon())
    f1 = tf.where(tf.is_nan(f1), tf.zeros_like(f1), f1)
    return 1 - K.mean(f1)

def Mean_IOU(y_true, y_pred):
    nb_classes = K.int_shape(y_pred)[-1]
    iou = []
    true_pixels = K.argmax(y_true, axis=-1)
    pred_pixels = K.argmax(y_pred, axis=-1)
    void_labels = K.equal(K.sum(y_true, axis=-1), 0)
    for i in range(0, nb_classes): # exclude first label (background) and last label (void)
        true_labels = K.equal(true_pixels, i) & ~void_labels
        pred_labels = K.equal(pred_pixels, i) & ~void_labels
        inter = tf.to_int32(true_labels & pred_labels)
        union = tf.to_int32(true_labels | pred_labels)
        legal_batches = K.sum(tf.to_int32(true_labels), axis=1)>0
        ious = K.sum(inter, axis=1)/K.sum(union, axis=1)
        iou.append(K.mean(tf.gather(ious, indices=tf.where(legal_batches)))) # returns average IoU of the same objects
    iou = tf.stack(iou)
    legal_labels = ~tf.debugging.is_nan(iou)
    iou = tf.gather(iou, indices=tf.where(legal_labels))
    return K.mean(iou)

def iot_coef(y_true, y_pred, smooth=1):
    """
    IoU = (|X & Y|)/ (|X or Y|)
    """
    y_pred=tf.to_int32(y_pred>0.5)
    y_true = tf.to_int32(y_true> 0.5)

    intersection = K.sum(y_true * y_pred, axis=[1, 2, 3])
    union = K.sum(y_true, axis=[1, 2, 3]) + K.sum(y_pred, axis=[1, 2, 3])
    return 2.0*K.mean((intersection + smooth) / (union + smooth), axis=0)

def iou_coef(y_true, y_pred, smooth=1):
    """
    IoU = (|X & Y|)/ (|X or Y|)
    """
    intersection = K.sum(y_true * y_pred, axis=[1, 2, 3])
    union = K.sum(y_true, axis=[1, 2, 3]) + K.sum(y_pred, axis=[1, 2, 3])
    return 2.0*K.mean((intersection + smooth) / (union + smooth), axis=0)
    # intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    # union = K.sum((y_true,-1) + K.sum(y_pred,-1)) - intersection
    # return (intersection + smooth) / ( union + smooth)

SMOOTH = 1e-6    

def binary_accuracy_numpy(x,y):
    vl=((x>0.5)==(y>0.5)).sum()
    return vl/y.size 




# Numpy version
# Well, it's the same function, so I'm going to omit the comments



def iou_numpy(outputs, labels, smooth=1,negativesValue=1):
    """
    IoU = (|X & Y|)/ (|X or Y|)
    """
    cs=labels.shape[-1]
    if cs>1:
        rr=[]
        for i in range(cs):
            rr.append(iou_numpy(outputs[:,:,i:i+1], labels[:,:,i:i+1],negativesValue))
        return np.mean(rr, axis=0)
    outputs = outputs.squeeze()>0.5
    labels = labels.squeeze()>0.5
    if labels.max()==0:
        if (outputs>0.5).max()==0:
            return negativesValue
    intersection = (outputs & labels).sum()
    union = (outputs | labels).sum()
    
    iou = (intersection + SMOOTH) / (union + SMOOTH)
    return iou

def dice_numpy( outputs,labels,negativesValue=1, threshold=0.5):

    cs=labels.shape[-1]
    if cs>1:
        rr=[]
        for i in range(cs):
            rr.append(dice_numpy(outputs[:,:,i:i+1], labels[:,:,i:i+1],negativesValue))
        return np.mean(rr, axis=0)
    outputs = outputs.squeeze()
    labels = labels.squeeze()
    true = (outputs>threshold)
    pred = (labels>threshold)
    
    if labels.max()==0:
        if (outputs>threshold).max()==0:
            return negativesValue
          
    intersection =np.sum (true & pred)
    im_sum = np.sum(true) + np.sum(pred)

    return 2.0 * intersection / (im_sum + SMOOTH)
    
def dice_numpy_mask( outputs,labels,negativesValue=1, threshold=0.5):

    if len(labels.shape) > 2:
        raise ValueError("Max 2 dimensions supported");
    outputs = outputs.squeeze()
    labels = labels.squeeze()
    true = (outputs>threshold)
    pred = (labels>threshold)
    
    if labels.max()==0:
        if (outputs>threshold).max()==0:
            return negativesValue
          
    intersection =np.sum (true & pred)
    im_sum = np.sum(true) + np.sum(pred)

    return 2.0 * intersection / (im_sum + SMOOTH)        

def dice_numpy_true_negative_is_one( outputs,labels):    
    return dice_numpy(outputs,labels,1)

def dice_numpy_skip_true_negative( outputs,labels):
    return dice_numpy(outputs,labels,None)

def dice_numpy_true_negative_is_zero( outputs,labels):
    return dice_numpy(outputs,labels,0)

def iou_numpy_true_negative_is_one( outputs,labels):    
    return iou_numpy(outputs,labels,1)

def iou_numpy_skip_true_negative( outputs,labels):
    return iou_numpy(outputs,labels,None)

def iou_numpy_true_negative_is_zero( outputs,labels):
    return iou_numpy(outputs,labels,0)
    

def dice(true, pred):
    true = tf.to_float(true>0.5)
    pred = tf.to_float(pred>0.5)

    intersection =K.sum (true * pred)
    im_sum = K.sum(true) + K.sum(pred)

    return 2.0 * intersection / (im_sum + EPS)

def iou_coef_loss(y_true, y_pred):
    return -iou_coef(y_true, y_pred)

def jaccard_distance_loss(y_true, y_pred, smooth=100):
    """
    Jaccard = (|X & Y|)/ (|X|+ |Y| - |X & Y|)
            = sum(|A*B|)/(sum(|A|)+sum(|B|)-sum(|A*B|))

    The jaccard distance loss is usefull for unbalanced datasets. This has been
    shifted so it converges on 0 and is smoothed to avoid exploding or disapearing
    gradient.

    Ref: https://en.wikipedia.org/wiki/Jaccard_index

    @url: https://gist.github.com/wassname/f1452b748efcbeb4cb9b1d059dce6f96
    @author: wassname
    """
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    sum_ = K.sum(K.abs(y_true) + K.abs(y_pred), axis=-1)
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    return (1 - jac) * smooth
EPS = 1e-10



def dice_all(true, pred):
    return K.mean([dice(t, p) for t, p in zip(true, pred)])

def dice_coef(y_true, y_pred, smooth=1):
    """
    Dice = (2*|X & Y|)/ (|X|+ |Y|)
         =  2*sum(|A*B|)/(sum(A^2)+sum(B^2))
    ref: https://arxiv.org/pdf/1606.04797v1.pdf
    """
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    return (2. * intersection + smooth) / (K.sum(K.square(y_true),-1) + K.sum(K.square(y_pred),-1) + smooth)

def dice_coef_loss(y_true, y_pred):
    return 1-dice_coef(y_true, y_pred)

def log_loss(y_true, y_pred):
    '''
    expects probability-like input, e.g. softmax output
    '''
    y_pred = y_pred / K.expand_dims(K.sum(y_pred, axis=1))
    epsilon = 10.**(-15)
    ub = 1.0 - epsilon
    lb = epsilon
    y = K.maximum(K.minimum(y_pred,ub),lb)
    logs = K.log(y)
    components = y_true * logs
    sum = K.sum(components)
    result = sum * -1.
    result /= K.cast(K.shape(y_true)[0], "float")
    return result

log_loss.need_threshold = False

bce = keras.metrics.get("binary_crossentropy")
bce.need_threshold = False

# code download from: https://github.com/bermanmaxim/LovaszSoftmax
def lovasz_grad(gt_sorted):
    """
    Computes gradient of the Lovasz extension w.r.t sorted errors
    See Alg. 1 in paper
    """
    gts = tf.reduce_sum(gt_sorted)
    intersection = gts - tf.cumsum(gt_sorted)
    union = gts + tf.cumsum(1. - gt_sorted)
    jaccard = 1. - intersection / union
    jaccard = tf.concat((jaccard[0:1], jaccard[1:] - jaccard[:-1]), 0)
    return jaccard



def lovasz_hinge(logits, labels, per_image=True, ignore=None):
    """
    Binary Lovasz hinge loss
      logits: [B, H, W] Variable, logits at each pixel (between -\infty and +\infty)
      labels: [B, H, W] Tensor, binary ground truth masks (0 or 1)
      per_image: compute the loss per image instead of per batch
      ignore: void class id
    """
    if per_image:
        def treat_image(log_lab):
            log, lab = log_lab
            log, lab = tf.expand_dims(log, 0), tf.expand_dims(lab, 0)
            log, lab = flatten_binary_scores(log, lab, ignore)
            return lovasz_hinge_flat(log, lab)
        losses = tf.map_fn(treat_image, (logits, labels), dtype=tf.float32)
        loss = tf.reduce_mean(losses)
    else:
        loss = lovasz_hinge_flat(*flatten_binary_scores(logits, labels, ignore))
    return loss

def lovasz_hinge_flat(logits, labels):
    """
    Binary Lovasz hinge loss
      logits: [P] Variable, logits at each prediction (between -\infty and +\infty)
      labels: [P] Tensor, binary ground truth labels (0 or 1)
      ignore: label to ignore
    """

    def compute_loss():
        labelsf = tf.cast(labels, logits.dtype)
        signs = 2. * labelsf - 1.
        errors = 1. - logits * tf.stop_gradient(signs)
        errors_sorted, perm = tf.nn.top_k(errors, k=tf.shape(errors)[0], name="descending_sort")
        gt_sorted = tf.gather(labelsf, perm)
        grad = lovasz_grad(gt_sorted)
        loss = tf.tensordot(tf.nn.elu(errors_sorted), tf.stop_gradient(grad), 1, name="loss_non_void")
        return loss

    # deal with the void prediction case (only void pixels)
    loss = tf.cond(tf.equal(tf.shape(logits)[0], 0),
                   lambda: tf.reduce_sum(logits) * 0.,
                   compute_loss,
                   strict=True,
                   name="loss"
                   )
    return loss

def flatten_binary_scores(scores, labels, ignore=None):
    """
    Flattens predictions in the batch (binary case)
    Remove labels equal to 'ignore'
    """
    scores = tf.reshape(scores, (-1,))
    labels = tf.reshape(labels, (-1,))
    if ignore is None:
        return scores, labels
    valid = tf.not_equal(labels, ignore)
    vscores = tf.boolean_mask(scores, valid, name='valid_scores')
    vlabels = tf.boolean_mask(labels, valid, name='valid_labels')
    return vscores, vlabels

def lovasz_loss(y_true, y_pred):
    y_true, y_pred = K.cast(K.squeeze(y_true, -1), 'int32'), K.cast(K.squeeze(y_pred, -1), 'float32')
    #logits = K.log(y_pred / (1. - y_pred))
    logits = y_pred #Jiaxin
    loss = lovasz_hinge(logits, y_true, per_image = True, ignore = None)
    return loss

# MIT License
#
# Copyright (c) 2017 Muhammed Kocabas
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE
import tensorflow as tf
from keras import backend as K
'''
Compatible with tensorflow backend
'''

def focal_loss(y_true, y_pred):
    gamma=2#0.75
    alpha=0.25

    yc = tf.clip_by_value(y_pred, 1e-15, 1 - 1e-15)
    pt_1 = tf.where(tf.equal(y_true, 1), yc, tf.ones_like(yc))
    pt_0 = tf.where(tf.equal(y_true, 0), yc, tf.zeros_like(yc))
    return (-K.sum(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1)) - K.sum(
        (1 - alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0)))

def plus(s1,f1,s2,f2):
    def impl(x,y):
        return s1*f1(x,y)+s2*f2(x,y)
    return impl

def composite_loss(s:str):
    res=s.split("+")
    ops=[]
    for arg in res:
        elm=arg.split("*")

        scales=1.0
        func=None
        for v in elm:
            if v[0].isalpha():
                if func is not None:
                    raise ValueError("Only one member per component")
                func=v
            if v[0].isdecimal():
                scales=scales*float(v)
        func=keras.losses.get(func)
        ops.append((scales,func))

    def loss(x,y):
        fr=None
        for v in ops:
            val=v[1](x,y)*v[0]
            if fr is None:
                fr=val
            else:
                fr=fr+val
        return fr
    return loss


def crf_nll(y_true, y_pred):
    """The negative log-likelihood for linear chain Conditional Random Field (CRF).
    This loss function is only used when the `crf.CRF` layer
    is trained in the "join" mode.
    # Arguments
        y_true: tensor with true targets.
        y_pred: tensor with predicted targets.
    # Returns
        A scalar representing corresponding to the negative log-likelihood.
    # Raises
        TypeError: If CRF is not the last layer.
    # About GitHub
        If you open an issue or a pull request about CRF, please
        add `cc @lzfelix` to notify Luiz Felix.
    """

    crf, idx = y_pred._keras_history[:2]
    if crf._outbound_nodes:
        raise TypeError('When learn_model="join", CRF must be the last layer.')
    if crf.sparse_target:
        y_true = K.one_hot(K.cast(y_true[:, :, 0], 'int32'), crf.units)
    X = crf._inbound_nodes[idx].input_tensors[0]
    mask = crf._inbound_nodes[idx].input_masks[0]
    nloglik = crf.get_negative_log_likelihood(y_true, X, mask)
    return nloglik


def crf_loss(y_true, y_pred):
    """General CRF loss function depending on the learning mode.
    # Arguments
        y_true: tensor with true targets.
        y_pred: tensor with predicted targets.
    # Returns
        If the CRF layer is being trained in the join mode, returns the negative
        log-likelihood. Otherwise returns the categorical crossentropy implemented
        by the underlying Keras backend.
    # About GitHub
        If you open an issue or a pull request about CRF, please
        add `cc @lzfelix` to notify Luiz Felix.
    """
    crf, idx = y_pred._keras_history[:2]
    if crf.learn_mode == 'join':
        return crf_nll(y_true, y_pred)
    else:
        if crf.sparse_target:
            return keras.losses.sparse_categorical_crossentropy(y_true, y_pred)
        else:
            return keras.losses.categorical_crossentropy(y_true, y_pred)
        
keras.utils.get_custom_objects()["crf_loss"]=crf_loss