import tensorflow as tf
import numpy as np
from tensorflow.keras.metrics import binary_crossentropy
from sklearn.metrics import accuracy_score


def masked_MSE(mask):
    def loss(y_true, y_pred):

        y_pred_c = y_pred
        y_true_c = y_true

        y_pred = tf.convert_to_tensor(y_pred)
        y_true = tf.cast(y_true, y_pred.dtype)
        # print(y_pred.shape)
        # print(y_true.shape)
        sq_diff = tf.multiply(tf.math.squared_difference(y_pred, y_true), mask)
        # print(sq_diff)
        # print(mask)
        if np.isnan(tf.reduce_mean(sq_diff).numpy()):
            with open('test.npy', 'wb') as f:
                np.save(f, y_pred_c)
                np.save(f, y_true_c)
                np.save(f, mask)
                np.save(f, sq_diff)
        return tf.reduce_mean(sq_diff)
    return loss


# Used to debug when training on multiple GPUs resolved the first dimension of y_true for some reason
# def masked_binary_crossentropy(mask):
#     def loss(y_true, y_pred):
#         y_true_masked = np.array([tf.boolean_mask(arr, mask) for arr in y_true])
#         y_pred_masked = np.array([tf.boolean_mask(arr, mask) for arr in y_pred])
#         return binary_crossentropy(y_true_masked, y_pred_masked, from_logits=True)
#     return loss


def masked_binary_crossentropy(mask):
    def loss(y_true, y_pred):
        y_true_masked = tf.boolean_mask(y_true, mask, axis=1)
        y_pred_masked = tf.boolean_mask(y_pred, mask, axis=1)
        return tf.math.reduce_mean(binary_crossentropy(y_true_masked, y_pred_masked, from_logits=True))
    return loss


def masked_accuracy(mask):
    def loss(y_true, y_pred):
        y_true_masked = tf.boolean_mask(y_true, mask)
        y_pred_masked = tf.boolean_mask(y_pred, mask)
        return accuracy_score(y_true_masked, y_pred_masked)
    return loss


def adv_conv_res_thic_loss(mask):
    ''
    def loss(y_true, y_pred):
        y_true_masked = tf.boolean_mask(y_true, mask)
        y_pred_masked = tf.boolean_mask(y_pred, mask)
        return 0
    return loss
