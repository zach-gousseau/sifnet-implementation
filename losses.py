import tensorflow as tf
from tensorflow.keras.metrics import binary_crossentropy
from sklearn.metrics import accuracy_score

def masked_MSE(mask):
    def loss(y_true, y_pred):
        y_pred = tf.convert_to_tensor(y_pred)
        y_true = tf.cast(y_true, y_pred.dtype)
        sq_diff = tf.multiply(tf.math.squared_difference(y_pred, y_true), mask)
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
        y_true_masked = tf.boolean_mask(y_true, mask)
        y_pred_masked = tf.boolean_mask(y_pred, mask)
        return binary_crossentropy(y_true_masked, y_pred_masked, from_logits=True)
    return loss

def masked_accuracy(mask):
    def loss(y_true, y_pred):
        y_true_masked = tf.boolean_mask(y_true, mask)
        y_pred_masked = tf.boolean_mask(y_pred, mask)
        return accuracy_score(y_true_masked, y_pred_masked)
    return loss