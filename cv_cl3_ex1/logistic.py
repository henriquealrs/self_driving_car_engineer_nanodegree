import tensorflow as tf
import numpy as np
import utils


def softmax(logits):
    """
    softmax implementation
    args:
    - logits [tensor]: 1xN logits tensor
    returns:
    - soft_logits [tensor]: softmax of logits
    """
    # IMPLEMENT THIS FUNCTION
    exp = tf.exp(logits)
    sum = tf.reduce_sum(exp, axis=1)
    print(f'\n{sum}')
    print(f'\n Reshape sum: {tf.reshape(sum, (-1, 1))}')
    soft_logits =  exp / tf.reshape(sum, (-1, 1))
    return soft_logits


def cross_entropy(scaled_logits, one_hot):
    """
    Cross entropy loss implementation
    args:
    - scaled_logits [tensor]: NxC tensor where N batch size / C number of classes
    - one_hot [tensor]: one hot tensor
    returns:
    - loss [tensor]: cross entropy 
    """
    masked = tf.boolean_mask(scaled_logits, one_hot, axis=0)
    print(f'\n{masked}')
    ce = -1 * tf.math.log(masked)
    return ce


def model(X: tf.Tensor, W: tf.Tensor, b: tf.Tensor):
    """
    logistic regression model
    args:
    - X [tensor]: input HxWx3
    - W [tensor]: weights
    - b [tensor]: bias
    returns:
    - output [tensor]
    """
    X_trans = tf.reshape(X, (1, X.shape[0] * X.shape[1] * X.shape[2]))
    temp = tf.transpose(tf.matmul(X_trans, W))
    y = temp + tf.reshape(b, (-1,1))
    return softmax(tf.transpose(y))


def accuracy(y_hat, Y):
    """
    calculate accuracy
    args:
    - y_hat [tensor]: NxC tensor of models predictions
    - y [tensor]: N tensor of ground truth classes
    returns:
    - acc [tensor]: accuracy
    """
    predictions = tf.argmax(y_hat, axis=1)
    print(predictions)
    n_predictions = Y.shape[0]
    print(y_hat, Y)
    correct = tf.math.equal(Y, tf.cast(predictions, dtype=tf.int32))
    print(correct)
    acc = tf.reduce_sum(tf.cast(correct, dtype=tf.int32)) / n_predictions
    return acc

if __name__ == '__main__':
    utils.check_softmax(softmax)
    utils.check_ce(cross_entropy)
    utils.check_model(model)
    utils.check_acc(accuracy)
