#!/usr/bin/env python3
"""Module that defines the NST class for neural style transfer"""
import numpy as np
import tensorflow as tf


class NST:
    """Class that performs tasks for neural style transfer"""

    style_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1',
                    'block4_conv1', 'block5_conv1']
    content_layer = 'block5_conv2'

    def __init__(self, style_image, content_image, alpha=1e4, beta=1):
        """Class constructor

        style_image: numpy.ndarray (h, w, 3) used as a style reference
        content_image: numpy.ndarray (h, w, 3) used as a content reference
        alpha: weight for content cost
        beta: weight for style cost
        """
        if (not isinstance(style_image, np.ndarray) or
                style_image.ndim != 3 or style_image.shape[2] != 3):
            raise TypeError(
                "style_image must be a numpy.ndarray with shape (h, w, 3)")
        if (not isinstance(content_image, np.ndarray) or
                content_image.ndim != 3 or content_image.shape[2] != 3):
            raise TypeError(
                "content_image must be a numpy.ndarray with shape (h, w, 3)")
        if (isinstance(alpha, bool) or
                not isinstance(alpha, (int, float)) or alpha < 0):
            raise TypeError("alpha must be a non-negative number")
        if (isinstance(beta, bool) or
                not isinstance(beta, (int, float)) or beta < 0):
            raise TypeError("beta must be a non-negative number")

        self.style_image = self.scale_image(style_image)
        self.content_image = self.scale_image(content_image)
        self.alpha = alpha
        self.beta = beta

    @staticmethod
    def scale_image(image):
        """Rescales an image so its pixels are in [0, 1] and its largest
        side is 512 pixels

        image: numpy.ndarray of shape (h, w, 3) containing the image

        Returns: the scaled image as a tf.Tensor of shape
            (1, h_new, w_new, 3)
        """
        if (not isinstance(image, np.ndarray) or
                image.ndim != 3 or image.shape[2] != 3):
            raise TypeError(
                "image must be a numpy.ndarray with shape (h, w, 3)")

        h, w, _ = image.shape
        if h > w:
            h_new = 512
            w_new = int(w * 512 / h)
        else:
            w_new = 512
            h_new = int(h * 512 / w)

        image = tf.expand_dims(image, axis=0)
        image = tf.image.resize(image, (h_new, w_new), method='bicubic')
        image = image / 255
        image = tf.clip_by_value(image, 0, 1)

        return image
