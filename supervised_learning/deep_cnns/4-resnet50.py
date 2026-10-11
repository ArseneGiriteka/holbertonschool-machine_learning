#!/usr/bin/env python3
"""Module that builds the ResNet-50 architecture"""
from tensorflow import keras as K
identity_block = __import__('2-identity_block').identity_block
projection_block = __import__('3-projection_block').projection_block


def resnet50():
    """
    Builds the ResNet-50 architecture as described in Deep Residual
    Learning for Image Recognition (2015)

    Input data is assumed to have shape (224, 224, 3)

    Returns: the keras model
    """
    init = K.initializers.he_normal(seed=0)
    X_input = K.Input(shape=(224, 224, 3))

    # conv1
    X = K.layers.Conv2D(filters=64, kernel_size=(7, 7), strides=(2, 2),
                        padding='same', kernel_initializer=init)(X_input)
    X = K.layers.BatchNormalization(axis=3)(X)
    X = K.layers.Activation('relu')(X)
    X = K.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2),
                              padding='same')(X)

    # conv2_x
    X = projection_block(X, [64, 64, 256], s=1)
    for _ in range(2):
        X = identity_block(X, [64, 64, 256])

    # conv3_x
    X = projection_block(X, [128, 128, 512])
    for _ in range(3):
        X = identity_block(X, [128, 128, 512])

    # conv4_x
    X = projection_block(X, [256, 256, 1024])
    for _ in range(5):
        X = identity_block(X, [256, 256, 1024])

    # conv5_x
    X = projection_block(X, [512, 512, 2048])
    for _ in range(2):
        X = identity_block(X, [512, 512, 2048])

    X = K.layers.AveragePooling2D(pool_size=(7, 7), strides=(1, 1))(X)
    X = K.layers.Dense(units=1000, activation='softmax',
                       kernel_initializer=init)(X)

    return K.models.Model(inputs=X_input, outputs=X)
