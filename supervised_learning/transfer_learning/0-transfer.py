#!/usr/bin/env python3
"""
Trains a CIFAR-10 classifier using transfer learning from ResNet50
(Keras Applications, pre-trained on ImageNet) and saves it as cifar10.h5
"""
from tensorflow import keras as K

IMG_SIZE = 224
HEAD_EPOCHS = 20
FINE_TUNE_EPOCHS = 2
BATCH_SIZE = 128


def preprocess_data(X, Y):
    """
    Pre-processes the CIFAR-10 data for the model

    X: numpy.ndarray of shape (m, 32, 32, 3) containing the CIFAR 10 data
    Y: numpy.ndarray of shape (m,) containing the CIFAR 10 labels for X

    Returns: X_p, Y_p
        X_p: numpy.ndarray containing the preprocessed X
        Y_p: numpy.ndarray containing the one-hot encoded labels
    """
    X_p = K.applications.resnet50.preprocess_input(X.astype('float32'))
    Y_p = K.utils.to_categorical(Y, 10)
    return X_p, Y_p


def build_head(input_shape):
    """
    Builds the trainable classification head placed on top of the
    frozen ResNet50 features

    input_shape: shape of the pooled features produced by ResNet50

    Returns: the head as a keras model
    """
    inputs = K.Input(shape=input_shape)
    X = K.layers.Dense(256, activation='relu',
                       kernel_initializer=K.initializers.he_normal())(inputs)
    X = K.layers.BatchNormalization()(X)
    X = K.layers.Dropout(0.4)(X)
    outputs = K.layers.Dense(10, activation='softmax')(X)
    return K.models.Model(inputs, outputs, name='head')


def main():
    """Loads CIFAR-10, trains the model and saves it as cifar10.h5"""
    (X_train, Y_train), (X_test, Y_test) = K.datasets.cifar10.load_data()
    X_train, Y_train = preprocess_data(X_train, Y_train)
    X_test, Y_test = preprocess_data(X_test, Y_test)

    base = K.applications.ResNet50(include_top=False, weights='imagenet',
                                   input_shape=(IMG_SIZE, IMG_SIZE, 3),
                                   pooling='avg')
    base.trainable = False

    inputs = K.Input(shape=(32, 32, 3))
    resized = K.layers.Resizing(IMG_SIZE, IMG_SIZE)(inputs)
    features = base(resized, training=False)
    extractor = K.models.Model(inputs, features)

    # The frozen layers always give the same output: compute it only once
    F_train = extractor.predict(X_train, batch_size=BATCH_SIZE, verbose=1)
    F_test = extractor.predict(X_test, batch_size=BATCH_SIZE, verbose=1)

    head = build_head(F_train.shape[1:])
    head.compile(optimizer=K.optimizers.Adam(1e-3),
                 loss='categorical_crossentropy', metrics=['accuracy'])
    head.fit(F_train, Y_train, batch_size=BATCH_SIZE, epochs=HEAD_EPOCHS,
             validation_data=(F_test, Y_test), verbose=1,
             callbacks=[K.callbacks.EarlyStopping(monitor='val_accuracy',
                                                  patience=4,
                                                  restore_best_weights=True)])

    model = K.models.Model(inputs, head(features))

    # Short fine-tuning of the last ResNet stage (conv5) at a low rate,
    # BatchNorm layers stay frozen (base is called with training=False)
    if FINE_TUNE_EPOCHS > 0:
        base.trainable = True
        tune = False
        for layer in base.layers:
            if layer.name.startswith('conv5_block1'):
                tune = True
            is_bn = isinstance(layer, K.layers.BatchNormalization)
            layer.trainable = tune and not is_bn
        model.compile(optimizer=K.optimizers.Adam(1e-5),
                      loss='categorical_crossentropy', metrics=['accuracy'])
        model.fit(X_train, Y_train, batch_size=64, epochs=FINE_TUNE_EPOCHS,
                  validation_data=(X_test, Y_test), verbose=1)

    model.compile(optimizer=K.optimizers.Adam(1e-5),
                  loss='categorical_crossentropy', metrics=['accuracy'])
    model.evaluate(X_test, Y_test, batch_size=BATCH_SIZE, verbose=1)
    model.save('cifar10.h5')


if __name__ == '__main__':
    main()
