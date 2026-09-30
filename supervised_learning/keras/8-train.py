#!/usr/bin/env python3
"""Trains a model using mini-batch gradient descent"""
import tensorflow.keras as K


def train_model(
        network, data, labels, batch_size, epochs,
        validation_data=None, early_stopping=False, patience=0,
        learning_rate_decay=False, alpha=0.1, decay_rate=1,
        save_best=False, filepath=None,
        verbose=True, shuffle=False):
    """Trains a model using mini-batch gradient descent

    Args:
        network: the model to train
        data: numpy.ndarray of shape (m, nx) containing the input data
        labels: one-hot numpy.ndarray of shape (m, classes) containing
            the labels of data
        batch_size: size of the batch used for mini-batch gradient descent
        epochs: number of passes through data for mini-batch gradient
            descent
        validation_data: data to validate the model with, if not None
        early_stopping: boolean that indicates whether early stopping
            should be used; only performed if validation_data exists
            and is based on validation loss
        patience: the patience used for early stopping
        learning_rate_decay: boolean that indicates whether learning
            rate decay should be used; only performed if validation_data
            exists, using stepwise inverse time decay
        alpha: the initial learning rate
        decay_rate: the decay rate
        save_best: boolean indicating whether to save the model after
            each epoch if it is the best, based on the lowest
            validation loss obtained
        filepath: the file path where the model should be saved
        verbose: boolean that determines if output should be printed
            during training
        shuffle: boolean that determines whether to shuffle the batches
            every epoch

    Returns:
        the History object generated after training the model
    """
    callbacks = []
    if early_stopping and validation_data is not None:
        callbacks.append(K.callbacks.EarlyStopping(
            monitor='val_loss', patience=patience))

    if learning_rate_decay and validation_data is not None:
        def scheduler(epoch):
            """Computes the learning rate using inverse time decay"""
            return alpha / (1 + decay_rate * epoch)

        callbacks.append(K.callbacks.LearningRateScheduler(
            scheduler, verbose=1))

    if save_best and validation_data is not None:
        callbacks.append(K.callbacks.ModelCheckpoint(
            filepath, monitor='val_loss', save_best_only=True))

    history = network.fit(
        data, labels, batch_size=batch_size,
        epochs=epochs, validation_data=validation_data,
        callbacks=callbacks, verbose=verbose, shuffle=shuffle)

    return history
