#!/usr/bin/env python3
"""
Defines the Yolo class that uses the Yolo v3 algorithm
to perform object detection
"""
import numpy as np
import tensorflow.keras as K


class Yolo:
    """
    Uses the Yolo v3 algorithm to perform object detection
    """

    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        """
        Class constructor

        model_path: path to where a Darknet Keras model is stored
        classes_path: path to the list of class names used for the
            Darknet model, listed in order of index
        class_t: float representing the box score threshold for the
            initial filtering step
        nms_t: float representing the IOU threshold for non-max suppression
        anchors: numpy.ndarray of shape (outputs, anchor_boxes, 2)
            containing all of the anchor boxes
        """
        self.model = K.models.load_model(model_path)
        with open(classes_path, 'r') as f:
            self.class_names = [line.strip() for line in f
                                if line.strip()]
        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors

    @staticmethod
    def sigmoid(x):
        """
        Computes the sigmoid function of x
        """
        return 1 / (1 + np.exp(-x))

    def process_outputs(self, outputs, image_size):
        """
        Processes the outputs of the Darknet model for a single image

        outputs: list of numpy.ndarrays of shape
            (grid_height, grid_width, anchor_boxes, 4 + 1 + classes)
            containing the predictions of the Darknet model
        image_size: numpy.ndarray containing the image's original size
            [image_height, image_width]

        Returns: tuple of (boxes, box_confidences, box_class_probs)
            boxes: list of numpy.ndarrays of shape
                (grid_height, grid_width, anchor_boxes, 4) containing the
                boundary boxes (x1, y1, x2, y2) relative to original image
            box_confidences: list of numpy.ndarrays of shape
                (grid_height, grid_width, anchor_boxes, 1)
            box_class_probs: list of numpy.ndarrays of shape
                (grid_height, grid_width, anchor_boxes, classes)
        """
        image_h, image_w = image_size
        input_h = self.model.input.shape[1]
        input_w = self.model.input.shape[2]
        boxes, box_confidences, box_class_probs = [], [], []

        for i, output in enumerate(outputs):
            grid_h, grid_w, anchor_boxes = output.shape[:3]
            t_x = output[..., 0]
            t_y = output[..., 1]
            t_w = output[..., 2]
            t_h = output[..., 3]

            c_x = np.arange(grid_w).reshape(1, grid_w, 1)
            c_y = np.arange(grid_h).reshape(grid_h, 1, 1)
            p_w = self.anchors[i, :, 0]
            p_h = self.anchors[i, :, 1]

            b_x = (self.sigmoid(t_x) + c_x) / grid_w
            b_y = (self.sigmoid(t_y) + c_y) / grid_h
            b_w = p_w * np.exp(t_w) / input_w
            b_h = p_h * np.exp(t_h) / input_h

            box = np.zeros(output[..., :4].shape)
            box[..., 0] = (b_x - b_w / 2) * image_w
            box[..., 1] = (b_y - b_h / 2) * image_h
            box[..., 2] = (b_x + b_w / 2) * image_w
            box[..., 3] = (b_y + b_h / 2) * image_h
            boxes.append(box)

            box_confidences.append(self.sigmoid(output[..., 4:5]))
            box_class_probs.append(self.sigmoid(output[..., 5:]))

        return boxes, box_confidences, box_class_probs
