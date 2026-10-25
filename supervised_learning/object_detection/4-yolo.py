#!/usr/bin/env python3
"""
Defines the Yolo class that uses the Yolo v3 algorithm
to perform object detection
"""
import glob
import os
import cv2
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

    def filter_boxes(self, boxes, box_confidences, box_class_probs):
        """
        Filters the boundary boxes using the box score threshold

        boxes: list of numpy.ndarrays of shape
            (grid_height, grid_width, anchor_boxes, 4) containing the
            processed boundary boxes for each output
        box_confidences: list of numpy.ndarrays of shape
            (grid_height, grid_width, anchor_boxes, 1) containing the
            processed box confidences for each output
        box_class_probs: list of numpy.ndarrays of shape
            (grid_height, grid_width, anchor_boxes, classes) containing the
            processed box class probabilities for each output

        Returns: tuple of (filtered_boxes, box_classes, box_scores)
            filtered_boxes: numpy.ndarray of shape (?, 4) containing all of
                the filtered bounding boxes
            box_classes: numpy.ndarray of shape (?,) containing the class
                number that each box in filtered_boxes predicts
            box_scores: numpy.ndarray of shape (?) containing the box scores
                for each box in filtered_boxes
        """
        filtered_boxes, box_classes, box_scores = [], [], []

        for box, confidence, class_probs in zip(boxes, box_confidences,
                                                box_class_probs):
            scores = confidence * class_probs
            classes = np.argmax(scores, axis=-1).reshape(-1)
            max_scores = np.max(scores, axis=-1).reshape(-1)
            mask = max_scores >= self.class_t

            filtered_boxes.append(box.reshape(-1, 4)[mask])
            box_classes.append(classes[mask])
            box_scores.append(max_scores[mask])

        filtered_boxes = np.concatenate(filtered_boxes)
        box_classes = np.concatenate(box_classes)
        box_scores = np.concatenate(box_scores)

        return filtered_boxes, box_classes, box_scores

    def non_max_suppression(self, filtered_boxes, box_classes, box_scores):
        """
        Applies non-max suppression to the filtered boundary boxes

        filtered_boxes: numpy.ndarray of shape (?, 4) containing all of the
            filtered bounding boxes
        box_classes: numpy.ndarray of shape (?,) containing the class number
            for the class that filtered_boxes predicts
        box_scores: numpy.ndarray of shape (?) containing the box scores for
            each box in filtered_boxes

        Returns: tuple of
            (box_predictions, predicted_box_classes, predicted_box_scores)
            box_predictions: numpy.ndarray of shape (?, 4) containing all of
                the predicted bounding boxes ordered by class and box score
            predicted_box_classes: numpy.ndarray of shape (?,) containing the
                class number for box_predictions ordered by class and score
            predicted_box_scores: numpy.ndarray of shape (?) containing the
                box scores for box_predictions ordered by class and score
        """
        box_predictions, predicted_classes, predicted_scores = [], [], []

        for cls in np.unique(box_classes):
            idx = np.where(box_classes == cls)[0]
            boxes = filtered_boxes[idx]
            scores = box_scores[idx]
            order = np.argsort(scores)[::-1]
            boxes = boxes[order]
            scores = scores[order]

            areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
            remaining = np.arange(len(boxes))
            while remaining.size > 0:
                best = remaining[0]
                box_predictions.append(boxes[best])
                predicted_classes.append(cls)
                predicted_scores.append(scores[best])

                rest = remaining[1:]
                x1 = np.maximum(boxes[best, 0], boxes[rest, 0])
                y1 = np.maximum(boxes[best, 1], boxes[rest, 1])
                x2 = np.minimum(boxes[best, 2], boxes[rest, 2])
                y2 = np.minimum(boxes[best, 3], boxes[rest, 3])
                inter = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
                iou = inter / (areas[best] + areas[rest] - inter)
                remaining = rest[iou < self.nms_t]

        return (np.array(box_predictions), np.array(predicted_classes),
                np.array(predicted_scores))

    @staticmethod
    def load_images(folder_path):
        """
        Loads all the images found in a folder

        folder_path: string representing the path to the folder holding all
            the images to load

        Returns: tuple of (images, image_paths)
            images: list of images as numpy.ndarrays
            image_paths: list of paths to the individual images in images
        """
        image_paths = glob.glob(os.path.join(folder_path, '*'))
        images = [cv2.imread(path) for path in image_paths]

        return images, image_paths
