import numpy as np
import cv2
import os

class YOLOPostProcessor:
    def __init__(self, anchor_file, class_names, obj_thresh=0.25, nms_thresh=0.45, input_size=(640, 640)):
        self.obj_thresh = obj_thresh
        self.nms_thresh = nms_thresh
        self.input_size = input_size
        self.class_names = class_names
        self.anchors = self._load_anchors(anchor_file)

    def _load_anchors(self, anchor_file):
        with open(anchor_file, "r") as f:
            values = [float(x.strip()) for x in f.readlines()]
        anchors = np.array(values).reshape(3, -1, 2).tolist()
        print("Anchors loaded:", anchors)
        return anchors

    def _filter_boxes(self, boxes, box_confidences, box_class_probs):
        box_confidences = box_confidences.reshape(-1)
        class_max_score = np.max(box_class_probs, axis=-1)
        classes = np.argmax(box_class_probs, axis=-1)
        _class_pos = np.where(class_max_score * box_confidences >= self.obj_thresh)
        scores = (class_max_score * box_confidences)[_class_pos]
        boxes = boxes[_class_pos]
        classes = classes[_class_pos]
        return boxes, classes, scores

    def _nms_boxes(self, boxes, scores):
        x = boxes[:, 0]
        y = boxes[:, 1]
        w = boxes[:, 2] - boxes[:, 0]
        h = boxes[:, 3] - boxes[:, 1]
        areas = w * h
        order = scores.argsort()[::-1]
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x[i], x[order[1:]])
            yy1 = np.maximum(y[i], y[order[1:]])
            xx2 = np.minimum(x[i] + w[i], x[order[1:]] + w[order[1:]])
            yy2 = np.minimum(y[i] + h[i], y[order[1:]] + h[order[1:]])
            w1 = np.maximum(0.0, xx2 - xx1 + 1e-5)
            h1 = np.maximum(0.0, yy2 - yy1 + 1e-5)
            inter = w1 * h1
            ovr = inter / (areas[i] + areas[order[1:]] - inter)
            inds = np.where(ovr <= self.nms_thresh)[0]
            order = order[inds + 1]
        return np.array(keep)

    def _box_process(self, position, anchors):
        grid_h, grid_w = position.shape[2:4]
        col, row = np.meshgrid(np.arange(0, grid_w), np.arange(0, grid_h))
        col = col.reshape(1, 1, grid_h, grid_w)
        row = row.reshape(1, 1, grid_h, grid_w)
        grid = np.concatenate((col, row), axis=1)
        stride = np.array([self.input_size[1] // grid_w, self.input_size[0] // grid_h]).reshape(1, 2, 1, 1)
        col = col.repeat(len(anchors), axis=0)
        row = row.repeat(len(anchors), axis=0)
        anchors = np.array(anchors).reshape(len(anchors), 2, 1, 1)
        box_xy = position[:, :2, :, :] * 2 - 0.5
        box_wh = np.power(position[:, 2:4, :, :] * 2, 2) * anchors
        box_xy += grid
        box_xy *= stride
        box = np.concatenate((box_xy, box_wh), axis=1)
        xyxy = np.copy(box)
        xyxy[:, 0, :, :] = box[:, 0, :, :] - box[:, 2, :, :] / 2
        xyxy[:, 1, :, :] = box[:, 1, :, :] - box[:, 3, :, :] / 2
        xyxy[:, 2, :, :] = box[:, 0, :, :] + box[:, 2, :, :] / 2
        xyxy[:, 3, :, :] = box[:, 1, :, :] + box[:, 3, :, :] / 2
        return xyxy

    def _flatten_output(self, out):
        ch = out.shape[1]
        out = out.transpose(0, 2, 3, 1)
        return out.reshape(-1, ch)

    def __call__(self, model_outputs):
        boxes, scores, class_probs = [], [], []

        split_outputs = []
        for i, out in enumerate(model_outputs):
            c, h, w = out.shape
            num_anchors = len(self.anchors[i])
            num_classes = c // num_anchors - 5
            assert c == num_anchors * (
                        num_classes + 5), f"Output channels ({c}) not compatible with anchors ({num_anchors}) and classes ({num_classes})"
            out = out.reshape(num_anchors, num_classes + 5, h, w)
            split_outputs.append(out)

        for i, out in enumerate(split_outputs):
            boxes.append(self._box_process(out[:, :4, :, :], self.anchors[i]))
            scores.append(out[:, 4:5, :, :])
            class_probs.append(out[:, 5:, :, :])

        boxes = [self._flatten_output(b) for b in boxes]
        scores = [self._flatten_output(s) for s in scores]
        class_probs = [self._flatten_output(p) for p in class_probs]

        boxes = np.concatenate(boxes)
        scores = np.concatenate(scores)
        class_probs = np.concatenate(class_probs)

        boxes, classes, scores = self._filter_boxes(boxes, scores, class_probs)

        if boxes.shape[0] == 0:
            return None, None, None

        nboxes, nclasses, nscores = [], [], []
        for c in set(classes):
            inds = np.where(classes == c)
            b = boxes[inds]
            s = scores[inds]
            keep = self._nms_boxes(b, s)
            nboxes.append(b[keep])
            nclasses.append(np.full_like(keep, c))
            nscores.append(s[keep])

        return (
            np.concatenate(nboxes),
            np.concatenate(nclasses),
            np.concatenate(nscores)
        )

    def draw(self, image, boxes, scores, classes):
        if boxes is None:
            return image

        for box, score, cl in zip(boxes, scores, classes):
            left, top, right, bottom = [int(b) for b in box]
            label = f"{self.class_names[cl]} {score:.2f}"
            cv2.rectangle(image, (left, top), (right, bottom), (0, 0, 255), 2)  # 红色框
            cv2.putText(image, label, (left, max(top - 6, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        return image

