import imutils
import cv2
import numpy as np




def area_of(left_top, right_bottom):
    """Compute the areas of rectangles given two corners.

    Args:
        left_top (N, 2): left top corner.
        right_bottom (N, 2): right bottom corner.

    Returns:
        area (N): return the area.
    """
    hw = np.clip(right_bottom - left_top, 0.0, None)
    return hw[..., 0] * hw[..., 1]


def iou_of(boxes0, boxes1, eps=1e-5):
    """Return intersection-over-union (Jaccard index) of boxes.

    Args:
        boxes0 (N, 4): ground truth boxes.
        boxes1 (N or 1, 4): predicted boxes.
        eps: a small number to avoid 0 as denominator.
    Returns:
        iou (N): IoU values.
    """
    overlap_left_top = np.maximum(boxes0[..., :2], boxes1[..., :2])
    overlap_right_bottom = np.minimum(boxes0[..., 2:], boxes1[..., 2:])

    overlap_area = area_of(overlap_left_top, overlap_right_bottom)
    area0 = area_of(boxes0[..., :2], boxes0[..., 2:])
    area1 = area_of(boxes1[..., :2], boxes1[..., 2:])
    return overlap_area / (area0 + area1 - overlap_area + eps)


def center_form_to_corner_form(locations):
    return np.concatenate([locations[..., :2] - locations[..., 2:] / 2,
                           locations[..., :2] + locations[..., 2:] / 2], len(locations.shape) - 1)


def corner_form_to_center_form(boxes):
    return np.concatenate([
        (boxes[..., :2] + boxes[..., 2:]) / 2,
        boxes[..., 2:] - boxes[..., :2]
    ], len(boxes.shape) - 1)


def hard_nms(box_scores, iou_threshold, top_k=-1, candidate_size=200):
    """

    Args:
        box_scores (N, 5): boxes in corner-form and probabilities.
        iou_threshold: intersection over union threshold.
        top_k: keep top_k results. If k <= 0, keep all the results.
        candidate_size: only consider the candidates with the highest scores.
    Returns:
         picked: a list of indexes of the kept boxes
    """
    scores = box_scores[:, -1]
    boxes = box_scores[:, :-1]
    picked = []
    # _, indexes = scores.sort(descending=True)
    indexes = np.argsort(scores)
    # indexes = indexes[:candidate_size]
    indexes = indexes[-candidate_size:]
    while len(indexes) > 0:
        # current = indexes[0]
        current = indexes[-1]
        picked.append(current)
        if 0 < top_k == len(picked) or len(indexes) == 1:
            break
        current_box = boxes[current, :]
        # indexes = indexes[1:]
        indexes = indexes[:-1]
        rest_boxes = boxes[indexes, :]
        iou = iou_of(
            rest_boxes,
            np.expand_dims(current_box, axis=0),
        )
        indexes = indexes[iou <= iou_threshold]

    return box_scores[picked, :]


def detection_predict(width, height, confidences, boxes, prob_threshold, iou_threshold=0.3, top_k=-1):
    boxes = boxes[0]
    confidences = confidences[0]
    picked_box_probs = []
    picked_labels = []
    for class_index in range(1, confidences.shape[1]):
        probs = confidences[:, class_index]
        mask = probs > prob_threshold
        probs = probs[mask]
        if probs.shape[0] == 0:
            continue
        subset_boxes = boxes[mask, :]
        box_probs = np.concatenate([subset_boxes, probs.reshape(-1, 1)], axis=1)
        box_probs = hard_nms(box_probs,
                                             iou_threshold=iou_threshold,
                                             top_k=top_k,
                                             )
        picked_box_probs.append(box_probs)
        picked_labels.extend([class_index] * box_probs.shape[0])
    if not picked_box_probs:
        return np.array([]), np.array([]), np.array([])
    picked_box_probs = np.concatenate(picked_box_probs)
    picked_box_probs[:, 0] *= width
    picked_box_probs[:, 1] *= height
    picked_box_probs[:, 2] *= width
    picked_box_probs[:, 3] *= height
    return picked_box_probs[:, :4].astype(np.int32), np.array(picked_labels), picked_box_probs[:, 4]


def crop_box_alignment(img, box):
    height,width,_=img.shape
    
    x1=box[0]
    y1=box[1]
    x2=box[2]
    y2=box[3]

    w = x2 - x1 + 1
    h = y2 - y1 + 1
    size = int(max([w, h])*1.1)
    cx = x1 + w//2
    cy = y1 + h//2
    x1 = cx - size//2
    x2 = x1 + size
    y1 = cy - size//2
    y2 = y1 + size
    dx = max(0, -x1)
    dy = max(0, -y1)
    x1 = max(0, x1)
    y1 = max(0, y1)

    edx = max(0, x2 - width)
    edy = max(0, y2 - height)
    x2 = min(width, x2)
    y2 = min(height, y2) 
    
    return x1,y1,x2,y2


def resize_pad(image, shape, pad_value):
    o_height, o_width = image.shape[:2]
    scale_h, scale_w = o_height / shape[0], o_width / shape[1]

    if (o_height == shape[0]) and (o_width == shape[1]):
        return image, 0, scale_h, scale_w

    padded_image = np.ones((shape[0],shape[1],3), dtype=np.uint8) * pad_value
    if scale_h > scale_w:
        image = imutils.resize(image, height=shape[0])
        pad_length = (shape[1] - image.shape[1]) // 2
        padded_image[:,pad_length:pad_length+image.shape[1],:] = image
    else:
        image = imutils.resize(image, width=shape[1])
        pad_length = (shape[0] - image.shape[0]) // 2
        padded_image[pad_length:pad_length+image.shape[0],:,:] = image

    return padded_image, pad_length, scale_h, scale_w



def preprocess_detection(image, target_shape=(480,640)):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image, pad_length, scale_h, scale_w = resize_pad(image, target_shape, pad_value=127)

    image_mean = np.array([127, 127, 127])
    image = (image - image_mean) / 128
    image = np.transpose(image, [2, 0, 1])
    image = np.expand_dims(image, axis=0)
    image = image.astype(np.float32)

    return image, pad_length, scale_h, scale_w


def preprocess_recognition(image, target_shape):
    image, _, _, _ = resize_pad(image, target_shape, pad_value=0)

    image = image / 255.0
    image_mean = np.array([0.485, 0.456, 0.406])
    image_std = np.array([0.229, 0.224, 0.225])
    image = (image - image_mean) / image_std
    
    image = np.transpose(image[None], [0,3,1,2])
    return image.astype(np.float32)


def preprocess_alignment(image, box, target_shape=(112,112)):
    aligned_box = crop_box_alignment(image, box)
    
    face_image = image[aligned_box[1]:aligned_box[3],aligned_box[0]:aligned_box[2],:]
    face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
    
    face_image, _, _, _ = resize_pad(face_image, target_shape, pad_value=0)

    face_image = np.transpose(face_image, [2,0,1])[None].astype(np.float32) / 255.0
    return face_image


def preprocess_face_embedding(image, target_shape, normalization):
    image, _, _, _ = resize_pad(image, target_shape, pad_value=0)
    image = image.astype(np.float32)

    if normalization == "LOW":
        pass
    elif normalization == "PERFORMANCE":
        mean, std = image.mean(), image.std()
        image = (image - mean) / std
    elif normalization == "BALANCE":
        image -= 127.5
        image /= 128.0
    else:
        raise ValueError(f"unimplemented normalization type - {normalization}")
    
    image = np.transpose(image, [2,0,1])[None]
    return image


def postprocess_recognition(logits, labels):
    exp_logits = np.exp(logits)
    prob = (exp_logits / exp_logits.sum(axis=1, keepdims=True))[0]

    return [labels[np.argmax(prob)], round(max(prob),2), prob]


def postprocess_alignment(landmarks, target_shape):
    landmarks = landmarks.reshape(-1,2)
    landmarks = np.round(landmarks * np.array(target_shape)).astype(np.int32)
    left_eye, right_eye = landmarks[36], landmarks[45]

    distance = left_eye - right_eye
    angle = -np.arctan(distance[1] / distance[0]) * 180 / np.pi

    return angle


def onnx_infer(image, sess):
    return sess.run(None, {sess.get_inputs()[0].name: image})