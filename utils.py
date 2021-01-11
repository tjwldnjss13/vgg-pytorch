import torch
import numpy as np


def make_batch(datas, category=None):
    if category is None:
        data_batch = datas[0].unsqueeze(0)
        i = 1
        while i < len(datas):
            temp = datas[i].unsqueeze(0)
            data_batch = torch.cat([data_batch, temp], dim=0)
            i += 1
    else:
        data_batch = datas[category][0].unsqueeze(0)
        i = 1
        while i < len(datas[category]):
            temp = datas[category][i].unsqueeze(0)
            data_batch = torch.cat([data_batch, temp], dim=0)
            i += 1

    data_batch = data_batch.squeeze(dim=0)

    return data_batch


def make_pretrain_annotation_batch(anns, ann_category):
    print(anns[0][ann_category])
    ann_batch = anns[0][ann_category].unsqueeze(0)
    for i in range(1, len(anns)):
        temp = anns[i][ann_category].unsqueeze(0)
        ann_batch = torch.cat([ann_batch, temp], dim=0)

    return ann_batch


def crop(in_, out_size):
    in_size = (in_.shape[1], in_.shape[2])
    drop_size = (in_size[0] - out_size[0], in_size[1] - out_size[1])
    drop_size = (drop_size[0] // 2, drop_size[1] // 2)
    out_ = in_[:, drop_size[0]:drop_size[0] + out_size[0], drop_size[1]:drop_size[1] + out_size[1]]

    return out_


def pad_4dim(x, ref, cuda=True):
    zeros = torch.zeros(x.shape[0], x.shape[1], 1, x.shape[3])
    if cuda:
        zeros = zeros.cuda()
    while x.shape[2] < ref.shape[2]:
        x = torch.cat([x, zeros], dim=2)
    zeros = torch.zeros(x.shape[0], x.shape[1], x.shape[2], 1)
    if cuda:
        zeros = zeros.cuda()
    while x.shape[3] < ref.shape[3]:
        x = torch.cat([x, zeros], dim=3)

    return x


def pad_3dim(x, ref_size):
    zeros = torch.zeros(x.shape[0], 1, x.shape[2]).cuda()
    while x.shape[1] < ref_size[0]:
        x = torch.cat([x, zeros], dim=1)
    zeros = torch.zeros(x.shape[0], x.shape[1], 1).cuda()
    while x.shape[2] < ref_size[1]:
        x = torch.cat([x, zeros], dim=2)

    return x


def pad_2dim(x, ref_size):
    zeros = torch.zeros(1, x.shape[1], dtype=torch.long).cuda()
    while x.shape[0] < ref_size[0]:
        x = torch.cat([x, zeros], dim=0)
    zeros = torch.zeros(x.shape[0], 1, dtype=torch.long).cuda()
    while x.shape[1] < ref_size[1]:
        x = torch.cat([x, zeros], dim=1)

    return x


def calculate_ious(box1, box2):
    ious = np.zeros((box1.shape[0], box2.shape[0]), dtype=np.float32)

    for i, b1 in enumerate(box1):
        b1 = box1[i]
        for j, b2 in enumerate(box2):
            b2 = box2[j]
            ious[i, j] = calculate_iou(b1, b2)

    return ious


def calculate_iou(box1, box2):
    # Inputs:
    #    box: [y1, x1, y2, x2]
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

    y1 = np.maximum(box1[0], box2[0])
    x1 = np.maximum(box1[1], box2[1])
    y2 = np.minimum(box1[2], box2[2])
    x2 = np.minimum(box1[3], box2[3])

    iou = 0
    if y1 < y2 and x1 < x2:
        inter = (y2 - y1) * (x2 - x1)
        union = area1 + area2 - inter
        iou = inter / union

    return iou


def mean_iou_segmentation(output, predict):
    a, b = (output[:, 1, :, :] > 0), (predict > 0)

    a_area = len(a.nonzero())
    b_area = len(b.nonzero())
    union = a_area + b_area
    inter = len((a & b).nonzero())
    iou = inter / (union - inter)

    return iou


def nms(boxes, probs, threshold):
    # Inputs:
    #    boxes: [[y1, x1, y2, x2]]

    nms_list = list()

    box_prob_tuple = list(zip(boxes, probs))

    def prob(t):
        return t[1]

    box_prob_tuple.sort(key=prob, reverse=True)
    base = box_prob_tuple.pop(0)
    box_base, prob_base = base[0], base[1]

    nms_list.append(base)

    for i, t in enumerate(box_prob_tuple):
        box, prob = t[0], t[1]
        iou = calculate_iou(box_base, box)

        if iou < threshold:
            nms_list.append(t)

    return nms_list



def nms_ground_truth(anchor_boxes, ground_truth, score, iou_threshold):
    n_gt = ground_truth.shape[0] if len(ground_truth.shape) > 1 else 1
    anchor_boxes_nms = []

    for i in range(n_gt):
        anchor_boxes_cat = anchor_boxes[i]
        ious_boxes_gts = calculate_ious(anchor_boxes_cat, np.array([ground_truth[i]]))
        argmax_iou_boxes_gt = np.argmax(ious_boxes_gts, axis=0)
        max_iou_box = anchor_boxes_cat[argmax_iou_boxes_gt][0]
        anchor_boxes_nms.append(max_iou_box)
        for j in range(anchor_boxes_cat.shape[0]):
            if j == argmax_iou_boxes_gt:
                continue
            iou_temp = calculate_iou(max_iou_box, anchor_boxes_cat[j])
            if iou_temp >= iou_threshold:
                anchor_boxes_nms.append(anchor_boxes_cat[j])

    return np.array(anchor_boxes_nms)


def time_calculator(sec):
    if sec < 60:
        return 0, 0, sec
    if sec < 3600:
        M = sec // 60
        S = sec % (M * 60)
        return 0, int(M), S
    H = sec // 3600
    sec = sec % 3600
    M = sec // 60
    S = sec % 60
    return int(H), int(M), S

