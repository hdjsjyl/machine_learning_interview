import numpy as np

thre = 0.5

def helper(gt_bboxes, image_ids, det_bboxes):
    new_gtBboxes = {}
    npos = 0
    for key, value in gt_bboxes.items():
        tmp = {}
        tmp['gt'] = value
        tmp['det'] = [False for i in range(len(value))]
        new_gtBboxes[key] = tmp
        npos += len(value)

    scores = det_bboxes[:, 4]
    order = np.argsort(-scores)
    det_bboxes = det_bboxes[order]
    image_ids = image_ids[order]

    tp = [0 for i in range(len(det_bboxes))]
    fp = [0 for i in range(len(det_bboxes))]

    for i in range(len(det_bboxes)):
        box = det_bboxes[i]
        id = image_ids[i]
        gt = new_gtBboxes[id]['gt']
        if gt.shape[0] > 0:
            x1 = np.maximum(gt[:, 0], box[0])
            y1 = np.maximum(gt[:, 1], box[1])
            x2 = np.minimum(gt[:, 2], box[2])
            y2 = np.minimum(gt[:, 3], box[3])

            w = np.maximum(x2-x1+1, 0)
            h = np.maximum(y2-y1+1, 0)

            inter = w*h
            union = (gt_bboxes[:, 2] - gt_bboxes[:, 0] + 1) * (gt_bboxes[:, 3] - gt_bboxes[:, 1] + 1) * (det_bboxes[2] - det_bboxes[0] + 1) * (det_bboxes[3] - det_bboxes[1] + 1) - inter
            over = inter / union

            mover = np.max(over)
            mid   = np.argmax(over)
            if mover > thre:
                if new_gtBboxes[id]['det'][mid] != 1:
                    new_gtBboxes[id]['det'][mid] = 1
                    tp[i] = 1
                else:
                    fp[i] = 1
            else:
                fp[i] = 1

        return helper2(fp, tp, npos)

def helper2(fp, tp, npos, mode='2'):
    tps = np.cumsum(tp)
    fps = np.cumsum(fp)
    recall = tps / npos
    precision = tps / np.maximum(tps + fps, 1e-4)

    if mode == '2':
        recall = np.concatenate(([0], recall, [1]))
        precision = np.concatenate(([0], precision, [0]))
        for i in range(len(precision)-1, 0, -1):
            precision[i-1] = np.maximum(precision[i-1], precision[i])
        i = np.where(recall[1:] != recall[:-1])[0]
        maps = np.sum((recall[i+1] - recall[i])* precision[i+1])
        return maps
    else:
        maps = 0
        for i in range(0.0, 1.1, 0.1):
            if np.sum(recall>=i) == 0:
                continue
            else:
                p = np.max(precision[recall>=i])
                maps += p / 11
        return maps

def nms(bboxes, thre):
    x1 = bboxes[:, 0]
    y1 = bboxes[:, 1]
    x2 = bboxes[:, 2]
    y2 = bboxes[:, 3]

    scores = bboxes[:, 4]
    order = np.argsort(-scores)

    res = []
    while len(order) > 0:
        bbox = bboxes[order[0]]
        res.append(bbox)

        x11 = np.maximum(x1[order[1:]], bbox[0])
        y11 = np.maximum(y1[order[1:]], bbox[1])
        x22 = np.maximum(x2[order[1:]], bbox[2])
        y22 = np.maximum(y2[order[1:]], bbox[3])

        iw = np.maximum(x22-x11+1, 0)
        ih = np.maximum(y22-y11+1, 0)

        inter = iw * ih
        union = (x2[order[1:]] - x1[order[1:]] + 1) * (y2[order[1:]] - y1[order[1:]] + 1) + (x2 - x1 + 1) * (y2 - y1 + 1) - inter
        ious = inter / union
        order = order[1:][ious < thre]

    return res

# step1.   load dataset
# step2.   make it iterable
# step2.1. create the model class
# step3.   instantiate the model class
# step4.   instantiate the optimizer
# step5.   instantiate the loss function
# step6.   train the model
# step7.   test the model

