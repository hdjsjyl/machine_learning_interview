from collections import defaultdict
import numpy as np
import os

def getMAP(gtBboxes, detBboxes, imageIds):
    """
    return map for each class
    :param gtBboxes: dictionary: key -> imageId, values -> gtbboxes, length is equal to the number of images
    :param detBboxes: nparray: length is N, in total there are N detection boxes;
    :param imageIds: nparray: length is N, each image id correlates to the detection box;
    :return: map: precision and recall; precision: tp/(fp+tp); recall: tp/(tp+fn)
    """
    overthresh = 0.5 # iou threshold value
    npos = 0  # the number of ground truth boxes

    new_gtBboxes = {}
    for key, value in gtBboxes.items():
        tmp = {}
        tmp['gt'] = value
        tmp['det'] = [False for i in range(len(value))]
        new_gtBboxes[key] = tmp
        npos += len(value)

    tp = [0 for i in range(len(detBboxes))]
    fp = [0 for i in range(len(detBboxes))]

    ## order the detbboxes in terms of confidence scores
    confidence = detBboxes[:, -1]
    order     = np.argsort(-confidence)
    detBboxes = detBboxes[order]
    imageIds  = imageIds[order]

    for i in range(len(detBboxes)):
        id = imageIds[i]
        gt = new_gtBboxes[id]['gt']
        det = (detBboxes[i])

        if gt.shape[0] > 0:
            x1 = np.maximum(gt[:, 0], det[0])
            y1 = np.maximum(gt[:, 1], det[1])
            x2 = np.minimum(gt[:, 2], det[2])
            y2 = np.minimum(gt[:, 3], det[3])

            iw = np.maximum(x2-x1+1, 0)
            ih = np.maximum(y2-y1+1, 0)

            inter = iw*ih
            union = (gt[:, 2] - gt[:, 0] + 1) * (gt[:, 3] - gt[:, 1] + 1) + (det[3] - det[1] + 1) * (det[2] - det[0] + 1) - inter
            iou   = inter/union
            max   = np.max(iou)
            idx   = np.argmax(iou)


            if max > overthresh:
                if new_gtBboxes[id]['det'][idx] != 1:
                    tp[i] = 1
                    new_gtBboxes[id]['det'][idx] = 1
                else:
                    fp[i] = 1
            else:
                fp[i] = 1

    map = getVOCMAP(tp, fp, npos)
    return map


def getVOCMAP(tp, fp, npos, num_images, mode='2'):
    tp = np.cumsum(tp)
    fp = np.cumsum(fp)

    rec       = tp / npos
    precision = tp / np.maximum(tp+fp, 1e-4)

    if mode == '1':
        # voc 11 points evaluation method
        ap = 0
        for t in np.arange(0.0, 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(precision[rec >= t])
            ap += p / 11
        return ap

    elif mode == '2':
        # correct AP calculation
        mrec = np.concatenate(([0], rec, [1]))
        mpre = np.concatenate(([0], precision, [0]))

        for i in range(len(mpre) - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        i = np.where(mrec[1:] != mrec[:-1])[0]
        map = np.sum((mrec[i+1] - mrec[i]) * mpre[i+1])
        return map

    elif mode == '3':
        ## log average miss rate
        fppi = fp/num_images
        mr   = 1 - rec

        logavgmr = []
        for t in np.logspace(-2.0, 0.0, num=9):
            tmp = np.min(mr[fppi <= t])
            logavgmr.append(np.max(np.log(tmp), 1e-10))
        res = np.exp(np.mean(logavgmr))

        return res


if __name__ == '__main__':
    gtFile   = './objectDetection/data/groundtruths/1.txt'
    detFile  = './objectDetection/data/detections/1.txt'
    classes = set()
    gtBboxes = defaultdict(list)
    imageNumber = 0
    with open(gtFile, 'r') as f:
        lines = f.readlines()
        for line in lines:
            data  = line.strip()
            box   = data.split(' ')
            boxes = box[1:]
            boxes = [float(i) for i in boxes]
            cls   = box[0]
            classes.add(cls)
            gtBboxes[cls].append(boxes)
            imageNumber += 1

    detBboxes = defaultdict(list)
    with open(detFile, 'r') as f:
        lines = f.readlines()
        for line in lines:
            data  = line.strip()
            box   = data.split(' ')
            boxes = box[1:]
            boxes = [float(i) for i in boxes]
            cls   = box[0]
            detBboxes[cls].append(boxes)

    for key, value in gtBboxes.items():
        gt  = np.array(gtBboxes[key])
        det = np.array(detBboxes[key])
        ids = [1 for i in range(len(det))]
        ids = np.array(ids)

        gts = {}
        gts[1] = gt
        print(key, getMAP(gts, det, ids))