import numpy as np
import cv2

# def nms(bboxes, thresh):
#     x1 = bboxes[:, 0]
#     y1 = bboxes[:, 1]
#     x2 = bboxes[:, 2]
#     y2 = bboxes[:, 3]
#     scores = bboxes[:, 4]
#
#     order = np.argsort(-scores)
#     res = []
#
#     while len(order) > 0:
#         bbox = bboxes[order[0]]
#         res.append(bbox)
#
#         x11 = np.maximum(x1[order[1:]], bbox[0])
#         y11 = np.maximum(y1[order[1:]], bbox[1])
#         x21 = np.minimum(x2[order[1:]], bbox[2])
#         y21 = np.minimum(y2[order[1:]], bbox[3])
#
#         iw = np.maximum(x21-x11+1, 0)
#         ih = np.maximum(y21-y11+1, 0)
#
#         inter = iw*ih
#
#
#         a1 = (x2[order[1:]] - x1[order[1:]] + 1) * (y2[order[1:]] - y1[order[1:]] + 1) - inter
#         a2 = (bbox[2] - bbox[0] + 1) * (bbox[3] - bbox[1] + 1)
#         union = a1 + a2
#
#         ious = inter/union
#         order = order[1:][ious < thresh]
#
#     return np.array(res)
def nms(bboxes, thresh):
    x1s = bboxes[:, 0]
    y1s = bboxes[:, 1]
    x2s = bboxes[:, 2]
    y2s = bboxes[:, 3]

    scores = bboxes[:, 4]
    order = np.argsort(-scores)
    res = []
    while len(order) > 0:
        bbox = bboxes[order[0]]
        res.append(bbox)

        x11 = np.maximum(x1s[order[1:]], bbox[0])
        y11 = np.maximum(y1s[order[1:]], bbox[1])
        x22 = np.minimum(x2s[order[1:]], bbox[2])
        y22 = np.minimum(y2s[order[1:]], bbox[3])

        iw = np.maximum(x22-x11+1, 0)
        ih = np.maximum(y22-y11+1, 0)

        inter = iw*ih

        union = (x22-x11+1)*(y22-y11+1) + (bbox[2] - bbox[0]+1)*(bbox[3] - bbox[1] + 1) - inter

        ious = inter/union

        order = order[1:][ious < thresh]

    return np.array(res)

def draw_bbox(bboxs, pic_name):
    pic = np.zeros((850, 850), np.uint8)
    for bbox in bboxs:
        x1, y1, x2, y2 = map(int, bbox[:-1])
        pic = cv2.rectangle(pic, (x1, y1), (x2, y2), (255, 0, 0), 2)
    cv2.imshow(pic_name, pic)
    cv2.waitKey(0)


if __name__ == "__main__":
    x = 1
    y = 0
    if y == 0:
        print(x)
    bboxs = np.array([
        [720, 690, 820, 800, 0.5],
        [204, 102, 358, 250, 0.5],
        [257, 118, 380, 250, 0.8],
        [700, 700, 800, 800, 0.4],
        [280, 135, 400, 250, 0.7],
        [255, 118, 360, 235, 0.7]])
    thresh = 0.3
    draw_bbox(bboxs, "Before_NMS")
    result = nms(bboxs, thresh)
    # print(len(result))
    draw_bbox(result, "After_NMS")
