import itertools
from collections import defaultdict
import logging
logger = logging.getLogger(__name__)
import time

#from chainercv.utils import non_maximum_suppression
import numpy as np
from PIL import ImageDraw


def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = itertools.tee(iterable)
    next(b, None)
    return zip(a, b)


DEFAULT_KEYPOINT_NAMES = [
    'nose',
    'left_eye',
    'right_eye',
    'left_ear',
    'right_ear',
    'left_shoulder',
    'right_shoulder',
    'left_elbow',
    'right_elbow',
    'left_wrist',
    'right_wrist',
    'left_hip',
    'right_hip',
    'left_knee',
    'right_knee',
    'left_ankle',
    'right_ankle'
]


# update keypoints
KEYPOINT_NAMES = ['neck'] + DEFAULT_KEYPOINT_NAMES
# update keypoints
KEYPOINT_NAMES = ['instance'] + KEYPOINT_NAMES

COLOR_MAP = {
    'instance': (225, 225, 225),
    'nose': (255, 0, 0),
    'neck': (255, 85, 0),
    'right_shoulder': (255, 170, 0),
    'right_elbow': (255, 255, 0),
    'right_wrist': (170, 255, 0),
    'left_shoulder': (85, 255, 0),
    'left_elbow': (0, 127, 0),
    'left_wrist': (0, 255, 85),
    'right_hip': (0, 170, 170),
    'right_knee': (0, 255, 255),
    'right_ankle': (0, 170, 255),
    'left_hip': (0, 85, 255),
    'left_knee': (0, 0, 255),
    'left_ankle': (85, 0, 255),
    'right_eye': (170, 0, 255),
    'left_eye': (255, 0, 255),
    'right_ear': (255, 0, 170),
    'left_ear': (255, 0, 85),
}

EDGES_BY_NAME = [
    ['instance', 'neck'],
    ['neck', 'nose'],
    ['nose', 'left_eye'],
    ['left_eye', 'left_ear'],
    ['nose', 'right_eye'],
    ['right_eye', 'right_ear'],
    ['neck', 'left_shoulder'],
    ['left_shoulder', 'left_elbow'],
    ['left_elbow', 'left_wrist'],
    ['neck', 'right_shoulder'],
    ['right_shoulder', 'right_elbow'],
    ['right_elbow', 'right_wrist'],
    ['neck', 'left_hip'],
    ['left_hip', 'left_knee'],
    ['left_knee', 'left_ankle'],
    ['neck', 'right_hip'],
    ['right_hip', 'right_knee'],
    ['right_knee', 'right_ankle'],
]

EDGES = [[KEYPOINT_NAMES.index(s), KEYPOINT_NAMES.index(d)] for s, d in EDGES_BY_NAME]

TRACK_ORDER_0 = ['instance', 'neck', 'nose', 'left_eye', 'left_ear']
TRACK_ORDER_1 = ['instance', 'neck', 'nose', 'right_eye', 'right_ear']
TRACK_ORDER_2 = ['instance', 'neck', 'left_shoulder', 'left_elbow', 'left_wrist']
TRACK_ORDER_3 = ['instance', 'neck', 'right_shoulder', 'right_elbow', 'right_wrist']
TRACK_ORDER_4 = ['instance', 'neck', 'left_hip', 'left_knee', 'left_ankle']
TRACK_ORDER_5 = ['instance', 'neck', 'right_hip', 'right_knee', 'right_ankle']

TRACK_ORDERS = [TRACK_ORDER_0, TRACK_ORDER_1, TRACK_ORDER_2, TRACK_ORDER_3, TRACK_ORDER_4, TRACK_ORDER_5]
DIRECTED_GRAPHS = []

for keypoints in TRACK_ORDERS:
    es = [EDGES_BY_NAME.index([a, b]) for a, b in pairwise(keypoints)]
    ts = [KEYPOINT_NAMES.index(b) for a, b in pairwise(keypoints)]
    DIRECTED_GRAPHS.append([es, ts])


def non_maximum_suppression(bbox, thresh, score=None, limit=None):
    """
    Taken From
    https://github.com/chainer/chainercv/blob/master/chainercv/utils/bbox/non_maximum_suppression.py
    """
    if len(bbox) == 0:
        return np.zeros((0,), dtype=np.int32)

    if score is not None:
        order = score.argsort()[::-1]
        bbox = bbox[order]
    bbox_area = np.prod(bbox[:, 2:] - bbox[:, :2], axis=1)

    selec = np.zeros(bbox.shape[0], dtype=bool)
    for i, b in enumerate(bbox):
        tl = np.maximum(b[:2], bbox[selec, :2])
        br = np.minimum(b[2:], bbox[selec, 2:])
        area = np.prod(br - tl, axis=1) * (tl < br).all(axis=1)

        iou = area / (bbox_area[i] + bbox_area[selec] - area)
        if (iou >= thresh).any():
            continue

        selec[i] = True
        if limit is not None and np.count_nonzero(selec) >= limit:
            break

    selec = np.where(selec)[0]
    if score is not None:
        selec = order[selec]
    return selec.astype(np.int32)


class Estimator():

    def __init__(self, insize, outsize, local_size):
        self.insize = insize
        self.outsize = outsize
        self.local_size = local_size
        inW, inH = insize
        outW, outH = outsize
        self.gridsize = (int(inW / outW), int(inH / outH))
        self.keypoint_names = KEYPOINT_NAMES
        self.edges = np.array(EDGES)

    def restore_size(self, w, h):
        inW, inH = self.insize
        return inW * w, inH * h

    def restore_xy(self, x, y):
        gridW, gridH = self.gridsize
        outW, outH = self.outsize
        X, Y = np.meshgrid(np.arange(outW), np.arange(outH))
        return (x + X) * gridW, (y + Y) * gridH

    def extract_feature(self, feature_map):
        K = len(self.keypoint_names)
        outW, outH = self.outsize
        resp = feature_map[0 * K:1 * K, :, :]
        conf = feature_map[1 * K:2 * K, :, :]
        x = feature_map[2 * K:3 * K, :, :]
        y = feature_map[3 * K:4 * K, :, :]
        w = feature_map[4 * K:5 * K, :, :]
        h = feature_map[5 * K:6 * K, :, :]
        e = feature_map[6 * K:, :, :].reshape((
            len(self.edges),
            self.local_size[1], self.local_size[0],
            outH, outW
        ))
        return resp, conf, x, y, w, h, e

    def estimate(self, feature_map, detection_thresh=0.125):
        resp, conf, x, y, w, h, e = self.extract_feature(feature_map)
        start = time.time()
        delta = resp * conf
        K = len(self.keypoint_names)
        outW, outH = self.outsize
        ROOT_NODE = 0  # instance
        start = time.time()
        rx, ry = self.restore_xy(x, y)
        rw, rh = self.restore_size(w, h)
        ymin, ymax = ry - rh / 2, ry + rh / 2
        xmin, xmax = rx - rw / 2, rx + rw / 2
        bbox = np.array([ymin, xmin, ymax, xmax])
        bbox = bbox.transpose(1, 2, 3, 0)
        root_bbox = bbox[ROOT_NODE]
        score = delta[ROOT_NODE]
        candidate = np.where(score > detection_thresh)
        score = score[candidate]
        root_bbox = root_bbox[candidate]
        selected = non_maximum_suppression(
            bbox=root_bbox, thresh=0.3, score=score)
        root_bbox = root_bbox[selected]
        logger.info("detect instance {:.5f}".format(time.time() - start))
        start = time.time()

        humans = []
        e = e.transpose(0, 3, 4, 1, 2)
        ei = 0  # index of edges which contains ROOT_NODE as begin
        # alchemy_on_humans
        for hxw in zip(candidate[0][selected], candidate[1][selected]):
            human = {ROOT_NODE: bbox[(ROOT_NODE, hxw[0], hxw[1])]}  # initial
            for graph in DIRECTED_GRAPHS:
                eis, ts = graph
                i_h, i_w = hxw
                for ei, t in zip(eis, ts):
                    index = (ei, i_h, i_w)  # must be tuple
                    u_ind = np.unravel_index(np.argmax(e[index]), e[index].shape)
                    j_h = i_h + u_ind[0] - self.local_size[1] // 2
                    j_w = i_w + u_ind[1] - self.local_size[0] // 2
                    if j_h < 0 or j_w < 0 or j_h >= outH or j_w >= outW:
                        break
                    if delta[t, j_h, j_w] < detection_thresh:
                        break
                    human[t] = bbox[(t, j_h, j_w)]
                    i_h, i_w = j_h, j_w

            humans.append(human)
        logger.info("alchemy time {:.5f}".format(time.time() - start))
        logger.info("num humans = {}".format(len(humans)))
        return humans


def draw_humans(keypoint_names, edges, pil_image, humans, scale, mask=None):
    """
    This is what happens when you use alchemy on humans...
    note that image should be PIL object
    """
    start = time.time()
    drawer = ImageDraw.Draw(pil_image)
    r = 2
    for human in humans:
        for k, b in human.items():
            if mask:
                fill = (255, 255, 255) if k == 0 else None
            else:
                fill = None
            ymin, xmin, ymax, xmax = scale * b
            if k == 0:
                # adjust size
                t = 1
                xmin = int(xmin * t + xmax * (1 - t))
                xmax = int(xmin * (1 - t) + xmax * t)
                ymin = int(ymin * t + ymax * (1 - t))
                ymax = int(ymin * (1 - t) + ymax * t)
                if mask:
                    resized = mask.resize(((xmax - xmin), (ymax - ymin)))
                    pil_image.paste(resized, (xmin, ymin), mask=resized)
                else:
                    drawer.rectangle(xy=[xmin, ymin, xmax, ymax],
                                     fill=fill,
                                     outline=tuple(COLOR_MAP[keypoint_names[k]]))
            else:
                """
                drawer.rectangle(xy=[xmin, ymin, xmax, ymax],
                                 fill=fill,
                                 outline=tuple(COLOR_MAP[keypoint_names[k]]))
                """
                x = (xmin + xmax) / 2
                y = (ymin + ymax) / 2
                drawer.ellipse(xy=[x - r, y - r, x + r, y + r],
                               fill=fill,
                               outline=tuple(COLOR_MAP[keypoint_names[k]]))
        for s, t in edges:
            if s in human and t in human:
                by = scale * (human[s][0] + human[s][2]) / 2
                bx = scale * (human[s][1] + human[s][3]) / 2
                ey = scale * (human[t][0] + human[t][2]) / 2
                ex = scale * (human[t][1] + human[t][3]) / 2

                drawer.line([bx, by, ex, ey],
                            fill=tuple(COLOR_MAP[keypoint_names[s]]), width=3)

    logger.info("draw humans {:.5f}".format(time.time() - start))
    return pil_image
