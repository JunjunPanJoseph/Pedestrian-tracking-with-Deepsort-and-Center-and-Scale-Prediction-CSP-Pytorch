from deep_sort import DeepSort
from config import Config
import numpy as np
config = Config


class DeepsortTracker(object):
    def __init__(self, config=config):
        self.config = config

        self.deepsort = DeepSort(config.deepsort_checkpoint, use_cuda=config.use_cuda)

    def detect(self, img, boxes_x1y1x2y2conf):
        box_xcycwh = []
        box_conf = []
        for box_x1y1x2y2conf in boxes_x1y1x2y2conf:
            box = box_x1y1x2y2conf
            box_xcycwh.append(np.array([(box[0] + box[2]) // 2, (box[1] + box[3]) // 2, box[2] - box[0], box[3] - box[1]], dtype=np.int32))
            box_conf.append(box[4])
        box_xcycwh = np.array(box_xcycwh)
        outputs, track_states = self.deepsort.update(box_xcycwh, box_conf, img)
        if outputs == []:
            return [], []
        box_x1y1x2y2 = outputs[:, :4]
        identities = outputs[:, -1]
        return box_x1y1x2y2, identities, track_states
