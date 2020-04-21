import cv2 as cv2
import numpy as np

def is_inside(o, i):
    ox, oy, ow, oh = o
    ix, iy, iw, ih = i
    return ox > ix and oy > iy and ox + ow < ix + iw and oy + oh < iy + ih

class model_HOG_SVM():
    def __init__(self):
        self.hog = cv2.HOGDescriptor()
        self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        self.winStride = (8, 8)
        self.scale = 1.05
        self.rescale_dim = (8, 8)

    def detect(self, img):
        proposals, w = self.hog.detectMultiScale(img, winStride=self.winStride, scale=self.scale)
        proposals_filted = []
        features = []
        for ri, r in enumerate(proposals):
            for qi, q in enumerate(proposals):
                a = is_inside(r, q)
                if ri != qi and a:
                    break
            else:
                proposals_filted.append(r)
        for proposal in proposals_filted:
            x, y, w, h = proposal
            roi = img[y:y + h, x:x + w, :]
            resized = cv2.resize(roi, self.rescale_dim, interpolation=cv2.INTER_AREA)
            # Temp feature: reshape image and concat it to a vector
            feature = np.reshape(resized, [-1])
            features.append(feature)

        return proposals_filted, features