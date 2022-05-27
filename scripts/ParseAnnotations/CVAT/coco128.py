from pathlib import Path
import matplotlib.pyplot as plt
from typing import List, Dict

import numpy as np
import cv2

from scripts.utils import search_file


class CoCo128(object):
    def __init__(self, dataDir: str):
        self.dataDir = Path(dataDir)
        self.imgDir = self.dataDir / 'images/train'
        self.lblDir = self.dataDir / 'labels/train'

        if not self.dataDir.exists() or not self.imgDir.exists() or not self.lblDir.exists():
            raise Exception("Data Directory Not Complete !")

        self.fileNames = [file.stem for file in self.lblDir.iterdir()]

    def __len__(self):
        return len(self.fileNames)

    def __getitem__(self, index):
        imgFile = search_file(self.imgDir.as_posix(), str(self.fileNames[index]) + '.*')
        lblFile = search_file(self.lblDir.as_posix(), str(self.fileNames[index]) + '.*')
        img = plt.imread(imgFile)
        bboxes = self.readTextAnnots(lblFile)
        bboxes = self.extract_start_end_points(bboxes, img)
        print(bboxes)
        img = self.plot_bboxes(img, bboxes)
        plt.imshow(img)
        plt.show()

    @staticmethod
    def readTextAnnots(textFile: str) -> List[Dict]:
        bboxes = list()
        f = open(textFile, 'r')
        for bbox in f.readlines():
            annotation = bbox.split(' ')
            if len(annotation) != 5:
                continue
            bboxes.append(dict(text=annotation[0],
                               x_center=float(annotation[1]),
                               y_center=float(annotation[2]),
                               width=float(annotation[3]),
                               height=float(annotation[4])))
        return bboxes

    @staticmethod
    def plot_bboxes(img: np.ndarray, bboxes: List[List[float]]) -> np.ndarray:
        bbox_color = (255, 0, 0)
        bbox_thickness = 2
        for bbox in bboxes:
            # bbox --> [x_min, y_min, x_max, y_max, 'Text']
            start_point = (int(bbox[0]), int(bbox[1]))
            end_point = (int(bbox[2]), int(bbox[3]))
            # cv2.rectange:  img -> HWC & start_points/end_points --> (int, int)
            img = cv2.rectangle(img, start_point, end_point, bbox_color, bbox_thickness)
        return img

    @staticmethod
    def extract_start_end_points(bboxes: List, image: np.ndarray) -> List[List[float]]:
        if len(image.shape) == 2:
            height, width = image.shape
        else:
            height, width, _ = image.shape
        standard_bboxes = list()
        for box in bboxes:
            x_min = (box.get('x_center') - (box.get('width') / 2)) * width
            x_max = (box.get('x_center') + (box.get('width') / 2)) * width
            y_min = (box.get('y_center') - (box.get('height') / 2)) * height
            y_max = (box.get('y_center') + (box.get('height') / 2)) * height
            standard_bboxes.append([x_min, y_min, x_max, y_max])
        return standard_bboxes


if __name__ == '__main__':
    parser = CoCo128(dataDir='/home/mansoor/Projects/Yolov5-nano-Word-Detector/dataset/Words')
    list([_ for _ in parser])


