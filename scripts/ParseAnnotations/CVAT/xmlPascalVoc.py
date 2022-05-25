import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import Dataset
import xml.etree.ElementTree as ET
from typing import Sequence, List, Optional, Dict, Any, Union, Tuple
from operator import itemgetter
from pathlib import Path
import imghdr
import cv2

from scripts.Augmentation.object_detection import PlotImagesAnnotations


class PascalVocXml(Dataset):

    @staticmethod
    def polygon2rect(coordinates: Sequence[str]) -> Tuple[Any, Any, Any, Any]:
        """
        Params:
            coodinates = ['x0,y0', 'x1,y1', 'x2,y2', 'x3,y3', 'x4,y4', ... ]
            min_length of coodinates = 4
        Return:
            coordinates = ['x0,y0', 'x1,y1', 'x2,y2', 'x3,y3']
        """
        float_coodinates = [list(map(float, point.split(','))) for point in coordinates]
        float_coodinates = np.asarray(float_coodinates)
        x_min = float_coodinates[:, 0].min()
        x_max = float_coodinates[:, 0].max()
        y_min = float_coodinates[:, 1].min()
        y_max = float_coodinates[:, 1].max()
        return x_min, y_min, x_max, y_max

    def read_xml(self) -> Dict[Optional[str], Dict[str, Union[List[List[Union[str, Any]]], str, None]]]:
        """
        Return: {
                'image_name' :
                    [
                        [x_min, y_min, x_max, y_max, 'Text'], ...
                    ],
                ...
                }
        """
        annotations_files = self.gtFile
        all_annots = dict()
        tree = ET.parse(annotations_files.as_posix())
        root = tree.getroot()
        for img in root.findall('image'):
            if img.attrib.get('name') not in self.imgFiles:
                # Skip image file that have no annotations
                continue
            bboxes = list()
            for bbox in img.getchildren():
                if bbox.attrib.get('label') in ['Textline', 'Punctuation'] or not bbox.attrib.get('points'):
                    # Rest all labels are assumed to represent word bbox irrespective of language
                    continue
                coordinates = bbox.attrib.get('points').split(';')
                if len(coordinates) < 4:
                    continue
                x_min, y_min, x_max, y_max = self.polygon2rect(coordinates)
                if x_min >= x_max or y_min >= y_max:
                    # if this condition is voilated, that can lead to error in bbox & image transformation
                    continue
                    # raise Exception("error!") # continue
                # specifying label is necessary for albumentation transform
                bboxes.append([x_min, y_min, x_max, y_max, self.label])
            if bboxes:
                all_annots[img.attrib.get('name')] = dict(bboxes=bboxes, width=img.attrib.get('width'),
                                                          height=img.attrib.get('height'))
        return all_annots

    # todo
    def read_json(self):
        # read json in case the whole task on cvat is exported.
        pass

    def __init__(self, datasetDir: str, imgDir: str, gtFile: str):

        self.datasetDir = Path(datasetDir)
        self.imgDir = Path(imgDir)
        self.gtFile = Path(gtFile)
        self.label = 'Text'  # temporary ...
        self.plot_annots = PlotImagesAnnotations()

        self.imgFiles = [file.name for file in self.imgDir.iterdir() if imghdr.what(file)]
        self.annotations = self.read_xml()
        self.imgFiles = list(self.annotations.keys())  # skip image files that don't have annotations
        self.imgBBoxes = dict()  # { imgFile: [x_min, y_min, x_max, y_max, 'Text'], ... }

    def __call__(self, *args, **kwargs):
        for imgFile in self.imgFiles:
            bboxes = self.annotations.get(imgFile).get('bboxes')
            self.imgBBoxes[Path(imgFile).stem] = bboxes

    def __len__(self):
        return len(self.imgFiles)

    def __getitem__(self, index):
        img_file = self.imgDir / self.imgFiles[index]
        bboxes = self.annotations.get(self.imgFiles[index]).get('bboxes')
        img = cv2.imread(img_file.as_posix())
        print("Image Shape:   ", img.shape)
        print("Image Shape Attrib:   ", (self.annotations.get(self.imgFiles[index]).get('height'),
                                         self.annotations.get(self.imgFiles[index]).get('width')))
        bbox_img = self.plot_annots.plot_bbox(img, bboxes)
        plt.imshow(bbox_img)
        plt.show()
        return dict(imgFile=img_file, image=None, annots=self.annotations.get(self.imgFiles[index]))


if __name__ == '__main__':
    dataset = PascalVocXml(
        '/home/mansoor/Projects/Yolov5-nano-Word-Detector/dataset/words-dataset/wla-1',
        '/home/mansoor/Projects/Yolov5-nano-Word-Detector/dataset/words-dataset/wla-1/images',
        '/home/mansoor/Projects/Yolov5-nano-Word-Detector/dataset/words-dataset/wla-1/annotations.xml',
    )
    list([_ for _ in dataset])
