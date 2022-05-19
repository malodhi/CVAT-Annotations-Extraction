import matplotlib.pyplot as plt
from torch.utils.data import Dataset
import xml.etree.ElementTree as ET
from typing import Sequence, List, Optional, Dict, Any, Union
from operator import itemgetter
from pathlib import Path
import imghdr
import cv2

from scripts.Augmentation.imgAugmentation import PlotImagesAnnotations


class PascalVocXml(Dataset):

    @staticmethod
    def polygon2rect(coordinates: Sequence[str]) -> List[Optional[str]]:
        """
        Params:
            coodinates = ['x0,y0', 'x1,y1', 'x2,y2', 'x3,y3', 'x4,y4', ... ]
            min_length of coodinates = 4
        Return:
            coordinates = ['x0,y0', 'x1,y1', 'x2,y2', 'x3,y3']
        """
        float_coodinates = [list(map(float, point.split(','))) for point in coordinates]
        x_min, y_min = min(float_coodinates, key=itemgetter(0))[0], min(float_coodinates, key=itemgetter(1))[1]
        x_max, y_max = max(float_coodinates, key=itemgetter(0))[0], max(float_coodinates, key=itemgetter(1))[1]
        start_point, top_right_point, end_point, bottom_left_point = \
            (x_min, y_min), (x_max, y_min), (x_max, y_max), (x_min, y_max)
        rect_coodinates = [start_point, top_right_point, end_point, bottom_left_point]
        return [",".join([str(point[0]), str(point[1])]) for point in rect_coodinates]

    def read_xml(self) -> Dict[Any, List[List[Union[int, str]]]]:
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
            for bbox in list(img): #img.getchildren():
                if bbox.attrib.get('label') == 'Textline':
                    # Rest all labels are assumed to represent word bbox irrespective of language
                    continue
                coordinates = bbox.attrib.get('points').split(';')
                if len(coordinates) < 4:
                    continue
                if len(coordinates) > 4:
                    coordinates = self.polygon2rect(coordinates)
                start_point = coordinates[0].split(',')
                end_point = coordinates[2].split(',')
                x_min, y_min = int(float(start_point[0])), int(float(start_point[1]))
                x_max, y_max = int(float(end_point[0])), int(float(end_point[1]))
                if x_min >= x_max or y_min >= y_max:
                    # if this condition is voilated, that can lead to error in bbox & image transformation
                    continue
                # specifying label is necessary for albumentation transform
                bboxes.append([x_min, y_min, x_max, y_max, self.label])
            if bboxes:
                all_annots[img.attrib.get('name')] = bboxes
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

        self.imgFiles = [file.name for file in self.imgDir.iterdir() if imghdr.what(file)]
        self.annotations = self.read_xml()
        # skip image files that don't have annotations
        self.imgFiles = list(self.annotations.keys())

        self.plot_annots = PlotImagesAnnotations()

    def __len__(self):
        return len(self.imgFiles)

    def __getitem__(self, index):

        img_file = self.imgDir / self.imgFiles[index]
        bboxes = self.annotations.get(self.imgFiles[index])
        img = cv2.imread(img_file.as_posix())
        bbox_img = self.plot_annots.plot_bbox(img, bboxes)
        plt.imshow(bbox_img)
        plt.show()


if __name__ == '__main__':
    dataset = PascalVocXml(
        '/home/mansoor/Downloads/task_templates character-level annotations-2022_05_11_15_46_00-cvat for images 1.1',
        '/home/mansoor/Downloads/task_templates character-level annotations-2022_05_11_15_46_00-cvat for images '
        '1.1/images',
        '/home/mansoor/Downloads/task_templates character-level annotations-2022_05_11_15_46_00-cvat for images '
        '1.1/annotations.xml',
    )
    list([_ for _ in dataset])
