from torch.utils.data import Dataset
import xml.etree.ElementTree as ET
from matplotlib import pyplot as plt
import albumentations as A
from typing import List, Union, Tuple
from pathlib import Path
import numpy as np
import imghdr
import copy
import cv2


class PlotImagesAnnotations(object):

    def show_multiple_images(self, images, titles=None) -> None:
        assert ((titles is None) or (len(images) == len(titles)))
        n_images = len(images)
        if titles is None:
            titles = ['Image (%d)' % i for i in range(1, n_images + 1)]
        fig = plt.figure()
        for n, (image, title) in enumerate(zip(images, titles)):
            a = fig.add_subplot(self.figure_cols, np.ceil(n_images / float(self.figure_cols)), n + 1)
            if image.ndim == 2:
                plt.gray()
            plt.imshow(image)
            a.set_title(title)
        fig.set_size_inches(np.array(fig.get_size_inches()) * n_images)
        plt.show()

    def plot_bbox(self, img: np.ndarray, bboxes: List[Union[int, int, int, int, str]]) -> np.ndarray:
        for bbox in bboxes:
            # bbox --> [x_min, y_min, x_max, y_max, 'Text']
            start_point = (int(bbox[0]), int(bbox[1]))
            end_point = (int(bbox[2]), int(bbox[3]))
            # cv2.rectange:  img -> HWC & start_points/end_points --> (int, int)
            img = cv2.rectangle(img, start_point, end_point, self.bbox_color, self.bbox_thickness)
        return img

    def view_augmentations(self, transform_images: List) -> None:
        bbox_imgs = list()
        for img in transform_images:
            bbox_img = self.plot_bbox(img['image'], img['bboxes'])
            bbox_imgs.append(bbox_img)
        self.show_multiple_images(bbox_imgs)

    def __init__(self,
                 bbox_color: Tuple[int, int, int] = (255, 0, 0),
                 bbox_thickness: int = 2,
                 figure_cols: int = 2):
        self.bbox_color = bbox_color
        self.bbox_thickness = bbox_thickness
        self.figure_cols = figure_cols


class AugmentData(object):
    """
    https://albumentations.ai/docs/getting_started/bounding_boxes_augmentation/
    https://albumentations.ai/docs/getting_started/transforms_and_targets/
    """

    def augment_image(self, image: np.ndarray = np.empty((1, 1, 3)),
                      image_bboxes: List[Union[int, int, int, int, str]] = list()) -> List:
        """
        Return:
            [
                dict ( image=np.ndarray, bboxes=List ), ...
            ]
        """
        img_deepcopy = copy.deepcopy(image)
        transform_images = list()
        for _ in range(self.num_of_augmentations):
            transform_images.append(self.transform(image=img_deepcopy, bboxes=image_bboxes))
        return transform_images

    def __init__(self, annotations_format: str = 'pascal_voc',
                 num_of_augmentations: int = 5,
                 transforms=None):

        self.num_of_augmentations = num_of_augmentations
        self.annotations_format = annotations_format

        self.transform = transforms

        if self.transform:
            self.transform = A.Compose(self.transform, bbox_params=A.BboxParams(
                format=self.annotations_format, min_area=300))
        else:
            self.transform = A.Compose([
                A.HorizontalFlip(p=0.8)], bbox_params=A.BboxParams(format=self.annotations_format, min_area=300))


class CnicCvatDataset(Dataset, AugmentData, PlotImagesAnnotations):

    def read_xml_annot(self) -> List:
        """
        Return: {
                'image_name' :
                    [
                        [x_min, y_min, x_max, y_max, 'Text'], ...
                    ],
                ...
                }
        """
        annotations_files = self.annotations_xml
        all_annots = dict()
        tree = ET.parse(annotations_files.as_posix())
        root = tree.getroot()
        for img in root.findall('image'):
            if img.attrib.get('name') not in self.imgs_files:
                # Skip image file that have no annotations
                continue
            bboxes = list()
            for bbox in img.getchildren():
                if bbox.attrib.get('label') == 'Textline':
                    # Rest all labels are assumed to represent word bbox irrespective of language
                    continue
                coordinates = bbox.attrib.get('points').split(';')
                if len(coordinates) != 4:
                    # Augmentation available only for rectangles/squares with 4 coordinates therefore continue
                    continue
                start_point = coordinates[0].split(',')
                end_point = coordinates[2].split(',')
                x_min, y_min = int(float(start_point[0])), int(float(start_point[1]))
                x_max, y_max = int(float(end_point[0])), int(float(end_point[1]))
                if x_min > x_max or y_min > y_max:
                    # if this condition is voilated, that can lead to error in bbox & image transformation
                    continue
                bboxes.append([x_min, y_min, x_max, y_max, self.label])
            if bboxes:
                all_annots[img.attrib.get('name')] = bboxes
        return all_annots

    def __init__(self, img_dir: str = '',
                 annotations_xml: str = '',
                 transforms=None):
        AugmentData.__init__(self, transforms=transforms)
        PlotImagesAnnotations.__init__(self)

        self.img_dir = Path(img_dir)
        self.annotations_xml = Path(annotations_xml)
        self.label = 'Text'

        self.imgs_files = [file.name for file in self.img_dir.iterdir() if imghdr.what(file)]
        # skip annotations that don't have images
        self.annotations = self.read_xml_annot()
        # skip image files that don't have annotations
        self.imgs_files = list(self.annotations.keys())

    def __len__(self):
        return len(self.imgs_files)

    def __getitem__(self, index):

        img_file = self.img_dir / self.imgs_files[index]
        bboxes = self.annotations.get(self.imgs_files[index])

        img = cv2.imread(img_file.as_posix())

        transform_images = self.augment_image(img, bboxes)

        self.view_augmentations(transform_images)


if __name__ == '__main__':

    __all__ = [

    ]
    dataset = CnicCvatDataset('/home/mansoor/Projects/Craft-Training/dataset/task_cnic text annotation '
                              'v1.0-2021_10_21_13_55_32-cvat for images 1.1/Images/train/images',
                              '/home/mansoor/Projects/Craft-Training/dataset/task_cnic text annotation '
                              'v1.0-2021_10_21_13_55_32-cvat for images 1.1/Images/annotations.xml')
    # dataset.plot_save_bboxes(view_plots=True, save_plots=False)
    list([_ for _ in dataset])
