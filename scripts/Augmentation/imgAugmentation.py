from torch.utils.data import Dataset
import xml.etree.ElementTree as ET
from matplotlib import pyplot as plt
from typing import List, Union, Tuple, Optional, Sequence
from operator import itemgetter
import albumentations as A
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

    @staticmethod
    def clip_or_remove_bboxes(bboxes: List[Union[int, int, int, int, str]], rows: int, cols: int,
                              remove_bboxes: bool = True) -> List[Union[int, int, int, int, str]]:
        """
        Params:
            bboxes -->   [ [x_min, y_min, x_max, y_max, 'Text'],
                            [x_min, y_min, x_max, y_max, 'Text'],  ... ]
        Objective:
            Either remove the bboxes with out-of-bound coordinates (Recommended) OR clip them to max/min of image shape

        Issues:
            np.asarray --> this operation ultimately converts all the datatypes into single datatypes.
                            because out {bboxes} have both str and int values therefore asarray function
                            converts int to str values. Even converting types of few columns, using astypes,
                            would not be helpfull.
            np.concatenate --> this function will behave same as above. If we try to cocatenate or stack int values
                            matrix and str matrix then the resulting matrix dataype will be str only.
                            Thus, to avoid this issue we convert the matrix into list and merge them.
        """

        bboxes = np.asarray(bboxes)
        # separate bboxes labels and coordinates:
        labels = bboxes[:, -1]
        coordinates = bboxes[:, [0, 1, 2, 3]].astype('float')
        if not remove_bboxes:
            """
            Note: this approach is not best and reliable since you may get the error:
                >> y_max is less than or equal to y_min for bbox
                Therefore, It highly recommended to remove the out-of-bound bboxes.
            """
            coordinates[:, [0, 2]] = coordinates[:, [0, 2]] / cols
            coordinates[:, [1, 3]] = coordinates[:, [1, 3]] / rows
            coordinates = np.clip(coordinates, 0, 1)
            coordinates[:, [0, 2]] = coordinates[:, [0, 2]] * cols
            coordinates[:, [1, 3]] = coordinates[:, [1, 3]] * rows
        else:
            # find out the rows/bboxes that have coordinates out of the range [0, 1]
            outOfBond_rows = list(set(np.where((coordinates[:, [0, 2]] > cols) | (coordinates[:, [0, 2]] < 0) |
                                               (coordinates[:, [1, 3]] > rows) | (coordinates[:, [1, 3]] < 0))[0]))
            coordinates = np.delete(coordinates, outOfBond_rows, 0)
            labels = np.delete(labels, outOfBond_rows, 0)
        coordinates, labels = coordinates.tolist(), labels.tolist()
        [coord.append(label) for coord, label in zip(coordinates, labels)]
        return coordinates

    def augment_image(self, image: np.ndarray = np.empty((1, 1, 3)),
                      image_bboxes: List[Union[int, int, int, int, str]] = list()) -> List:
        """
        Return:
            [
                dict (image=np.ndarray, bboxes=List ), ...
            ]
        """
        img_deepcopy = copy.deepcopy(image)
        _height, _width, _ = image.shape
        # Note: we clip the bbox coordinates because CVAT can make points outside image which results in
        #       error while using annotations_format 'pascal_voc' to normalize the coordinates.
        image_bboxes = self.clip_or_remove_bboxes(image_bboxes, _height, _width)
        transform_images = list()
        for _ in range(self.num_of_augmentations):
            transform_images.append(self.transform(image=img_deepcopy, bboxes=image_bboxes))
        return transform_images

    @staticmethod
    def pixel_level_transforms() -> List:
        return list([
            A.ColorJitter(p=0.7),
            A.Downscale(p=0.4),
            A.ChannelShuffle(p=0.3),
            A.ChannelDropout(p=0.2),
            A.HueSaturationValue(p=0.2),
            A.MotionBlur(p=0.1),
        ])

    @staticmethod
    def spatial_level_transforms() -> List:
        return list([
            A.IAAAffine(p=0.7),
            A.IAAPerspective(p=0.6),
            A.Flip(p=0.7),
            A.RandomScale(p=0.7),
            A.Transpose(p=0.15),
        ])

    def __init__(self, annotations_format: str = 'pascal_voc',  # albumentations
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
                A.OneOf(self.pixel_level_transforms(), p=1),
                A.OneOf(self.spatial_level_transforms(), p=1),
                A.Resize(height=800, width=600, always_apply=True, p=1)], p=1,
                bbox_params=A.BboxParams(format=self.annotations_format))




class AnyDataset(Dataset, AugmentData, PlotImagesAnnotations):

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
    dataset = CnicCvatDataset('/home/mansoor/Projects/Craft-Training/dataset/task_cnic text annotation '
                              'v1.0-2021_10_21_13_55_32-cvat for images 1.1/Images/train/images',
                              '/home/mansoor/Projects/Craft-Training/dataset/task_cnic text annotation '
                              'v1.0-2021_10_21_13_55_32-cvat for images 1.1/Images/annotations.xml', )
    # dataset.plot_save_bboxes(view_plots=True, save_plots=False)
    list([_ for _ in dataset])