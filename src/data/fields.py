import os
import glob
import random
import h5py

from PIL import Image
import numpy as np
from src.data.core import Field


class IndexField(Field):
    """ Basic index field."""
    def load(self, model_path, idx, category, photo_num):

        """ Loads the index field.
        Args:
            model_path (str): path to model
            idx (int): ID of data point
            category (int): index of category
            photo_num
        """

        return idx

    def check_complete(self, files):

        """ Check if field is complete.
        Args:
            files: files
        """

        return True


class CategoryField(Field):
    """ Basic category field."""
    def load(self, model_path, idx, category, photo_num):

        """ Loads the category field.
        Args:
            model_path (str): path to model
            idx (int): ID of data point
            category (int): index of category
            photo_num
        """

        return category

    def check_complete(self, files):

        """ Check if field is complete.
        Args:
            files: files
        """

        return True


class ImagesField(Field):

    """ Image Field.
    It is the field used for loading images.

    Args:
        folder_name (str): folder name
        transform (): list of transformations applied to loaded images
        extension (str): image extension
        random_view (bool): whether a random view should be used
        with_camera (bool): whether camera data should be provided
    """

    def __init__(self, folder_name, transform=None,
                 extension='jpg', random_view=True, with_camera=False):
        self.folder_name = folder_name
        self.transform = transform
        self.extension = extension
        self.random_view = random_view
        self.with_camera = with_camera

    def load(self, model_path, idx, category, photo_num):

        """ Loads the data point.
        Args:
            model_path (str): path to model
            idx (int): ID of data point
            category (int): index of category
            photo_num (int): index of photo
        """

        folder = os.path.join(model_path, self.folder_name)
        files = sorted(glob.glob(os.path.join(folder, '*.%s' % self.extension)))
        files.sort()

        if self.random_view:
            # idx_img = random.randint(0, len(files)-1)
            filename = files[photo_num]
        else:
            idx_img = 0
            filename = files[idx_img]

        image = Image.open(filename).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)

        data = {
            None: image
        }

        return data

    def check_complete(self, files):

        """ Check if field is complete.
        Args:
            files: files
        """

        complete = (self.folder_name in files)
        # check camera
        return complete


# 3D Fields
class PointsField(Field):

    """ Point Field.
    It provides the field to load point data. This is used for the points
    randomly sampled in the bounding volume of the 3D shape.
    Args:
        file_name (str): file name
        transform : list of transformations which will be applied to the
            points tensor
        with_transforms (bool): whether scaling and rotation data should be
            provided
    """

    def __init__(self, file_name, transform=None, with_transforms=False, unpackbits=False):
        self.file_name = file_name
        self.transform = transform
        self.with_transforms = with_transforms
        self.unpackbits = unpackbits

    def load(self, model_path, idx, category, photo_num):

        """ Loads the data point.
        Args:
            model_path (str): path to model
            idx (int): ID of data point
            category (int): index of category
            photo_num (int): index of photo
        """

        file_path = os.path.join(model_path, self.file_name)
        data_dict = h5py.File(file_path)
        data = data_dict['pc_sdf_sample'][()]
        points = data[:, :3]
        occupancies = data[:, 3]

        # Break symmetry if given in float16:
        if points.dtype == np.float16:
            points = points.astype(np.float32)
            points += 1e-4 * np.random.randn(*points.shape)
        else:
            points = points.astype(np.float32)

        occupancies = occupancies.astype(np.float32)

        data = {
            None: points,
            'occ': occupancies,
        }

        if self.with_transforms:
            data['loc'] = data_dict['loc'].astype(np.float32)
            data['scale'] = data_dict['scale'].astype(np.float32)

        if self.transform is not None:
            data = self.transform(data)

        return data

    def check_complete(self, files):
        pass
