import logging
import numpy as np
import yaml
import os
from PIL import Image
import torch
from torch.utils import data
from torchvision import transforms


logger = logging.getLogger(__name__)


# Fields
class Field(object):

    """ Data fields class."""

    def load(self, data_path, idx, category, photo_num):

        """ Loads a data point.
        Args:
            data_path (str): path to data file
            idx (int): index of data point
            category (int): index of category
            photo_num (int): index of photo
        """
        raise NotImplementedError

    def check_complete(self, files):

        """ Checks if set is complete.
        Args:
            files: files
        """
        raise NotImplementedError


class Shapes3dDataset(data.Dataset):

    """ 3D Shapes dataset class."""

    def __init__(self, dataset_folder, fields, split=None,
                 categories=None, no_except=True, transform=None):
        """ Initialization of the the 3D shape dataset.
        Args:
            dataset_folder (str): dataset folder
            fields (dict): dictionary of fields
            split (str): which split is used
            categories (list): list of categories to use
            no_except (bool): no exception
            transform (callable): transformation applied to data points
        """
        # Attributes
        self.dataset_folder = dataset_folder
        self.fields = fields
        self.no_except = no_except
        self.transform = transform

        # If categories is None, use all subfolders
        if categories is None:
            categories = os.listdir(dataset_folder)
            categories = [c for c in categories
                          if os.path.isdir(os.path.join(dataset_folder, c))]

        # Read metadata file
        metadata_file = os.path.join(dataset_folder, 'metadata.yaml')

        if os.path.exists(metadata_file):
            with open(metadata_file, 'r') as f:
                self.metadata = yaml.load(f)
        else:
            self.metadata = {
                c: {'id': c, 'name': 'n/a'} for c in categories
            } 
        
        # Set index
        for c_idx, c in enumerate(categories):
            self.metadata[c]['idx'] = c_idx

        # Get all models
        self.models = []
        if split == 'train':
            for c_idx, c in enumerate(categories):
                subpath = os.path.join(dataset_folder, c)
                if not os.path.isdir(subpath):
                    logger.warning('Category %s does not exist in dataset.' % c)

                split_file = os.path.join(subpath, split + '.lst')
                with open(split_file, 'r') as f:
                    models_c = f.read().split('\n')

                for m in models_c:
                    for num in range(4):
                        self.models += [
                            {'category': c, 'model': m, 'photo_num': num}
                        ]
        else:
            for c_idx, c in enumerate(categories):
                subpath = os.path.join(dataset_folder, c)
                if not os.path.isdir(subpath):
                    logger.warning('Category %s does not exist in dataset.' % c)

                split_file = os.path.join(subpath, split + '.lst')
                with open(split_file, 'r') as f:
                    models_c = f.read().split('\n')

                for m in models_c:
                    self.models += [
                        {'category': c, 'model': m, 'photo_num': 0}
                    ]

    def __len__(self):
        """ Returns the length of the dataset."""
        return len(self.models)

    def __getitem__(self, idx):
        """ Returns an item of the dataset.
        Args:
            idx (int): ID of data point
        """
        category = self.models[idx]['category']
        model = self.models[idx]['model']
        photo_num = self.models[idx]['photo_num']
        c_idx = self.metadata[category]['idx']

        model_path = os.path.join(self.dataset_folder, category, model)
        data_load = {}

        for field_name, field in self.fields.items():
            # noinspection PyBroadException
            try:
                field_data = field.load(model_path, idx, c_idx, photo_num)
            except Exception:
                if self.no_except:
                    logger.warning(
                        'Error occured when loading field %s of model %s'
                        % (field_name, model)
                    )
                    return None
                else:
                    raise

            if isinstance(field_data, dict):
                for k, v in field_data.items():
                    if k is None:
                        data_load[field_name] = v
                    else:
                        data_load['%s.%s' % (field_name, k)] = v
            else:
                data_load[field_name] = field_data

        if self.transform is not None:
            data_load = self.transform(data_load)

        return data_load

    def get_model_dict(self, idx):
        return self.models[idx]

    def test_model_complete(self, category, model):
        """ Tests if model is complete.
        Args:
            model (str): modelname
        """
        model_path = os.path.join(self.dataset_folder, category, model)
        files = os.listdir(model_path)
        for field_name, field in self.fields.items():
            if not field.check_complete(files):
                logger.warning('Field "%s" is incomplete: %s' % (field_name, model_path))
                return False
        return True


IMAGE_EXTENSIONS = (
    '.jpg', '.jpeg', '.JPG', '.JPEG', '.png', '.PNG'
)


class ImageDataset(data.Dataset):
    r""" Image Dataset.
    Args:
        dataset_folder (str): path to the dataset dataset
        img_size (int): size of the cropped images
        transform (list): list of transformations applied to the data points
    """

    def __init__(self, dataset_folder, img_size=224, transform=None, return_idx=False):

        """
        Arguments:
            dataset_folder (path): path to the Image dataset
            img_size (int): required size of the cropped images
            return_idx (bool): wether to return index
        """

        self.img_size = img_size
        self.img_path = dataset_folder
        self.file_list = os.listdir(self.img_path)
        self.file_list = [
            f for f in self.file_list
            if os.path.splitext(f)[1] in IMAGE_EXTENSIONS
        ]
        self.len = len(self.file_list)
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor()
            ])
        self.return_idx = return_idx

    def get_model(self, idx):
        """ Returns the model.
        Args:
            idx (int): ID of data point
        """
        f_name = os.path.basename(self.file_list[idx])
        f_name = os.path.splitext(f_name)[0]
        return f_name

    def get_model_dict(self, idx):
        f_name = os.path.basename(self.file_list[idx])
        model_dict = {
            'model': f_name
        }
        return model_dict

    def __len__(self):
        """ Returns the length of the dataset."""
        return self.len

    def __getitem__(self, idx):
        """ Returns the data point.
        Args:
            idx (int): ID of data point
        """
        f = os.path.join(self.img_path, self.file_list[idx])
        img_in = Image.open(f)
        img = Image.new("RGB", img_in.size)
        img.paste(img_in)
        if self.transform:
            img = self.transform(img)

        idx = torch.tensor(idx)

        data_img = {
            'inputs': img,
        }

        if self.return_idx:
            data_img['idx'] = idx

        return data_img


class SubsamplePoints(object):

    """ Points subsampling transformation class.
        It subsamples the points data.
    Args:
        N (int): number of points to be subsampled
    """

    def __init__(self, N):
        self.N = N

    def __call__(self, data_in):

        """ Calls the transformation.
        Args:
            data_in (dictionary): data dictionary
        """

        points = data_in[None]
        occ = data_in['occ']

        data_out = data_in.copy()
        if isinstance(self.N, int):
            idx = np.random.randint(points.shape[0], size=self.N)
            data_out.update({
                None: points[idx, :],
                'occ':  occ[idx],
            })
        else:
            Nt_out, Nt_in = self.N
            occ_binary = (occ >= 0.5)
            points0 = points[~occ_binary]
            points1 = points[occ_binary]

            idx0 = np.random.randint(points0.shape[0], size=Nt_out)
            idx1 = np.random.randint(points1.shape[0], size=Nt_in)

            points0 = points0[idx0, :]
            points1 = points1[idx1, :]
            points = np.concatenate([points0, points1], axis=0)

            occ0 = np.zeros(Nt_out, dtype=np.float32)
            occ1 = np.ones(Nt_in, dtype=np.float32)
            occ = np.concatenate([occ0, occ1], axis=0)

            volume = occ_binary.sum() / len(occ_binary)
            volume = volume.astype(np.float32)

            data_out.update({
                None: points,
                'occ': occ,
                'volume': volume,
            })
        return data_out


def collate_remove_none(batch):

    """ Collater that puts each data field into a tensor with outer dimension batch size.
    Args:
        batch: batch
    """

    batch = list(filter(lambda x: x is not None, batch))
    return data.dataloader.default_collate(batch)


def worker_init_fn(worker_id):
    """ Worker init function to ensure true randomness."""
    random_data = os.urandom(4)
    base_seed = int.from_bytes(random_data, byteorder="big")
    np.random.seed(base_seed + worker_id)
