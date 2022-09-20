from src import data
from torchvision import transforms


def get_dataset(mode, cfg, return_idx=False, return_category=False):

    """ Returns the dataset.
    Args:
        mode (nn.Module): the model which is used
        cfg (dict): config dictionary
        return_idx (bool): whether to include an ID field
        return_category: whether to include category info
    """

    dataset_type = cfg['data']['dataset']
    dataset_folder = cfg['data']['path']
    categories = cfg['data']['classes']

    # Get split
    splits = {
        'train': cfg['data']['train_split'],
        'val': cfg['data']['val_split'],
        'test': cfg['data']['test_split'],
    }

    split = splits[mode]

    # Create dataset
    if dataset_type == 'train_set':
        # Dataset fields
        # Method specific fields (usually correspond to output)
        fields = get_data_fields(mode, cfg)
        # Input fields
        inputs_field = get_inputs_field(mode, cfg)
        if inputs_field is not None:
            fields['inputs'] = inputs_field

        if return_idx:
            fields['idx'] = data.IndexField()

        if return_category:
            fields['category'] = data.CategoryField()

        dataset = data.Shapes3dDataset(
            dataset_folder, fields,
            split=split,
            categories=categories,
        )
    elif dataset_type == 'test_set':
        dataset = data.ImageDataset(
            dataset_folder, img_size=cfg['data']['img_size'],
            return_idx=return_idx,
        )
    else:
        raise ValueError('Invalid dataset "%s"' % cfg['data']['dataset'])

    return dataset


def get_data_fields(mode, cfg):

    """ Returns the data fields.
    Args:
        mode (str): the mode which is used
        cfg (dict): imported yaml config
    """

    # fields = {}
    points_transform = data.SubsamplePoints(cfg['data']['points_subsample'])
    with_transforms = cfg['model']['use_camera']

    fields = {'points': data.PointsField(
        cfg['data']['points_file'], points_transform,
        with_transforms=with_transforms,
        unpackbits=cfg['data']['points_unpackbits'],
    )}

    if mode in ('val', 'test'):
        points_iou_file = cfg['data']['points_iou_file']
        if points_iou_file is not None:
            fields['points_iou'] = data.PointsField(
                points_iou_file,
                with_transforms=with_transforms,
                unpackbits=cfg['data']['points_unpackbits'],
            )

    return fields


def get_inputs_field(mode, cfg):

    """ Returns the inputs fields.
    Args:
        mode (str): the mode which is used
        cfg (dict): config dictionary
    """

    input_type = cfg['data']['input_type']

    if input_type is None:
        inputs_field = None
    elif input_type == 'img':
        if mode == 'train' and cfg['data']['img_augment']:
            resize_op = transforms.RandomResizedCrop(
                cfg['data']['img_size'], (0.75, 1.), (1., 1.))
        else:
            resize_op = transforms.Resize((cfg['data']['img_size']))

        transform = transforms.Compose([
            resize_op, transforms.ToTensor(),
        ])

        with_camera = cfg['data']['img_with_camera']

        if mode == 'train':
            random_view = True
        else:
            random_view = False

        inputs_field = data.ImagesField(
            cfg['data']['img_folder'], transform,
            with_camera=with_camera, random_view=random_view
        )
    else:
        raise ValueError(
            'Invalid input type (%s)' % input_type)
    return inputs_field
