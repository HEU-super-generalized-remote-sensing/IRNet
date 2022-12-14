import os
import torch
import torch.nn as nn
import numpy as np
from tqdm import trange, tqdm
from collections import defaultdict
from torch.nn import functional as F
from torch import distributions as dist
from src.models.common import compute_iou, make_3d_grid
from src.utils import visualize as vis


class BaseTrainer(object):
    """ Base trainer class."""

    def evaluate(self, val_loader):
        """ Performs an evaluation.
        Args:
            val_loader (dataloader): pytorch dataloader
        """
        eval_list = defaultdict(list)

        for data in tqdm(val_loader):
            eval_step_dict = self.eval_step(data)

            for k, v in eval_step_dict.items():
                eval_list[k].append(v)

        eval_dict = {k: np.mean(v) for k, v in eval_list.items()}
        return eval_dict

    def evaluate_sdf(self, val_loader):
        """ Performs an evaluation.
        Args:
            val_loader (dataloader): pytorch dataloader
        """
        eval_list = defaultdict(list)

        for data in tqdm(val_loader):
            eval_step_dict = self.eval_sdf_step(data)

            for k, v in eval_step_dict.items():
                eval_list[k].append(v)

        eval_dict = {k: np.mean(v) for k, v in eval_list.items()}
        return eval_dict

    def train_step(self, *args, **kwargs):
        """ Performs a training step."""
        raise NotImplementedError

    def eval_step(self, *args, **kwargs):
        """ Performs an evaluation step."""
        raise NotImplementedError

    def eval_sdf_step(self, *args, **kwargs):
        """ Performs an evaluation sdf step."""
        raise NotImplementedError

    def visualize(self, *args, **kwargs):
        """ Performs  visualization."""
        raise NotImplementedError


class Trainer(BaseTrainer):

    """ Trainer object for the Occupancy Network.
    Args:
        model (nn.Module): Occupancy Network model
        optimizer (optimizer): pytorch optimizer object
        device (device): pytorch device
        input_type (str): input type
        vis_dir (str): visualization directory
        threshold (float): threshold value
        eval_sample (bool): whether to evaluate samples

    """

    def __init__(self, model, optimizer, device=None, input_type='img',
                 vis_dir=None, threshold=0.5, eval_sample=False):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.input_type = input_type
        self.vis_dir = vis_dir
        self.threshold = threshold
        self.eval_sample = eval_sample
        self.activate = nn.ReLU(inplace=False)

        if vis_dir is not None and not os.path.exists(vis_dir):
            os.makedirs(vis_dir)

    def train_step(self, data):
        """ Performs a training step.
        Args:
            data (dict): data dictionary
        """
        self.model.train()
        self.optimizer.zero_grad()
        loss = self.compute_loss(data)
        loss.backward()
        # print(loss.grad)
        self.optimizer.step()
        return loss.item()

    def eval_step(self, data):
        """ Performs an evaluation step.
        Args:
            data (dict): data dictionary
        """
        self.model.eval()

        device = self.device
        threshold = self.threshold
        eval_dict = {}

        # Compute elbo
        points = data.get('points').to(device)
        occ = data.get('points.occ').to(device)

        inputs = data.get('inputs', torch.empty(points.size(0), 0)).to(device)
        voxels_occ = data.get('voxels')

        points_iou = data.get('points_iou').to(device)
        occ_iou = data.get('points_iou.occ').to(device)

        kwargs = {}

        with torch.no_grad():
            elbo, rec_error, kl = self.model.compute_elbo(
                points, occ, inputs, **kwargs)

        eval_dict['loss'] = -elbo.mean().item()
        eval_dict['rec_error'] = rec_error.mean().item()
        eval_dict['kl'] = kl.mean().item()

        # Compute iou
        batch_size = points.size(0)

        with torch.no_grad():
            p_out = self.model(points_iou, inputs,
                               sample=self.eval_sample, **kwargs)

        occ_iou_np = (occ_iou >= 0.5).cpu().numpy()
        occ_iou_hat_np = (p_out.probs >= threshold).cpu().numpy()
        iou = compute_iou(occ_iou_np, occ_iou_hat_np).mean()
        eval_dict['iou'] = iou

        # Estimate voxel iou
        if voxels_occ is not None:
            voxels_occ = voxels_occ.to(device)
            points_voxels = make_3d_grid(
                (-0.5 + 1/64,) * 3, (0.5 - 1/64,) * 3, (32,) * 3)
            points_voxels = points_voxels.expand(
                batch_size, *points_voxels.size())
            points_voxels = points_voxels.to(device)
            with torch.no_grad():
                p_out = self.model(points_voxels, inputs,
                                   sample=self.eval_sample, **kwargs)

            voxels_occ_np = (voxels_occ >= 0.5).cpu().numpy()
            occ_hat_np = (p_out.probs >= threshold).cpu().numpy()
            iou_voxels = compute_iou(voxels_occ_np, occ_hat_np).mean()

            eval_dict['iou_voxels'] = iou_voxels

        return eval_dict

    def eval_sdf_step(self, data):
        """ Performs an evaluation step.
        Args:
            data (dict): data dictionary
        """
        self.model.eval()

        device = self.device
        threshold = self.threshold
        eval_dict = {}

        # Compute elbo
        points = data.get('points').to(device)
        occ = data.get('points.occ').to(device)

        inputs = data.get('inputs', torch.empty(points.size(0), 0)).to(device)
        voxels_occ = data.get('voxels')

        points_iou = data.get('points_iou').to(device)
        occ_iou = data.get('points_iou.occ').to(device)

        kwargs = {}

        with torch.no_grad():
            elbo, rec_error, kl = self.model.compute_sdf_elbo(
                points, occ, inputs, **kwargs)

        eval_dict['loss'] = elbo.mean().item()
        eval_dict['rec_error'] = rec_error.mean().item()
        eval_dict['kl'] = kl.mean().item()

        # Compute iou
        batch_size = points.size(0)

        with torch.no_grad():
            p_out = self.model(points_iou, inputs,
                               sample=self.eval_sample, **kwargs)

        occ_iou_np = (occ_iou >= 0).cpu().numpy()
        occ_iou_hat_np = (p_out.logits >= 0).cpu().numpy()
        iou = (occ_iou_np == occ_iou_hat_np).astype(np.int16)
        iou = np.sum(iou) / occ_iou_np.shape[1]
        eval_dict['iou'] = iou

        # Estimate voxel iou
        if voxels_occ is not None:
            voxels_occ = voxels_occ.to(device)
            points_voxels = make_3d_grid(
                (-0.5 + 1 / 64,) * 3, (0.5 - 1 / 64,) * 3, (32,) * 3)
            points_voxels = points_voxels.expand(
                batch_size, *points_voxels.size())
            points_voxels = points_voxels.to(device)
            with torch.no_grad():
                p_out = self.model(points_voxels, inputs,
                                   sample=self.eval_sample, **kwargs)

            voxels_occ_np = (voxels_occ >= 0.5).cpu().numpy()
            occ_hat_np = (p_out.probs >= threshold).cpu().numpy()
            iou_voxels = compute_iou(voxels_occ_np, occ_hat_np).mean()

            eval_dict['iou_voxels'] = iou_voxels

        return eval_dict

    def visualize(self, data):
        """ Performs a visualization step for the data.
        Args:
            data (dict): data dictionary
        """
        device = self.device

        batch_size = data['points'].size(0)
        inputs = data.get('inputs', torch.empty(batch_size, 0)).to(device)

        shape = (32, 32, 32)
        p = make_3d_grid([-0.5] * 3, [0.5] * 3, shape).to(device)
        p = p.expand(batch_size, *p.size())

        kwargs = {}
        with torch.no_grad():
            p_r = self.model(p, inputs, sample=self.eval_sample, **kwargs)

        occ_hat = p_r.probs.view(batch_size, *shape)
        voxels_out = (occ_hat >= self.threshold).cpu().numpy()

        for i in trange(batch_size):
            input_img_path = os.path.join(self.vis_dir, '%03d_in.png' % i)
            vis.visualize_data(
                inputs[i].cpu(), self.input_type, input_img_path)
            vis.visualize_voxels(
                voxels_out[i], os.path.join(self.vis_dir, '%03d.png' % i))

    def compute_loss(self, data):
        """ Computes the loss.
        Args:
            data (dict): data dictionary
        """
        device = self.device
        p = data.get('points').to(device)
        gt_sdf = data.get('points.occ').to(device)
        inputs = data.get('inputs', torch.empty(p.size(0), 0)).to(device)

        kwargs = {}

        c = self.model.encode_inputs(inputs)
        q_z = self.model.infer_z(p, gt_sdf, c, **kwargs)
        z = q_z.rsample()

        # KL-divergence
        # kl = dist.kl_divergence(q_z, self.model.p0_z).sum(dim=-1)
        # loss = kl.mean()

        # General points
        pred_sdf = self.model.decode(p, z, c, **kwargs).logits
        t = torch.tensor(0.02).to(device)
        weight_mask = torch.add(torch.ge(gt_sdf, -t), torch.ge(t, gt_sdf)).pow(2)
        sdf_loss = torch.mul(torch.mean(torch.mul(torch.abs(pred_sdf - gt_sdf * 10), weight_mask.float())), 1000)

        return sdf_loss
