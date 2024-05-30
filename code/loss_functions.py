import torch
from utils import geo_utils
from torch import nn
from torch.nn import functional as F
from pytorch3d import transforms as py3d_trans


class ESFMLoss(nn.Module):
    def __init__(self, conf):
        super().__init__()
        self.infinity_pts_margin = conf.get_float("loss.infinity_pts_margin")
        self.normalize_grad = conf.get_bool("loss.normalize_grad")

        self.hinge_loss = conf.get_bool("loss.hinge_loss")
        if self.hinge_loss:
            self.hinge_loss_weight = conf.get_float("loss.hinge_loss_weight")
        else:
            self.hinge_loss_weight = 0

    def forward(self, pred_cam, data, epoch=None):
        Ps = pred_cam["Ps_norm"]
        pts_2d = Ps @ pred_cam["pts3D"]  # [m, 3, n]

        # Normalize gradient
        if self.normalize_grad:
            pts_2d.register_hook(lambda grad: F.normalize(grad, dim=1) / data.valid_pts.sum())

        # Get point for reprojection loss
        if self.hinge_loss:
            projected_points = geo_utils.get_positive_projected_pts_mask(pts_2d, self.infinity_pts_margin)
        else:
            projected_points = geo_utils.get_projected_pts_mask(pts_2d, self.infinity_pts_margin)

        # Calculate hinge Loss
        hinge_loss = (self.infinity_pts_margin - pts_2d[:, 2, :]) * self.hinge_loss_weight

        # Calculate reprojection error
        pts_2d = (pts_2d / torch.where(projected_points, pts_2d[:, 2, :], torch.ones_like(projected_points).float()).unsqueeze(dim=1))
        reproj_err = (pts_2d[:, 0:2, :] - data.norm_M.reshape(Ps.shape[0], 2, -1)).norm(dim=1)

        return torch.where(projected_points, reproj_err, hinge_loss)[data.valid_pts].mean()
    
class RelativeGTLoss(nn.Module):
    def __init__(self, conf):
        super().__init__()
        self.calibrated = conf.get_bool('dataset.calibrated')

    def forward(self, pred_cam, data, epoch=None):
        R_gt = data.y[:, 0:3, 0:3]
        #R_gt_rel = R_gt[1:] @ R_gt[:-1].inverse()
        t_gt = -torch.bmm(R_gt.inverse(), data.y[:, 0:3, 3].unsqueeze(-1))
        t_gt_rel = t_gt[1:] - t_gt[:-1]
        #R_gt_quat = py3d_trans.matrix_to_quaternion(R_gt_rel)
        t_gt_rel = t_gt_rel.squeeze()

        R_pred = pred_cam["Ps_norm"][:, 0:3, 0:3]
        #R_pred_rel = R_pred[1:] @ R_pred[:-1].transpose(1,2)
        t_pred = -torch.bmm(R_pred.transpose(1, 2), pred_cam["Ps_norm"][:, 0:3, 3].unsqueeze(dim=-1))
        t_pred_rel = t_pred[1:] - t_pred[:-1]
        #R_pred_quat = py3d_trans.matrix_to_quaternion(R_pred_rel)
        t_pred_rel = t_pred_rel.squeeze()

        translation_loss = (t_gt_rel - t_pred_rel).norm(p=2, dim=1).mean()
        #rotation_loss = (R_gt_quat - R_pred_quat).norm(p=2, dim=1).mean()
        loss = translation_loss #+ rotation_loss
        return loss


class ScaleFactorLoss(nn.Module):
    def __init__(self, conf):
        super().__init__()
        self.calibrated = conf.get_bool('dataset.calibrated')

    def forward(self, pred_cam, data, epoch=None):
        R_gt = data.y[:, 0:3, 0:3]
        R_gt_quat = py3d_trans.matrix_to_quaternion(R_gt)
        t_gt = -torch.bmm(R_gt.inverse(), data.y[:, 0:3, 3].unsqueeze(-1)).squeeze()

        Rs_pred = pred_cam["Ps_norm"][:, 0:3, 0:3]
        Rs_pred_quat = py3d_trans.matrix_to_quaternion(Rs_pred)
        ts = -torch.bmm(Rs_pred.transpose(1, 2), pred_cam["Ps_norm"][:, 0:3, 3].unsqueeze(dim=-1)).squeeze()
        translation_loss = (t_gt - ts).norm(p=2, dim=1).mean()
        rotation_loss = (Rs_pred_quat - R_gt_quat).norm(p=2, dim=1).mean()
        loss = translation_loss + rotation_loss
        return loss

class GTLoss(nn.Module): # Not tested
    def __init__(self, conf):
        super().__init__()
        self.calibrated = conf.get_bool('dataset.calibrated')

    def forward(self, pred_cam, data, epoch=None):
        #Get orientation
        Vs_gt = data.y[:, 0:3, 0:3].transpose(1, 2)
        if self.calibrated:
            #Rs_gt = geo_utils.rot_to_quat(torch.bmm(data.Ns_invT, Vs_gt).transpose(1, 2))
            Rs_gt = py3d_trans.matrix_to_quaternion(Vs_gt)

        # Get Location
        t_gt = -torch.bmm(Vs_gt, data.y[:, 0:3, 3].unsqueeze(-1)).squeeze()

        # Normalize scene by cameras
        trans = t_gt.mean(dim=0)
        scale = (t_gt - trans).norm(p=2, dim=1).mean()

        t_gt = (t_gt - trans)/scale
        new_Ps = geo_utils.batch_get_camera_matrix_from_Vt(Vs_gt, t_gt)

        Vs = pred_cam["Ps_norm"][:, 0:3, 0:3].transpose(1, 2)
        ts = -torch.bmm(Vs, pred_cam["Ps_norm"][:, 0:3, 3].unsqueeze(dim=-1)).squeeze()

        # Translation error
        translation_err = (t_gt - ts).norm(p=2, dim=1)

        # Calculate error
        if self.calibrated:
            # Rs = geo_utils.rot_to_quat(torch.bmm(data.Ns_invT, Vs).transpose(1, 2))
            Rs = py3d_trans.matrix_to_quaternion(Vs)
            orient_err = (Rs - Rs_gt).norm(p=2, dim=1)
        else:
            Vs_gt = Vs_gt / Vs_gt.norm(p='fro', dim=(1, 2), keepdim=True)
            Vs = Vs / Vs.norm(p='fro', dim=(1, 2), keepdim=True)
            orient_err = torch.min((Vs - Vs_gt).norm(p='fro', dim=(1, 2)), (Vs + Vs_gt).norm(p='fro', dim=(1, 2)))

        orient_loss = orient_err.mean()
        tran_loss = translation_err.mean()
        loss = tran_loss + orient_loss 

        if epoch is not None and epoch % 1000 == 0:
            # Print loss
            print("loss = {}, orient err = {}, trans err = {}".format(loss, orient_loss, tran_loss))
    
        # # Groundtruth
        # Vs_gt = data.y
        # Rs_gt = Vs_gt[:, 0:3, 0:3].transpose(1, 2)
        # ts_gt = -torch.bmm(Rs_gt.transpose(1,2), Vs_gt[:, 0:3, 3].unsqueeze(-1)).squeeze()

        # # # Normalize scene by cameras
        # trans = ts_gt.mean(dim=0)
        # scale = (ts_gt - trans).norm(p=2, dim=1).mean()
        # ts_gt = (ts_gt - trans)/scale

        # # # Pred
        # Vs = pred_cam["Ps_norm"]
        # Rs_pred = Vs[:, 0:3, 0:3].transpose(1, 2)
        # ts_pred = -torch.bmm(Rs_pred.transpose(1,2), Vs[:, 0:3, 3].unsqueeze(-1)).squeeze()

        # # Rs_gt_quat = py3d_trans.matrix_to_quaternion(torch.bmm(data.Ns_invT, Rs_gt.transpose(1,2)))
        # # Rs_pred_quat = py3d_trans.matrix_to_quaternion(torch.bmm(data.Ns_invT, Rs_pred.transpose(1,2)))

        # translation_err = (ts_gt - ts_pred).norm(p=2, dim=1).mean()
        # # rotation_err = (Rs_pred_quat - Rs_gt_quat).norm(p=2, dim=1).mean()

        # loss = translation_err #+ rotation_err

        return loss

class CombinedLoss(nn.Module):
    def __init__(self, conf):
        super().__init__()
        #self.gtloss = ScaleFactorLoss(conf)
        self.esfmloss = ESFMLoss(conf)
        self.scaleloss = RelativeGTLoss(conf)

    def forward(self, pred_cam, data, epoch=None):
        #loss_gt = self.gtloss(pred_cam, data, epoch)
        loss_esfm = self.esfmloss(pred_cam, data, epoch)
        loss_scalefactor = self.scaleloss(pred_cam, data, epoch)
        return loss_scalefactor + loss_esfm
