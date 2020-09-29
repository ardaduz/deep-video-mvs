import numpy as np
import torch
from torch import svd

from .base_model import BaseModel


def homogeneous_to_euclidean(points):
    """Converts homogeneous points to euclidean

    Args:
        points numpy array or torch tensor of shape (N, M + 1): N homogeneous points of dimension M

    Returns:
        numpy array or torch tensor of shape (N, M): euclidean points
    """
    if isinstance(points, np.ndarray):
        return (points.T[:-1] / points.T[-1]).T
    elif torch.is_tensor(points):
        return (points.transpose(1, 0)[:-1] / points.transpose(1, 0)[-1]).transpose(1, 0)
    else:
        raise TypeError("Works only with numpy arrays and PyTorch tensors.")


def triangulate_point_from_multiple_views_linear_torch_batch(proj_matricies, points, confidences=None):
    """Similar as triangulate_point_from_multiple_views_linear() but for PyTorch.
    For more information see its documentation.
    Args:
        proj_matricies torch tensor of shape (N, 3, 4): sequence of projection matricies (3x4)
        points torch tensor of of shape (N, 2): sequence of points' coordinates
        confidences None or torch tensor of shape (N,): confidences of points [0.0, 1.0].
                                                        If None, all confidences are supposed to be 1.0
    Returns:
        point_3d numpy torch tensor of shape (3,): triangulated point
    """

    assert len(proj_matricies) == len(points)

    n_views = len(proj_matricies)

    if confidences is None:
        confidences = torch.ones(points.shape[1], n_views, dtype=torch.float32, device=points.device)

    ##multiple points
    points_t = points.transpose(0, 1)
    proj_mat = proj_matricies[:, 2:3].expand(n_views, 2, 4).unsqueeze(0)
    points_tview = points_t.view(points_t.size(0), n_views, 2, 1).expand(points_t.size(0), n_views, 2, 4)
    A_all = proj_mat * points_tview
    A_all -= proj_matricies[:, :2].unsqueeze(0)

    A_all *= confidences.view(confidences.size(0), n_views, 1, 1)

    A_all = A_all.contiguous().view(A_all.size(0), A_all.size(1) * A_all.size(2), 4)

    U, S, V = svd(A_all)

    points_3d_homo_all = -V[:, :, 3]
    points_3d = homogeneous_to_euclidean(points_3d_homo_all)

    return points_3d


def triangulate_batch_of_points(proj_matricies_batch, points_batch, confidences_batch=None):
    """Triangulates for a batch of points"""
    batch_size, n_views = proj_matricies_batch.shape[:2]

    points_3d_batch = []
    for batch_i in range(batch_size):
        n_points = points_batch[batch_i].shape[1]
        points = points_batch[batch_i]
        confidences = confidences_batch[batch_i] if confidences_batch is not None else None
        points_3d = triangulate_point_from_multiple_views_linear_torch_batch(proj_matricies_batch[batch_i], points, confidences=confidences)
        points_3d_batch.append(points_3d)

    return points_3d_batch


def integrate_tensor_2d(heatmaps, softmax=True):  # ,temperature = 1.0):
    """Applies softmax to heatmaps and integrates them to get their's "center of masses"
    Args:
        heatmaps torch tensor of shape (batch_size, n_heatmaps, h, w): input heatmaps
    Returns:
        coordinates torch tensor of shape (batch_size, n_heatmaps, 2): coordinates of center of masses of all heatmaps
    """
    batch_size, n_heatmaps, h, w = heatmaps.shape

    heatmaps = heatmaps.reshape((batch_size, n_heatmaps, -1))

    if softmax:
        heatmaps = torch.nn.functional.softmax(heatmaps, dim=2)
    else:
        heatmaps = torch.nn.functional.relu(heatmaps)

    heatmaps = heatmaps.reshape((batch_size, n_heatmaps, h, w))

    mass_x = heatmaps.sum(dim=2)
    mass_y = heatmaps.sum(dim=3)

    mass_times_coord_x = mass_x * torch.arange(w).type(torch.float).to(mass_x.device)
    mass_times_coord_y = mass_y * torch.arange(h).type(torch.float).to(mass_y.device)

    x = mass_times_coord_x.sum(dim=2, keepdim=True)
    y = mass_times_coord_y.sum(dim=2, keepdim=True)

    if not softmax:
        x = x / mass_x.sum(dim=2, keepdim=True)
        y = y / mass_y.sum(dim=2, keepdim=True)

    coordinates = torch.cat((x, y), dim=2)
    coordinates = coordinates.reshape((batch_size, n_heatmaps, 2))

    return coordinates


def unproject_ij(keypoints_2d, z, camera_matrix):
    """Unprojects points into 3D using intrinsics"""

    z = z.squeeze(2).squeeze(1)
    x = ((keypoints_2d[:, :, 0] - camera_matrix[:, [0], [2]]) / camera_matrix[:, [0], [0]]) * z
    y = ((keypoints_2d[:, :, 1] - camera_matrix[:, [1], [2]]) / camera_matrix[:, [1], [1]]) * z
    xyz = torch.cat((x.unsqueeze(1), y.unsqueeze(1), z.unsqueeze(1)), dim=1)
    return xyz


def reproject_points(pose, pts, intrinsic, Z):
    """Projects 3d points onto 2D image plane"""

    kp_arr = torch.ones((pts.shape[0], pts.shape[1], 3)).to(pts.device)
    kp_arr[:, :, :2] = pts

    K = intrinsic.unsqueeze(1)
    R = pose[:, :, :3, :3]
    T = pose[:, :, :3, 3:]

    kp_arr = kp_arr.unsqueeze(1)
    reproj_val = ((K @ R) @ (torch.inverse(K))) @ kp_arr.transpose(3, 2)

    proj_z = K @ T / Z
    reproj = reproj_val + proj_z
    reproj = reproj / reproj[:, :, 2:, :]

    return reproj[:, :, :2, :]


def patch_for_kp(keypoints, ker_size, out_length, roi_patch):
    """Creates patch for key-point"""

    keypts_array = keypoints.unsqueeze(1)
    n_view = roi_patch.shape[1]
    keypts_array = keypts_array.repeat(1, n_view, 1, 1)

    xc = keypts_array[:, :, :, 0]
    yc = keypts_array[:, :, :, 1]

    h = torch.ones((keypts_array.shape[0], n_view, keypts_array.shape[2])).to(roi_patch.device) * ker_size  # 3 #kernel_size
    w = ker_size * roi_patch[:, :, :, 3] / out_length
    theta = torch.zeros((keypts_array.shape[0], n_view, keypts_array.shape[2])).to(roi_patch.device)

    keypoint_patch = torch.stack((xc, yc, h, w, theta), 3)
    return keypoint_patch


def match_corr(embed_ref, embed_srch):
    """ Matches the two embeddings using the correlation layer. As per usual
    it expects input tensors of the form [B, C, H, W].
    Args:
        embed_ref: (torch.Tensor) The embedding of the reference image, or
            the template of reference (the average of many embeddings for
            example).
        embed_srch: (torch.Tensor) The embedding of the search image.
    Returns:
        match_map: (torch.Tensor) The correlation between
    """

    _, _, k1, k2 = embed_ref.shape
    b, c, h, w = embed_srch.shape

    if k1 == 1 and k2 == 1:
        pad_img = (0, 0)
    else:
        pad_img = (0, 1)
    match_map = torch.nn.functional.conv2d(embed_srch.contiguous().view(1, b * c, h, w), embed_ref, groups=b, padding=pad_img)

    match_map = match_map.permute(1, 0, 2, 3)

    return match_map


def create_transform_matrix(roi_patch):
    """Creates a 3x3 transformation matrix for the patches"""
    transform_matrix = torch.zeros((roi_patch.shape[0], roi_patch.shape[1], roi_patch.shape[2], 3, 3)).to(roi_patch.device)
    transform_matrix[:, :, :, 0, 0] = torch.cos(roi_patch[:, :, :, 4])
    transform_matrix[:, :, :, 0, 1] = -torch.sin(roi_patch[:, :, :, 4])
    transform_matrix[:, :, :, 0, 2] = roi_patch[:, :, :, 0]
    transform_matrix[:, :, :, 1, 0] = torch.sin(roi_patch[:, :, :, 4])
    transform_matrix[:, :, :, 1, 1] = torch.cos(roi_patch[:, :, :, 4])
    transform_matrix[:, :, :, 1, 2] = roi_patch[:, :, :, 1]
    transform_matrix[:, :, :, 2, 2] = 1.0

    return transform_matrix


def patch_sampler(roi_patch, out_length=640, distance=2, do_img=True, align_corners=False):
    """Creates, scales and aligns the patch"""

    ##create a regular grid centered at xc,yc
    if out_length > 1:
        width_sample = torch.linspace(-0.5, 0.5, steps=out_length)
    else:
        width_sample = torch.tensor([0.])

    height_sample = torch.linspace(-distance, distance, steps=2 * distance + 1)
    xv, yv = torch.meshgrid([width_sample, height_sample])
    zv = torch.ones(xv.shape)
    patch_sample = torch.stack((xv, yv, zv), 2).to(roi_patch.device)

    arange_array = patch_sample.repeat(roi_patch.shape[0], roi_patch.shape[1], roi_patch.shape[2], 1, 1, 1)

    ## scaling the x dimension to ensure unform sampling
    arange_array[:, :, :, :, :, 0] = (roi_patch[:, :, :, [3]].unsqueeze(4)) * arange_array[:, :, :, :, :, 0]
    aras = arange_array.shape
    arange_array = arange_array.contiguous().view(aras[0], aras[1], aras[2], aras[3] * aras[4], aras[5]).transpose(4, 3)

    # create matrix transform
    transform_matrix = create_transform_matrix(roi_patch)
    # transform
    patch_kp = transform_matrix @ arange_array

    patch_kp = patch_kp.view(aras[0], aras[1], aras[2], aras[5], aras[3], aras[4])
    patch_kp = patch_kp[:, :, :, :2, :, :].transpose(5, 3)
    return patch_kp, transform_matrix


def patch_for_depth_guided_range(keypoints, pose, intrinsic, img_shape, distance=2, min_depth=0.5, max_depth=10.0, align_corners=False):
    """Represents search patch for a key-point using xc,yc, h,w, theta"""

    # get epilines
    n_view = pose.shape[1]
    pts = keypoints

    kp_arr = torch.ones((pts.shape[0], pts.shape[1], 3)).to(pts.device)
    kp_arr[:, :, :2] = pts
    kp_arr = kp_arr.unsqueeze(1)
    Fund, _ = get_fundamental_matrix(pose, intrinsic, intrinsic)
    lines_epi = (Fund @ (kp_arr.transpose(3, 2))).transpose(3, 2)

    # image shape
    height = img_shape[2]
    width = img_shape[3]

    # default intercepts
    array_zeros = torch.zeros((pts.shape[0], n_view, pts.shape[1])).to(pts.device)
    array_ones = torch.ones((pts.shape[0], n_view, pts.shape[1])).to(pts.device)

    x2ord = array_zeros.clone().detach()
    y2ord = array_zeros.clone().detach()
    x3ord = array_zeros.clone().detach()
    y3ord = array_zeros.clone().detach()

    x0_f = array_zeros.clone().detach()
    y0_f = array_zeros.clone().detach()
    x1_f = array_zeros.clone().detach()
    y1_f = array_zeros.clone().detach()

    ##get x2,x3 and order
    x2_y2 = reproject_points(pose, keypoints, intrinsic, min_depth)
    x2 = x2_y2[:, :, 0, :]
    y2 = x2_y2[:, :, 1, :]
    x3_y3 = reproject_points(pose, keypoints, intrinsic, max_depth)
    x3 = x3_y3[:, :, 0, :]
    y3 = x3_y3[:, :, 1, :]

    x_ord = x3 >= x2
    x2ord[x_ord] = x2[x_ord]
    y2ord[x_ord] = y2[x_ord]
    x3ord[x_ord] = x3[x_ord]
    y3ord[x_ord] = y3[x_ord]

    cx_ord = x2 > x3
    x2ord[cx_ord] = x3[cx_ord]
    y2ord[cx_ord] = y3[cx_ord]
    x3ord[cx_ord] = x2[cx_ord]
    y3ord[cx_ord] = y2[cx_ord]

    if align_corners:
        x_ord0 = (x2ord >= 0) & (x2ord < width)
        x_ord1 = (x3ord >= 0) & (x3ord < width)

        y_ord0 = (y2ord >= 0) & (y2ord < height)
        y_ord1 = (y3ord >= 0) & (y3ord < height)
    else:
        x_ord0 = (x2ord >= -0.5) & (x2ord < (width - 0.5))
        x_ord1 = (x3ord >= -0.5) & (x3ord < (width - 0.5))

        y_ord0 = (y2ord >= -0.5) & (y2ord < (height - 0.5))
        y_ord1 = (y3ord >= -0.5) & (y3ord < (height - 0.5))

    all_range = x_ord0 & x_ord1 & y_ord0 & y_ord1

    x0_f[all_range] = x2ord[all_range]
    y0_f[all_range] = y2ord[all_range]

    x1_f[all_range] = x3ord[all_range]
    y1_f[all_range] = y3ord[all_range]

    cond_null = ~all_range
    x0_f[cond_null] = array_zeros.clone().detach()[cond_null]
    y0_f[cond_null] = array_zeros.clone().detach()[cond_null]
    x1_f[cond_null] = array_zeros.clone().detach()[cond_null]
    y1_f[cond_null] = array_zeros.clone().detach()[cond_null]

    ## find box representation using #xc,yc, h,w, theta
    xc = (x0_f + x1_f) / 2.
    yc = (y0_f + y1_f) / 2.
    h = torch.ones((pts.shape[0], n_view, pts.shape[1])).to(pts.device) * max(2 * distance, 1)
    w = torch.sqrt((x1_f - x0_f) ** 2 + (y1_f - y0_f) ** 2)

    theta = torch.atan2(-lines_epi[:, :, :, 0], lines_epi[:, :, :, 1])

    if torch.sum(torch.isnan(theta)):
        import pdb;
        pdb.set_trace()
    roi_patch = torch.stack((xc, yc, h, w, theta), 3)

    return roi_patch


def sample_descriptors_epi(keypoints, descriptors, s, normalize=True, align_corner=False):
    """Samples descriptors at point locations"""

    b, c, h, w = descriptors.shape

    keypoints = keypoints - s / 2 + 0.5
    keypoints /= torch.tensor([(w * s - s / 2 - 0.5), (h * s - s / 2 - 0.5)], device=keypoints.device)[None]

    keypoints = keypoints * 2 - 1
    if len(keypoints.shape) == 4:
        descriptors = torch.nn.functional.grid_sample(descriptors, keypoints.view(b, keypoints.shape[1], keypoints.shape[2], 2), mode='bilinear',
                                                      align_corners=align_corner)  ##pythorch 1.3+
    elif len(keypoints.shape) == 3:
        descriptors = torch.nn.functional.grid_sample(descriptors, keypoints.view(b, 1, -1, 2), mode='bilinear', align_corners=align_corner)  ##pythorch 1.3+

    if normalize:
        descriptors = torch.nn.functional.normalize(descriptors, p=2, dim=1)

    return descriptors


def vec_to_skew_symmetric(v):
    """Creates skew-symmetric matrix"""
    zero = torch.zeros_like(v[:, 0])
    M = torch.stack([
        zero, -v[:, 2], v[:, 1],
        v[:, 2], zero, -v[:, 0],
        -v[:, 1], v[:, 0], zero,
    ], dim=1)
    return M.reshape(-1, 3, 3)


def get_fundamental_matrix(T_10, K0, K1):
    """Generates fundamental matrix"""

    ##Expects BX3x3 matrix 
    k0 = torch.inverse(K0)
    k1 = torch.inverse(K1).transpose(1, 2)

    k0 = k0.unsqueeze(1)
    k1 = k1.unsqueeze(1)

    T_10 = T_10.view(-1, 4, 4)
    t_skew = vec_to_skew_symmetric(T_10[:, :3, 3])
    E = t_skew @ T_10[:, :3, :3]  ##Essential matrix
    E = E.view(k0.shape[0], -1, 3, 3)

    Fu = (k1 @ E) @ k0  ##Fundamental matrix
    F_norm = Fu[:, :, 2:, 2:]
    F_norm[F_norm == 0.] = 1.
    Fu = Fu / F_norm  ##normalize it
    return Fu, E


class TriangulationNet(BaseModel):
    """Triangulation module"""
    default_config = {

        'depth_range': True,
        'arg_max_weight': 1.0,

        'dist_ortogonal': 1,
        'kernel_size': 1,
        'out_length': 100,
        'has_confidence': True,

        'min_depth': 0.5,
        'max_depth': 10.0,
        'align_corners': False,

    }

    def _init(self):

        self.relu = torch.nn.ReLU(inplace=False)
        self.bn_match_convD = torch.nn.BatchNorm2d(1)

        ##confidence layers
        pool_shape = (self.config['out_length'], 1 + (5 - self.config['kernel_size']))
        pad_shape = (0, 1) if self.config['dist_ortogonal'] == 2 else (1, 1)

        if self.config['has_confidence']:
            self.convD_confa = torch.nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=pad_shape)
            self.bnconvD_confa = torch.nn.BatchNorm2d(1)
            self.pool_convD_conf = torch.nn.MaxPool2d(pool_shape, stride=self.config['out_length'], return_indices=False)

    def _forward(self, data):

        pose = data['pose']
        intrinsic = data['intrinsics']
        img_shape = data['img_shape']
        desc = data['descriptors']
        desc_views = data['descriptors_views']
        sequence_length = data['sequence_length']
        keypoints = data['keypoints']
        depth_all = data['depth']
        depth_ref = data['ref_depths']

        del data

        st = img_shape[2] // desc.shape[2]
        dist = self.config['dist_ortogonal']
        ker_size = self.config['kernel_size']
        out_length = self.config['out_length']

        pred = {}
        pred['keypoints'] = keypoints

        ## Creates patches for matching
        depth_at_kp = sample_descriptors_epi(keypoints, depth_all.unsqueeze(1), 1, False, self.config['align_corners'])
        roi_patch = patch_for_depth_guided_range(keypoints, pose, intrinsic, img_shape, distance=dist, min_depth=self.config['min_depth'],
                                                 max_depth=self.config['max_depth'], align_corners=self.config['align_corners'])
        keypoint_patch = patch_for_kp(keypoints, ker_size, out_length, roi_patch)

        ## Extract sampled keypoints 
        kp_image, transform_matrix = patch_sampler(roi_patch, out_length=out_length, distance=dist, do_img=True, align_corners=self.config['align_corners'])
        kp_anchor, _ = patch_sampler(keypoint_patch, out_length=ker_size, distance=ker_size // 2, do_img=False, align_corners=self.config['align_corners'])

        ## Reshape along batch dimenstion
        kp_image_shp = kp_image.shape
        kp_image = kp_image.contiguous().view(kp_image_shp[0] * kp_image_shp[1], kp_image_shp[2], kp_image_shp[3] * kp_image_shp[4], kp_image_shp[5])
        kp_anchor_shp = kp_anchor.shape
        kp_anchor = kp_anchor.contiguous().view(kp_anchor_shp[0] * kp_anchor_shp[1], kp_image_shp[2], kp_anchor_shp[3] * kp_anchor_shp[4], kp_anchor_shp[5])

        ## Sample
        desc_views_shp = desc_views.shape
        desc_views = desc_views.reshape(desc_views_shp[0] * desc_views_shp[1], desc_views_shp[2], desc_views_shp[3], desc_views_shp[4])
        descriptor_at_image = sample_descriptors_epi(kp_image.detach(), desc_views, st, True, self.config['align_corners'])
        descriptor_at_anchor = sample_descriptors_epi(kp_anchor.detach(), desc.repeat_interleave(sequence_length, dim=0), st, True,
                                                      self.config['align_corners'])

        del kp_image, kp_anchor, keypoint_patch, desc, desc_views

        descriptor_at_anchor = descriptor_at_anchor.contiguous().view(descriptor_at_anchor.shape[0], descriptor_at_anchor.shape[1], kp_anchor_shp[2],
                                                                      kp_anchor_shp[3], kp_anchor_shp[4])
        descriptor_at_image = descriptor_at_image.contiguous().view(descriptor_at_image.shape[0], descriptor_at_image.shape[1], kp_image_shp[2],
                                                                    kp_image_shp[3], kp_image_shp[4])

        descriptor_at_anchor = descriptor_at_anchor.transpose(2, 1)
        descriptor_at_image = descriptor_at_image.transpose(2, 1)

        dancs = descriptor_at_anchor.shape
        dimgs = descriptor_at_image.shape

        descriptor_at_anchor = descriptor_at_anchor.contiguous().view(dancs[0] * dancs[1], dancs[2], dancs[3], dancs[4])
        descriptor_at_image = descriptor_at_image.contiguous().view(dimgs[0] * dimgs[1], dimgs[2], dimgs[3], dimgs[4])

        ## Do cross correlation
        match_map = match_corr(descriptor_at_anchor, descriptor_at_image)
        match_map = self.bn_match_convD(match_map)
        match_map = self.relu(match_map)

        del descriptor_at_anchor, descriptor_at_image

        if self.config['has_confidence']:
            conf_da = match_map
            conf_da = torch.nn.functional.adaptive_max_pool2d(conf_da, (1, 1))
            conf_da = conf_da.contiguous().view(kp_image_shp[0], kp_image_shp[1], -1)

            sc_factor = 1.0
            conf_da = torch.sigmoid(sc_factor * conf_da)
            conf_damp = roi_patch[:, :, :, 3] > 0.
            conf_da = conf_da * (conf_damp.float() + 0.001)

            self_confidence = torch.ones((conf_da.shape[0], 1, conf_da.shape[2])).to(conf_da.device)
            conf_da = torch.cat((self_confidence, conf_da), 1)
            conf_da = conf_da.transpose(2, 1)
            pred['confidence'] = conf_da
        else:
            pred['confidence'] = None

        ## SOFTARGMAX
        out_kp_match = integrate_tensor_2d(match_map * self.config['arg_max_weight'], True)

        ## Change from local coordinates to image coordinates
        out_kp_match /= torch.tensor([match_map.shape[3] - 1., max(match_map.shape[2] - 1., 1.)], device=out_kp_match.device)[None]

        if match_map.shape[2] == 1:
            sub_roi = (torch.tensor([0.5, 0.]).unsqueeze(0).unsqueeze(1)).to(out_kp_match.device)
        else:
            sub_roi = 0.5

        out_kp_match -= sub_roi
        out_ones = torch.ones((out_kp_match.shape[0], 1, 1)).to(out_kp_match.device)
        out_kp_match = torch.cat((out_kp_match, out_ones), 2)
        out_kp_match = out_kp_match.view(kp_image_shp[0], kp_image_shp[1], kp_image_shp[2], 3)

        ## scale the local x coordinate to match sampling frequency
        mult_0 = roi_patch[:, :, :, [3]]
        mult_1 = torch.ones_like(mult_0)
        mult_1[mult_0 == 0.] = 0.
        roi_mult = torch.cat((mult_0, mult_1, mult_1), 3)
        out_kp_match *= roi_mult

        range_kp = roi_patch[:, :, :, 3] > 0.
        pred['range_kp'] = range_kp

        ##global coordinates
        val_kp_match = ((transform_matrix @ out_kp_match.unsqueeze(4))[:, :, :, :2, :]).squeeze(4)
        pred['multiview_matches'] = val_kp_match

        del out_kp_match, transform_matrix, match_map

        ## 3d GT
        keypoints_3d_gt = unproject_ij(keypoints, depth_at_kp, intrinsic)
        pred['keypoints3d_gt'] = keypoints_3d_gt.transpose(2, 1)

        ####  Triangulation
        pose_tiled = pose[:, :, :3, :]
        intrinsic_tiled = intrinsic
        confidence = pred['confidence']

        anchor_keypoints = keypoints.unsqueeze(1)
        multiview_matches = torch.cat((anchor_keypoints, val_kp_match), 1)

        projection_mat = []
        projection_ref = []
        proj_identity = torch.tensor([[1., 0., 0., 0.], [0., 1., 0., 0.], [0., 0., 1., 0.]])
        if torch.cuda.is_available():
            proj_identity = proj_identity.cuda()

        for batch_idx in range(pose_tiled.size(0)):
            proj_ref_idx = torch.mm(intrinsic_tiled[batch_idx], proj_identity).unsqueeze(0)
            projection_ref.append(proj_ref_idx)

            projection_mat_view = []
            for j in range(sequence_length):
                proj_mat_idx = torch.mm(intrinsic_tiled[batch_idx], pose_tiled[batch_idx][j]).unsqueeze(0)
                projection_mat_view.append(proj_mat_idx)

            projection_mat_view = torch.cat(projection_mat_view, 0).unsqueeze(0)
            projection_mat.append(projection_mat_view)

        projection_mat = torch.cat(projection_mat, 0)
        projection_ref = torch.cat(projection_ref, 0).unsqueeze(1)

        proj_matrices = torch.cat([projection_ref, projection_mat], 1)

        del projection_ref, projection_mat

        if self.config['has_confidence']:
            keypoints_3d = triangulate_batch_of_points(proj_matrices, multiview_matches, confidence)
        else:
            keypoints_3d = triangulate_batch_of_points(proj_matrices, multiview_matches)

        keypoints_3d = torch.stack(keypoints_3d, 0)
        if torch.sum(torch.isinf(keypoints_3d)) > 0:
            keypoints_3d = torch.clamp(keypoints_3d, min=-1000.0, max=1000.0)

        pred['keypoints_3d'] = keypoints_3d

        return pred

    def loss(self, pred, data):
        raise NotImplementedError

    def metrics(self):
        raise NotImplementedError
