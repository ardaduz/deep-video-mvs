import torch


def reorder_desc(desc, batch_sz):
    """Reorders Descriptors"""

    b, c, h, w = desc.shape
    desc = desc.view(-1, batch_sz, c, h, w)
    desc = desc.transpose(1, 0)
    return desc


def pose_square(pose):
    """Converts pose matrix of size 3x4 to a square matrix of size 4x4"""

    pose_sh = pose.shape
    if pose_sh[2] == 3:
        pose_row = torch.tensor([0., 0., 0., 1.])
        if pose.is_cuda:
            pose_row = pose_row.to(pose.device)
        pose_row = pose_row.repeat(pose_sh[0], pose_sh[1], 1, 1)
        pose = torch.cat((pose, pose_row), 2)

    return pose


def make_symmetric(anc, ref):
    """Makes anchor and reference tensors symmetric"""

    if (anc is None) or (ref is None):
        return None
    ancs = anc.shape
    views = torch.stack(ref, 0)
    if len(ancs) == 3:
        views = views.view(-1, ancs[1], ancs[2])
    else:
        views = views.view(-1, anc.shape[1], ancs[2], ancs[3])
    anc_ref = torch.cat((anc, views), 0)
    return anc_ref
