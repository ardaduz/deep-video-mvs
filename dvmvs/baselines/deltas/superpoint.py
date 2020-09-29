import torch
import torchvision

from .base_model import BaseModel


def simple_nms(scores, radius):
    """Performs non maximum suppression on the heatmap using max-pooling.
    This method does not suppress contiguous points that have the same score.
    Arguments:
        scores: the score heatmap, with shape `[B, H, W]`.
        size: an interger scalar, the radius of the NMS window.
    """

    def max_pool(x):
        return torch.nn.functional.max_pool2d(
            x, kernel_size=radius * 2 + 1, stride=1, padding=radius)

    zeros = torch.zeros_like(scores)
    max_mask = scores == max_pool(scores)
    for _ in range(2):
        supp_mask = max_pool(max_mask.float()) > 0
        supp_scores = torch.where(supp_mask, zeros, scores)
        new_max_mask = supp_scores == max_pool(supp_scores)
        max_mask = max_mask | (new_max_mask & (~supp_mask))
    return torch.where(max_mask, scores, zeros)


def remove_borders(keypoints, scores, b, h, w):
    mask_h = (keypoints[:, 0] >= b) & (keypoints[:, 0] < (h - b))
    mask_w = (keypoints[:, 1] >= b) & (keypoints[:, 1] < (w - b))
    mask = mask_h & mask_w
    return keypoints[mask], scores[mask]


def top_k_keypoints(keypoints, scores, k):
    if k >= len(keypoints):
        return keypoints, scores
    scores, indices = torch.topk(scores, k, dim=0)
    return keypoints[indices], scores


class Superpoint(BaseModel):
    default_config = {
        'has_detector': True,
        'has_descriptor': True,
        'descriptor_dim': 128,

        # Inference for Anchor
        'sparse_outputs': True,
        'nms_radius': 9,
        'detection_threshold': 0.0005,
        'top_k_keypoints': 128,
        'force_num_keypoints': True,
        'remove_borders': 4,
        'unique_keypoints': True,
        'frac_superpoint': 1.,

        'dense_depth': True,
        'min_depth': 0.5,
        'max_depth': 10.0,

        'model_type': 'resnet50',
        'align_corners': False,
        'height': 240,
        'width': 320,

    }

    def _init(self):

        self.relu = torch.nn.ReLU(inplace=True)
        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2)

        pretrained_features = torchvision.models.resnet50(pretrained=False)
        c_out = [2048, 8, 10]
        c_d = 512
        c_k = 64 + 256

        self.conv1 = pretrained_features.conv1
        self.bn1 = pretrained_features.bn1
        self.maxpool = pretrained_features.maxpool
        self.layer1 = pretrained_features.layer1
        self.layer2 = pretrained_features.layer2
        self.layer3 = pretrained_features.layer3
        self.layer4 = pretrained_features.layer4

        self.rgb_to_gray = torch.tensor([0.299, 0.587, 0.114])
        self.rgb_to_gray = self.rgb_to_gray.view(1, -1, 1, 1)

        self.mean_add_rgb = torch.tensor([0.485, 0.456, 0.406]).view(1, -1, 1, 1)
        self.std_mul_rgb = torch.tensor([0.229, 0.224, 0.225]).view(1, -1, 1, 1)

        self.mean_add = torch.tensor([0.5, 0.5, 0.5]).view(1, -1, 1, 1)
        self.std_mul = torch.tensor([0.5, 0.5, 0.5]).view(1, -1, 1, 1)

        if self.config['has_detector']:
            c_1, c_2 = 256, 128
            self.convPa = torch.nn.Conv2d(c_out[0], c_1, kernel_size=3, stride=1, padding=1)
            self.bnPa = torch.nn.BatchNorm2d(c_1)
            self.scale_factorPa = 4

            self.convPb = torch.nn.Conv2d(c_1, c_2, kernel_size=3, stride=1, padding=1)
            self.bnPb = torch.nn.BatchNorm2d(c_2)
            self.convPc = torch.nn.Conv2d(c_2, 65, kernel_size=1, stride=1, padding=0)

        if self.config['has_descriptor']:
            c_3, c_4 = 128, 256

            self.convDa = torch.nn.Conv2d(c_out[0], c_3, kernel_size=3, stride=1, padding=1)
            self.bnDa = torch.nn.BatchNorm2d(c_3)

            self.convDb = torch.nn.Conv2d(c_3 + c_d, c_4, kernel_size=1, stride=1, padding=0)
            self.bnDb = torch.nn.BatchNorm2d(c_4)

            self.convDc = torch.nn.Conv2d(c_4, c_4, kernel_size=3, stride=1, padding=1)
            self.bnDc = torch.nn.BatchNorm2d(c_4)

            self.convDd = torch.nn.Conv2d(c_4 + c_k, self.config['descriptor_dim'], kernel_size=1, stride=1, padding=0)

    def _forward(self, data):

        img_rgb = data['img']
        tsp = data['process_tsp']

        img_rgb = (img_rgb - self.mean_add_rgb.to(img_rgb.device)) / self.std_mul_rgb.to(img_rgb.device)
        img = img_rgb

        ##Run superpoint
        pred = {}
        pred['img_rgb'] = img_rgb

        x = self.relu(self.bn1(self.conv1(img)))
        if self.config['dense_depth']:
            pred['skip_half'] = x

        x = self.maxpool(x)

        x = self.layer1(x)
        if self.config['dense_depth']:
            pred['skip_quarter'] = x

        x = self.layer2(x)
        if self.config['dense_depth']:
            pred['skip_eight'] = x

        x = self.layer3(x)
        if self.config['dense_depth']:
            pred['skip_sixteenth'] = x

        x = self.layer4(x)

        if self.config['dense_depth']:
            pred['features'] = x

        # Detector Head.
        if self.config['has_detector'] and ('t' in tsp):
            cPa = self.relu(self.bnPa(self.convPa(x)))
            cPa = torch.nn.functional.interpolate(cPa, size=(self.config['height'] // 8, self.config['width'] // 8), mode='bilinear',
                                                  align_corners=self.config['align_corners'])
            cPa = self.relu(self.bnPb(self.convPb(cPa)))
            pred['scores'] = self.convPc(cPa)

        # Descriptor Head.
        if self.config['has_descriptor'] and ('s' in tsp):
            cDa = self.relu(self.bnDa(self.convDa(x)))
            cDa = torch.nn.functional.interpolate(cDa, size=(self.config['height'] // 8, self.config['width'] // 8), mode='bilinear',
                                                  align_corners=self.config['align_corners'])
            cDa = torch.cat((cDa, pred['skip_eight']), 1)
            cDa = self.relu(self.bnDb(self.convDb(cDa)))
            cDa = self.relu(self.bnDc(self.convDc(cDa)))

            skip_4 = torch.nn.functional.interpolate(pred['skip_quarter'], scale_factor=0.5, mode='bilinear', align_corners=self.config['align_corners'])
            skip_2 = torch.nn.functional.interpolate(pred['skip_half'], scale_factor=0.25, mode='bilinear', align_corners=self.config['align_corners'])

            cDa = torch.cat((cDa, skip_4, skip_2), 1)
            desc = self.convDd(cDa)
            desc = torch.nn.functional.normalize(desc, p=2, dim=1)
            pred['descriptors'] = desc

            # Sparse Key-Points

        if self.config['sparse_outputs'] and ('t' in tsp):
            st = 8  # encoder stride

            if self.config['has_detector']:
                scores = torch.nn.functional.softmax(pred['scores'], 1)[:, :-1]
                b, c, h, w = scores.shape
                scores = scores.permute(0, 2, 3, 1).reshape(b, h, w, st, st)
                scores = scores.permute(0, 1, 3, 2, 4).reshape(b, h * st, w * st)
                dense_scores = scores

                if self.config['nms_radius']:
                    scores = simple_nms(scores, self.config['nms_radius'])

                keypoints = [torch.nonzero(s > self.config['detection_threshold'], as_tuple=False) for s in scores]
                scores = [s[tuple(k.t())] for s, k in zip(scores, keypoints)]

                if self.config['remove_borders']:
                    keypoints, scores = list(zip(*[
                        remove_borders(
                            k, s, self.config['remove_borders'], h * st, w * st)
                        for k, s in zip(keypoints, scores)]))

                if self.config['top_k_keypoints']:
                    keypoints, scores = list(zip(*[
                        top_k_keypoints(k, s, int(self.config['frac_superpoint'] * self.config['top_k_keypoints']))
                        for k, s in zip(keypoints, scores)]))

                    if self.config['force_num_keypoints']:
                        new_keypoints, new_scores = [], []
                        for k, sc in zip(keypoints, scores):
                            num = self.config['top_k_keypoints'] - len(k)

                            new_x = torch.randint_like(k.new_empty(num), w * st)
                            new_y = torch.randint_like(k.new_empty(num), h * st)
                            new_k = torch.stack([new_y, new_x], -1)

                            if self.config['unique_keypoints']:
                                curr_k = torch.cat([k, new_k])
                                not_all_unique = True
                                while not_all_unique:
                                    unique_k = torch.unique(curr_k, dim=1)
                                    if unique_k.shape[0] == curr_k.shape[0]:
                                        not_all_unique = False
                                    else:
                                        new_x = torch.randint_like(k.new_empty(num), w * st)
                                        new_y = torch.randint_like(k.new_empty(num), h * st)
                                        new_k = torch.stack([new_y, new_x], -1)
                                        curr_k = torch.cat([k, new_k])

                            new_sc = sc.new_zeros(num)
                            new_keypoints.append(torch.cat([k, new_k], 0))
                            new_scores.append(torch.cat([sc, new_sc], 0))
                        keypoints, scores = new_keypoints, new_scores

                keypoints = [torch.flip(k, [1]).float() for k in keypoints]

                keypoints = torch.stack(keypoints, 0)
                pred['keypoints'] = keypoints
                pred['scores_sparse'] = scores

        return pred

    def loss(self, pred, data):
        raise NotImplementedError

    def metrics(self):
        raise NotImplementedError
