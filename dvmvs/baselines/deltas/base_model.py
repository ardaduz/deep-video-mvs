import collections
from abc import ABCMeta, abstractmethod

from torch import nn


def dict_update(d, u):
    """Improved update for nested dictionaries.

    Arguments:
        d: The dictionary to be updated.
        u: The update dictionary.

    Returns:
        The updated dictionary.
    """
    d = d.copy()
    for k, v in u.items():
        if isinstance(v, collections.Mapping):
            d[k] = dict_update(d.get(k, {}), v)
        else:
            d[k] = v
    return d


class BaseModel(nn.Module, metaclass=ABCMeta):
    """Base Model"""

    base_config = {
        'name': None,
        'trainable': True,
    }
    default_config = {}
    required_data_keys = []

    def __init__(self, config):
        nn.Module.__init__(self)

        default_config = dict_update(self.base_config, self.default_config)
        new_keys = set(config.keys()) - set(default_config.keys())
        if len(new_keys) > 0:
            raise ValueError(
                'Detected new keys in config: {}'.format(new_keys))
        self.config = dict_update(default_config, config)
        self._init()
        if not self.config['trainable']:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, data, **kwarg):
        for key in self.required_data_keys:
            assert key in data, 'Missing key {} in data'.format(key)
        return self._forward(data, **kwarg)

    @abstractmethod
    def _init(self):
        raise NotImplementedError

    @abstractmethod
    def _forward(self, data):
        raise NotImplementedError

    @abstractmethod
    def loss(self, pred, data):
        raise NotImplementedError

    @abstractmethod
    def metrics(self):
        raise NotImplementedError
