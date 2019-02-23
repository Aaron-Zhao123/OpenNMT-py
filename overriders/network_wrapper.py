import torch.nn as nn
import functools
import torch


# recursive getter
def rgetattr(obj, attr, *args):
    def _getattr(obj, attr):
        return getattr(obj, attr, *args)
    return functools.reduce(_getattr, [obj] + attr.split('.'))


# recursive setter
def rsetattr(obj, attr, val):
    pre, _, post = attr.rpartition('.')
    return setattr(rgetattr(obj, pre) if pre else obj, post, val)


class NetworkWrapperBase(object):

    _variables = ['weight']

    def _check_name(self, name):
        for v_partial in self._variables:
            if v_partial in name:
                return True
        return False

    def _override(self, update=False):
        pruner = self.transformer.get('pruner', None)
        quantizer = self.transformer.get('quantizer', None)

        if update:
          self.pruner.update_masks(self.named_parameters())

        # pruner
        for name, param in self.named_parameters():
            if self._check_name(name):
              mask = self.pruner.get_mask(param.data, name)
              rsetattr(self, name + '.data', param.data.mul_(mask))

# def override(model, transformer, update=False):
#     if update:
#         transformer.update_masks(model.named_parameters())
#     sparsities = []
#     for name, param in model.named_parameters():
#         # if self._check_name(name):
#         mask = transformer.get_mask(param.data, name)
#         # mask = torch.zeros(param.data.shape)
#         param.data = param.data.mul_(mask)
#         # rsetattr(model, name + '.data', param.data.mul_(mask))
#         sparsities.append((name, int(torch.sum(mask)), mask.numel()))
#     print(sparsities)
