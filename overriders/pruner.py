import numpy as np
import torch
import pickle
from torch.nn import Parameter


class Pruner(object):
    masks = {}

    _variables = ['rnn.weight', 'rnn.layer', 'embedding', 'attn']
    _exclude_variables = ['dummy']


    def _check_name(self, name):
        for v_partial in self._variables:
            for nv_partial in self._exclude_variables:
                if v_partial in name and not nv_partial in name:
                    return True
        return False

    def __init__(self, load_mask=None, device='cpu', save_mask='mask.pkl', prune_params={'alpha':0.5}):
        if load_mask is not None:
            self._load_masks(load_mask)
        self.prune_params = prune_params
        self.save_mask = save_mask
        self.device = device
        super(Pruner).__init__()

    def get_mask(self, value, name):
        mask = self.masks.get(name+'.mask')
        if mask is None:
            mask = Parameter(torch.ones(value.shape), requires_grad=False)
            self.masks[name + '.mask'] = value.to(self.device)
        return mask

    def update_masks(self, named_params):
        self.sparsities = []
        for n, p in named_params:
            if self._check_name(n):
                existing_mask = self.get_mask(p, n)
                mask = self._update_mask(p, existing_mask, params=self.prune_params)
                self.masks[n + '.mask'] = mask.to(self.device)
                self.sparsities.append((n, int(torch.sum(mask)), mask.numel()))
        self._save_masks(fname=self.save_mask)
        print("Updated and saved masks.")
        # print(self.sparsities)
        self._total_density()

    def _total_density(self):
        ones = sum([xs[1] for xs in self.sparsities])
        total = sum([xs[2] for xs in self.sparsities])
        print("Total density {}/{} = {}".format(ones, total, ones/total))


    def _update_mask(self, value, existing_mask, params):
        # dns style update
        alpha = params.get('alpha')
        if value.is_cuda:
            value = value.cpu()
        value = value.detach().numpy()
        if existing_mask.is_cuda:
            existing_mask = existing_mask.cpu()
        existing_mask = existing_mask.detach().numpy()
        # threshold = np.percentile(value, alpha*100)
        # mask = torch.Tensor((value > threshold).astype(np.float))
        mean = np.mean(value)
        std = np.std(value)
        threshold = mean + alpha * std
        on_mask = np.abs(value) > (1.1 * threshold)
        off_mask = np.abs(value) > (0.9 * threshold)
        new_mask = np.logical_or(existing_mask, on_mask)
        new_mask = np.logical_and(new_mask, off_mask)
        mask = torch.Tensor((new_mask).astype(np.float))
        mask = Parameter(mask, requires_grad=False)
        return mask

    def _load_masks(self, fname):
        with open(fname, 'rb') as f:
            self.masks = pickle.load(f)
        print("Loaded mask from {}".format(fname))

    def _save_masks(self, fname):
        with open(fname, 'wb') as f:
            pickle.dump(self.masks, f)

