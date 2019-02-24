import numpy as np
import torch
import pickle
import math

from torch.nn import Parameter
from sklearn.mixture import GaussianMixture


class CustomRound(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return torch.round(x)

    @staticmethod
    def backward(ctx, g):
        return g


class Quantizer(object):
    # the quantizer is our customized recentralization quantizer
    masks = {}
    _variables = ['weight']

    def _check_name(self, name):
        for v_partial in self._variables:
            if v_partial in name:
                return True
        return False

    def __init__(
            self, load_meta=None, device='cpu', save_meta='quantize.pkl', epsolon=1e-4,
            quantize_params={'width': 4, 'distance': 2.5, 'percentage': 0.25}):
        if load_meta is not None:
            self._load_meta(load_mask)
        self.quantize = quantize_params
        self.save_meta = save_meta
        self.device = device
        self.wsep = quantize_params['distance']
        self.width = quantize_params['width']
        self.percentage = quantize_params['percentage']
        self.gmm_stats = {}
        self.masks = {}
        self.quan = {}
        self.epsilon = epsolon
        self.trainable_scale = Parameter(torch.Tensor(1.0), requires_grad=True)
        super(Quantizer).__init__()

    def forward_pass(self, name, value):
        if self._check_name(name):
            # value = self.shift_quantize(value, self.width, bias=10)
            # return value
            # # return value
            nonzeros = value != 0
            mode, width, bias = self.quan[name]
            pmean, pstd, nmean, nstd = self.gmm_stats[name]
            pmask, nmask, zmask, nzmask = self.masks[name]

            if mode == 'seperate':
                pmask = torch.Tensor(pmask.astype(np.int))
                nmask = torch.Tensor(nmask.astype(np.int))
                nzmask = torch.Tensor(nzmask.astype(np.int))

                pvalue = pmask * (value - pmean)
                nvalue = nmask * (value - nmean)
                residual_quantized = nzmask * self.shift_quantize(pvalue+nvalue, width=width, bias=bias)
                quantized = (pmean + residual_quantized) * pmask + (nmean + residual_quantized) * nmask
            else:
                # nzmask = torch.Tensor(nzmask.astype(np.int))
                # quantized = nzmask * self.shift_quantize(value, width=width, bias=bias)
                quantized = self.shift_quantize(value, width=width, bias=bias)
            quantized *= self.trainable_scale
            quantized = quantized * self.interval_mask + value * (1 - self.interval_mask)
            return quantized
        return value

    def shift_quantize(self, value, width, bias, bypass_clip=False):
        descriminator = (2.0 ** (-bias)) / 2.0
        orig_value = value
        if isinstance(value, torch.Tensor):
            sign = (orig_value > descriminator).float()
            sign -= (orig_value < -descriminator).float()
            abs_orig_value = torch.abs(orig_value)
            exponent = torch.log2(abs_orig_value)
            exponent = torch.round(exponent)
        else:
            sign = (orig_value > descriminator).astype(np.float)
            sign -= (orig_value < -descriminator).astype(np.float)
            exponent = round(math.log(math.fabs(orig_value), 2))
            return sign * (2.0 ** exponent)

        exponent_min = -bias
        exponent_max = 2 ** width - 1 - bias

        if not bypass_clip:
            exponent = torch.clamp(exponent, exponent_min, exponent_max)
        qvalue = sign.float() * 2.0 ** (exponent)

        # bypass
        with torch.no_grad():
            qerror = orig_value - qvalue
        return value + qerror

    def update_quantizers(self, named_params, pruner_masks=None):
        for n, p in named_params:
            if self._check_name(n):
                if p.is_cuda:
                    p = p.cpu()
                p = p.detach().numpy()
                mask = pruner_masks[n+'.mask'].detach().numpy().astype(np.int)
                mode, width, pmean, nmean, pstd, nstd, pmask, nmask, zmask = self.mle_update(n, p, mask)
                nzmask = np.logical_not(nmask)
                value = p
                if mode == 'seperate':
                    pmean = self.shift_quantize(pmean, 2*width, width, bypass_clip=True)
                    nmean = self.shift_quantize(nmean, 2*width, width, bypass_clip=True)
                    value = value * pmask - pmean + value * nmask - nmean
                    value *= nzmask
                threshold = np.percentile(np.abs(value[nzmask]), self.percentage * 100)
                self.interval_mask = np.abs(value) > threshold
                self.interval_mask = torch.Tensor(self.interval_mask.astype(np.int))

                bias = self._shift_update(value, width)
                self.gmm_stats[n] = (pmean, pstd, nmean, nstd)
                self.masks[n] = (pmask, nmask, zmask, nzmask)
                self.quan[n] = (mode, width, bias)
        self._save_meta(self.save_meta)


    def _shift_update(self, value, width):
        max_exponent = np.ceil(np.log2(np.max(np.abs(value))))
        return 2 ** width - max_exponent

    def mle_update(self, name, value, nzmask=None):
        if nzmask is None:
            nzmask = value != 0

        mixture = BimodalGaussian(name, value[nzmask], 0.0)
        pmean, nmean = mixture.pmean, mixture.nmean
        pstd, nstd = mixture.pstd, mixture.nstd
        phi = mixture.phi
        print(
            'Found bimodal Gaussian mixture with '
            '{0} * N({0}, {0}) + {0} * N({0}, {0}).'.format('{:.3g}')
            .format(phi, pmean, pstd, 1 - phi, nmean, nstd))

        if self.wsep is not None:
            dist = mixture.wasserstein
            if dist > self.wsep:
                mode = 'seperate'
                width = self.width
            else:
                mode = 'uniform'
                width = self.width + 1
            print(
                'Wasserstein metric {}, threshold {}, we {}.'
                .format(dist, self.wsep, mode))
        else:
            width = self.width
            mode = 'seperate'
        pmask = mixture.predict1(value, True)
        pmask = np.logical_and(pmask, nzmask)
        nmask = np.logical_and(np.logical_not(pmask), nzmask)
        zmask = np.logical_not(nzmask)
        return (mode, width, pmean, nmean, pstd, nstd, pmask, nmask, zmask)

    def get_mask(self, value, name):
        mask = self.masks.get(name+'.mask')
        if mask is None:
            mask = Parameter(torch.ones(value.shape), requires_grad=False)
            self.masks[name + '.mask'] = value.to(self.device)
        return mask

    def _load_meta(self, fname):
        with open(fname, 'rb') as f:
            self.gmm_stats, self.masks, self.quan  = pickle.load(f)
        print("Loaded mask from {}".format(fname))

    def _save_meta(self, fname):
        with open(fname, 'wb') as f:
            pickle.dump((self.gmm_stats, self.masks, self.quan), f)


class BimodalGaussian(GaussianMixture):
    # copied from mayo
    def __init__(self, name, data, overflow_rate=0.0):
        data, bound = self._overflow(data, overflow_rate)
        pmean, _1, nmean, _2, phi = self._find_initial(data)
        means = np.array([[pmean], [nmean]])
        weights = np.array([phi, 1 - phi])
        super().__init__(2, means_init=means, weights_init=weights)
        self.name = name
        self.bound = bound
        self.data = data
        self._fit()

    @staticmethod
    def _overflow(data, orate):
        abs_data = np.abs(data)
        if orate <= 0:
            return data, np.max(abs_data)
        magnitudes = np.sort(abs_data)
        index = int((1 - orate) * data.size)
        max_value = magnitudes[min(max(0, index), data.size - 1)]
        return data[abs_data < max_value], max_value
        # return np.where(abs_data < max_value, data, np.sign(data) * max_value)

    @staticmethod
    def _find_initial(x):
        p = x > 0
        pv = x[np.where(p)]
        pmean, pstd = np.mean(pv), np.std(pv)
        n = np.logical_and(np.logical_not(p), x != 0)
        nv = x[np.where(n)]
        nmean, nstd = np.mean(nv), np.std(nv)
        phi = np.size(pv) / np.size(x)
        return pmean, pstd, nmean, nstd, phi

    def _fit(self):
        self.tmean = np.mean(self.data)
        self.tstd = np.sqrt(np.var(self.data))
        self.fit(self.data.reshape(-1, 1))
        if not self.converged_:
            raise ValueError('Unable to find MLE for weight distribution.')
        means, vars, weights = self.means_, self.covariances_, self.weights_
        mean1, mean2 = means[:, 0]
        std1, std2 = np.sqrt(vars[:, 0, 0])
        phi = weights[0]
        if mean1 < mean2:
            mean1, mean2 = mean2, mean1
            std1, std2 = std2, std1
            phi = 1 - phi
            log.debug('Components flipped.')
            # raise ValueError('Components flipped.')
        self.pmean, self.nmean = mean1, mean2
        self.pstd, self.nstd = std1, std2
        self.phi = phi

    def pdf1(self, x):
        pdf1 = super().predict_proba(x.reshape(-1, 1))
        return pdf1[:, 0].reshape(x.shape)

    def predict1(self, x, sampling=False):
        if sampling:
            return np.random.uniform(size=x.shape) < self.pdf1(x)
        return self.pdf1(x) >= 0.5

    def pdf(self, x):
        return np.exp(self.score_samples(x.reshape(-1, 1)))

    def plot(self):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from scipy.stats import norm
        data = self.data
        plt.hist(data.flatten(), bins=500, density=True, color='c', alpha=0.5)
        q = np.linspace(np.min(data), np.max(data), 10000)
        plt.plot(q, self.pdf(q), '-', label='pdf')
        p1 = norm.pdf(q, self.pmean, self.pstd)
        p2 = norm.pdf(q, self.nmean, self.nstd)
        plt.plot(q, self.phi * p1, '--', label='in1')
        plt.plot(q, (1 - self.phi) * p2, '--', label='in2')
        plt.savefig(f"plots/{self.name.replace('/', '_')}.pdf")
        plt.gcf().clear()

    @property
    def wasserstein(self):
        # scale and shift mixture distribution for 2-wasserstein criteria
        pmean = (self.pmean - self.tmean) / self.tstd
        nmean = (self.nmean - self.tmean) / self.tstd
        pstd, nstd = self.pstd / self.tstd, self.nstd / self.tstd
        # 2-wasserstein distance
        # FIXME only works well if self.phi is close to 0.5
        wmean = (pmean - nmean) ** 2
        wvar = pstd ** 2 + nstd ** 2 - 2 * pstd * nstd
        return wmean + wvar

    @property
    def crossover(self):
        # computes the probability of cross-over
        from scipy.stats import norm
        resolution = 1000
        x = np.linspace(self.nmean, self.pmean, resolution)
        i = np.argmin(np.abs(self.pdf1(x) - 0.5))
        xco = x[i]
        nco = norm.cdf(xco, self.nmean, self.nstd)
        pco = norm.cdf(xco, self.pmean, self.pstd)
        return self.phi * pco + (1 - self.phi) * (1 - nco)


