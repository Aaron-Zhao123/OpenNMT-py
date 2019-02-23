import numpy as np
import torch
import pickle

from torch.nn import Parameter
from sklearn.mixture import GaussianMixture


class Quantizer(object):
    # the quantizer is our customized recentralization quantizer
    masks = {}
    _variables = ['weight', 'bias']

    def _check_name(self, name):
        for v_partial in self._variables:
            if v_partial in name:
                return True
        return False

    def __init__(self, load_meta=None, device='cpu', save_meta='quantize.pkl', quantize_params={'width': 4, 'distance': 2.0}):
        if load_mask is not None:
            self._load_meta(load_mask)
        self.prune_params = prune_params
        self.save_mask = save_mask
        self.device = device
        super(Pruner).__init__()

    def forward_pass_quantizer(self, name, value):
        if self._check_name(name):
            nonzeros = value != 0
            mode, width, bias = self.quan[name]
            pmean, pstd, nmean, nstd = self.stats[name]
            pmask, nmask, zmask, nzmask = self.masks[name]

            if mode == 'seperate':
                pvalue = pmask * ((value - pmean) / (pstd + self.epsilon))
                nvalue = nmask * ((value - nmean) / (nstd + self.epsilon))
                quantized = zmask * self.shift_quantize(pvalue-nvalue, width=width, bias=bias)
                pmean_quantized = self.shift_quantize(pmean, width, bias, bypass_clip=True)
                nmean_quantized = self.shift_quantize(nmean, width, bias, bypass_clip=True)
                quantized = (pmean_quantized + quantized) * pmask + (nmean_quantized + quantized) * nmask
            else:
                quantized = nzmask * self.shift_quantize(value, width=width, bias=bias)
        return quantized

    def shift_quantize(self, value, width, bias, bypass_clip=False):
        descriminator = (2.0 ** (-bias)) / 2.0
        sign = value > descriminator
        sign -= (value < -descriminator)

        value = torch.abs(value)
        exponent = torch.round(torch.log(value,2))

        exponent_min = -bias
        exponent_max = 2 ** width - 1 - bias
        if not bypass_clip:
            exponent = torch.clip_by_value(exponent, exponent_min, exponent_max)

        return sign * 2.0 ** (exponent)

    def update_quantizers(self, named_params):
        for n, p in named_params:
            if self._check_name(n):
                mode, width, pmean, nmean, pstd, nstd, pmask, nmask, zmask = self.mle_update(p)
                nzmask = np.logical_not(nmask)
                if mode == 'seperate':
                    value = value * pmask - pmean + value * nmask - nmean
                    value *= nzmask
                bias = self._shift_update(value, width)
                self.stats[name] = (pmean, pstd, nmean, nstd)
                self.masks[name] = (pmask, nmask, zmask, nzmask)
                self.quan[name] = (mode, width, bias)


    def _shift_update(self, value, width):
        max_exponent = np.ceil(np.log2(np.max(np.abs(value))))
        return 2 ** width - 1 - max_exponent

    def mle_update(self, name, value):
        value = value.detach().numpy()
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
            self.masks,  = pickle.load(f)
        print("Loaded mask from {}".format(fname))

    def _save_masks(self, fname):
        with open(fname, 'wb') as f:
            pickle.dump(self.masks, f)


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


