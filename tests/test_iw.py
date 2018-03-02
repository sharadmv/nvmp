from base import BaseTest
from deepx import T
import numpy as np
from scipy.stats import invwishart
from scipy.special import multigammaln, digamma

from nvmp.stats import IW
from nvmp.core import Stats

class TestInverseWishart(BaseTest):

    def setup(self):
        super(TestInverseWishart, self).setup()
        np.random.seed(0)

    def log_z(self, S, nu):
        d = S.shape[-1]
        return (
            -0.5 * nu * np.linalg.slogdet(S)[1]
            + 0.5 * nu * d * np.log(2.)
            + multigammaln(0.5 * nu, d)
        )

    def stats(self, S, nu):
        d = S.shape[-1]
        nu = np.array(nu)
        return {
            Stats.HSI: -0.5 * np.linalg.inv(S) * nu,
            Stats.HLDS: 0.5 * (np.sum(digamma((nu[...,None] - np.arange(d)[None,...])/2.)[0], -1)
                                          + d*np.log(2.)) - np.linalg.slogdet(S)[1]
        }

    def natparam(self, S, nu):
        J = np.linalg.inv(sigma)
        d = S.shape[-1]
        return {
            Stats.HSI: S,
            Stats.HLDS: nu + d + 1.
        }

    def test_log_likelihood1(self):
        d = 2
        data = np.tile(np.eye(d)[None], [10, 1, 1])
        sigma = IW([T.eye(d), d + 1])
        np.testing.assert_almost_equal(
            self.session.run(sigma.log_likelihood(T.to_float(data))),
            invwishart(scale=np.eye(2), df=d + 1).logpdf(data.T),
            5
        )

    def test_log_likelihood2(self):
        d = 100
        data = np.tile(np.eye(d)[None], [10, 1, 1])
        S = np.diag(np.exp(np.random.normal(size=d)))
        sigma = IW([T.to_float(S), d + 1])
        np.testing.assert_almost_equal(
            self.session.run(sigma.log_likelihood(T.to_float(data))),
            invwishart(scale=S, df=d + 1).logpdf(data.T),
            3
        )

    def test_log_likelihood2(self):
        d = 100
        data = invwishart(scale=np.eye(d), df=d+1).rvs(size=100)
        S = np.eye(d)
        sigma = IW([T.to_float(S), d + 1])
        np.testing.assert_almost_equal(
            self.session.run(sigma.log_likelihood(T.to_float(data))),
            invwishart(scale=S, df=d + 1).logpdf(data.T),
            -1
        )

    def test_log_z(self):
        d = 100
        data = invwishart(scale=np.eye(d), df=d+1).rvs(size=100)
        S = np.eye(d)
        sigma = IW([T.to_float(S), d + 1])
        np.testing.assert_almost_equal(
            self.session.run(sigma.log_z()),
            self.log_z(S, d + 1),
            3
        )
        np.testing.assert_almost_equal(
            self.session.run(sigma.log_z('natural')),
            self.log_z(S, d + 1),
            3
        )

    def test_stats1(self):
        d = 2
        S = np.eye(d)
        sigma = IW([T.to_float(S), d + 1])
        stats = self.stats(S, d + 1)
        stats_ = self.session.run(sigma.expected_sufficient_statistics())
        [
            np.testing.assert_almost_equal(
                stats_[s],
                stats[s]
            )
            for s in sigma.statistics()
        ]
