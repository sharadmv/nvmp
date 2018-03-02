from base import BaseTest
from deepx import T
import numpy as np
from scipy.stats import multivariate_normal

from nvmp.stats import Gaussian
from nvmp.core import Stats

class TestGaussian(BaseTest):

    def setup(self):
        super(TestGaussian, self).setup()
        np.random.seed(0)

    def log_z(self, sigma, mu):
        d = mu.shape[-1]
        return (
            0.5 * np.einsum('...a,ab,...b->...', mu, np.linalg.inv(sigma), mu)
            + 0.5 * np.linalg.slogdet(sigma)[1]
            + d / 2. * np.log(2 * np.pi)
        )

    def stats(self, sigma, mu):
        return {
            Stats.X: mu,
            Stats.XXT: sigma + np.einsum('ia,ib->iab', mu, mu)
        }

    def natparam(self, sigma, mu):
        J = np.linalg.inv(sigma)
        return {
            Stats.X: np.einsum('...ab,...b->...a', J, mu),
            Stats.XXT: -0.5 * J
        }

    def test_log_likelihood1(self):
        d = 2
        data = np.ones([20, d])
        X = Gaussian([T.eye(d), T.zeros(d)])
        np.testing.assert_almost_equal(
            self.session.run(X.log_likelihood(T.to_float(data))),
            multivariate_normal(mean=np.zeros(d), cov=np.eye(d)).logpdf(data),
            5
        )

    def test_log_likelihood2(self):
        d = 100
        data = np.random.normal(size=[100, d])
        mu = np.random.normal(size=d)
        X = Gaussian([T.eye(d), T.to_float(mu)])
        np.testing.assert_almost_equal(
            self.session.run(X.log_likelihood(T.to_float(data))),
            multivariate_normal(mean=mu, cov=np.eye(d)).logpdf(data),
            3
        )

    def test_log_z(self):
        d = 100
        mu = np.random.normal(size=[100, d])
        X = Gaussian([T.eye(d, batch_shape=[100]), T.to_float(mu)])
        np.testing.assert_almost_equal(
            self.session.run(X.log_z()),
            self.log_z(np.eye(d), mu),
            3
        )
        np.testing.assert_almost_equal(
            self.session.run(X.log_z('natural')),
            self.log_z(np.eye(d), mu),
            3
        )


    def test_stats1(self):
        d = 2
        mu = np.zeros([10, d])
        X = Gaussian([T.eye(d, batch_shape=[10]), T.to_float(mu)])
        stats = self.stats(np.eye(d), mu)
        stats_ = self.session.run(X.expected_sufficient_statistics())
        np.testing.assert_almost_equal(
            stats_[Stats.X],
            stats[Stats.X],
        )
        np.testing.assert_almost_equal(
            stats_[Stats.XXT],
            stats[Stats.XXT],
        )

    def test_stats2(self):
        d = 2
        mu = np.ones([10, d])
        X = Gaussian([T.eye(d, batch_shape=[10]), T.to_float(mu)])
        stats = self.stats(np.eye(d), mu)
        stats_ = self.session.run(X.expected_sufficient_statistics())
        np.testing.assert_almost_equal(
            stats_[Stats.X],
            stats[Stats.X],
        )
        np.testing.assert_almost_equal(
            stats_[Stats.XXT],
            stats[Stats.XXT],
        )

    def test_stats3(self):
        d = 2
        mu = np.random.normal(size=[10, d])
        X = Gaussian([T.eye(d, batch_shape=[10]), T.to_float(mu)])
        stats = self.stats(np.eye(d), mu)
        stats_ = self.session.run(X.expected_sufficient_statistics())
        np.testing.assert_almost_equal(
            stats_[Stats.X],
            stats[Stats.X],
            3
        )
        np.testing.assert_almost_equal(
            stats_[Stats.XXT],
            stats[Stats.XXT],
            3
        )

    def test_natural(self):
        d = 2
        mu = np.random.normal(size=[10, d])
        X = Gaussian([T.eye(d, batch_shape=[10]), T.to_float(mu)])
        param = self.natparam(np.tile(np.eye(d)[None], [10, 1, 1]), mu)
        param_ = self.session.run(X.get_parameters('natural'))
        np.testing.assert_almost_equal(
            param_[Stats.X],
            param[Stats.X],
            3
        )
        np.testing.assert_almost_equal(
            param_[Stats.XXT],
            param[Stats.XXT],
            3
        )

    def test_posterior1(self):
        d = 2
        y = np.random.normal(size=d)
        X = Gaussian([T.eye(d), T.zeros(d)])
        Y = Gaussian([T.eye(d), X])

        log_likelihood = Y.log_likelihood(T.to_float(y))
        stats = X.statistics()
        params = ({
            stat: grad
            for stat, grad in zip(
                stats,
                T.grad(log_likelihood, [X.statistic(stat) for stat in stats])
            )
        })
        prior = X.get_parameters('natural')
        posterior = self.session.run(Gaussian.natural_to_regular({
            stat: params[stat] + prior[stat]
            for stat in X.statistics()
        }))
        actual = [np.eye(2) * 0.5, y/2.]
        [np.testing.assert_almost_equal(a, b, 3)
         for a, b in zip(actual, posterior)
        ]

    def test_posterior2(self):
        d = 10
        y = np.random.normal(size=[10, d])
        X = Gaussian([T.eye(d), T.zeros(d)])
        Y = Gaussian([T.eye(d), X])

        log_likelihood = Y.log_likelihood(T.to_float(y))
        stats = X.statistics()
        params = ({
            stat: grad
            for stat, grad in zip(
                stats,
                T.grad(log_likelihood, [X.statistic(stat) for stat in stats])
            )
        })
        prior = X.get_parameters('natural')
        posterior = self.session.run(Gaussian.natural_to_regular({
            stat: params[stat] + prior[stat]
            for stat in X.statistics()
        }))
        actual = [np.eye(10) / (y.shape[0] + 1), y.mean(axis=0) * (1 - 1./ (y.shape[0] + 1))]
        [np.testing.assert_almost_equal(a, b, 3)
         for a, b in zip(actual, posterior)
        ]
