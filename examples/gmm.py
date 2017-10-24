from nvmp.stats import Dirichlet

K = 3
D = 2

pi = ~ Dirichlet(T.ones(K))
mu, sigma = NIW(T.eye(D), T.zeros(D), 1, 1)
