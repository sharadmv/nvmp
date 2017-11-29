import numpy as np
from deepx import nn
from nvmp import T

from nvmp.stats import *
from nvmp.core import get_current_graph, trace


L = 4
D = 6
batch_size = T.placeholder(np.int32, None)

X = Gaussian([T.eye(L), T.zeros(L)])
# obs_net = nn.Relu(L, 100) >> nn.Relu(100) >> nn.Relu(D)
Y = Gaussian([T.eye(L), X])

q_X = Gaussian([T.variable(T.eye(L)), T.variable(T.zeros(L))])

sess = T.interactive_session()
