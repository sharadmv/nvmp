import tqdm
import seaborn
seaborn.set_style('white')
import matplotlib.pyplot as plt
import numpy as np
np.set_printoptions(suppress=True)
from deepx import T
from deepx.nn import *
from deepx.stats import *
from nvmp import nn

K = 4
D = 2
H = 20

action = 0.1

actual = [
    [[1, 0, -action],
    [0, 1, action],
    [0, 0,   1]],
    [[1, 0, action],
    [0, 1, action],
    [0, 0,   1]],
    [[1, 0, action],
    [0, 1, -action],
    [0, 0,   1]],
    [[1, 0, -action],
    [0, 1, -action],
    [0, 0,   1]],
]

noise = 1e-4
def generate_data(N, angle=np.pi/6):
    X = np.zeros((N, H + 1, D))
    X[:, 0] = np.random.uniform(-10, 10, size=(N, D))
    matrix = np.array([[np.cos(angle), np.sin(angle)],
                       [-np.sin(angle), np.cos(angle)]])
    for i in range(N):
        for t in range(1, H + 1):
            x = X[i, t - 1]
            # if x[0] >= 0 and x[1] >= 0:
            X[i, t] = matrix.dot(x)
            # if x[0] < 0 and x[1] >= 0:
                # X[i, t] = X[i, t-1] + [-0.5, -0.5]
            # if x[0] < 0 and x[1] < 0:
                # X[i, t] = X[i, t-1] + [0.5, 0.5]
            # if x[0] >= 0 and x[1] < 0:
                # X[i, t] = X[i, t-1] + [0.5, 0.5]
    X += np.random.normal(size=X.shape, scale=np.sqrt(noise))
    return X

data = generate_data(1000)
N = data.shape[0]
yt, yt1 = data[:, :-1], data[:, 1:]
yt, yt1 = yt.reshape([-1, D]), yt1.reshape([-1, D])

transition_net = Tanh(D, 500) >> Tanh(500) >> nn.Gaussian(D)
transition_net.initialize()

rec_net = Tanh(D, 500) >> Tanh(500) >> nn.Gaussian(D)
rec_net.initialize()

Yt  = T.placeholder(T.floatx(), [None, D])
Yt1 = T.placeholder(T.floatx(), [None, D])
batch_size = T.shape(Yt)[0]
num_batches = N / T.to_float(batch_size)

Yt_message = Gaussian.pack([
    T.tile(T.eye(D)[None] * noise, [batch_size, 1, 1]),
    T.einsum('ab,ib->ia', T.eye(D) * noise, Yt)
])
Yt1_message = Gaussian.pack([
    T.tile(T.eye(D)[None] * noise, [batch_size, 1, 1]),
    T.einsum('ab,ib->ia', T.eye(D) * noise, Yt1)
])
transition = Gaussian(transition_net(Yt)).expected_value()

max_iter = 1000
tol = 1e-5
def cond(i, prev_elbo, elbo, qxt, qxt1):
    return T.logical_and(i < max_iter, abs(prev_elbo - elbo) >= tol)

def step(i, prev_elbo, elbo, qxt_param, qxt1_param):
    qxt, qxt1 = Gaussian(qxt_param, 'natural'), Gaussian(qxt1_param, 'natural')

    qxt_message = Gaussian.regular_to_natural(transition_net(qxt.sample()[0]))
    qxt1 = Gaussian(qxt_message + Yt1_message, 'natural')

    qxt1_message = Gaussian.regular_to_natural(rec_net(qxt1.sample()[0]))
    qxt = Gaussian(qxt1_message + Yt_message, 'natural')

    prev_elbo, elbo = elbo, T.sum(kl_divergence(qxt1, Gaussian(qxt_message, 'natural')))
    return i + 1, prev_elbo, elbo, qxt.get_parameters('natural'), qxt1.get_parameters('natural')

num_var, _, kl, qxt_param, qxt1_param = T.core.while_loop(cond, step, [0, float('inf'), 0.0,
                  Gaussian.regular_to_natural([
                     T.eye(D, batch_shape=[batch_size]) * 1e-4,
                     Yt]),
                  Gaussian.regular_to_natural([
                     T.eye(D, batch_shape=[batch_size]) * 1e-4,
                     Yt1]),
                ])
qxt = Gaussian(qxt_param, 'natural')
qxt1 = Gaussian(qxt1_param, 'natural')
pyt = Gaussian([
    T.tile(T.eye(D)[None] * noise, [batch_size, 1, 1]),
    qxt.expected_value()
])
pyt1 = Gaussian([
    T.tile(T.eye(D)[None] * noise, [batch_size, 1, 1]),
    qxt1.expected_value()
])
log_likelihood = T.sum(pyt.log_likelihood(Yt) + pyt1.log_likelihood(Yt1))

elbo = (log_likelihood - kl) / T.to_float(batch_size)

grads = T.gradients(-elbo, transition_net.get_parameters() + rec_net.get_parameters())
grads, _ = T.core.clip_by_global_norm(grads, 1) # gradient clipping
grads_and_vars = list(zip(grads, transition_net.get_parameters() + rec_net.get_parameters()))
train_op = T.core.train.AdamOptimizer(1e-4).apply_gradients(grads_and_vars)
# train_op = T.core.train.AdamOptimizer(1e-5).minimize(-elbo, var_list=transition_net.get_parameters() + rec_net.get_parameters())
sess = T.interactive_session()

plt.figure()
plt.ion()
plt.show()

def train(num_iters, batch_size=10):
    for i in tqdm.trange(num_iters):
        idx = np.random.permutation(N)[:batch_size]
        _, n, k, e, ll = sess.run([train_op, num_var, kl,elbo, log_likelihood], {Yt:yt[idx], Yt1:yt1[idx]})
        if i % 10 == 0:
            print(n, e, k / batch_size, ll / batch_size)
            draw()

def draw():
    X, Y = np.meshgrid(np.arange(-5, 5, 1), np.arange(-5, 5, 1))
    X = X.reshape([-1, 1])
    Y = Y.reshape([-1, 1])
    data = np.concatenate([X, Y], -1)
    angle = np.pi/6
    matrix = np.array([[np.cos(angle), np.sin(angle)],
                       [-np.sin(angle), np.cos(angle)]])
    yt1 = np.einsum('ab,ia->ib', matrix, data)
    transitions = sess.run(transition, {Yt:data, Yt1:yt1})
    field = transitions - data
    X_field = field[:, 0]
    Y_field = field[:, 1]
    plt.cla()
    plt.quiver(X, Y, X_field, Y_field, scale=10)
    plt.pause(0.01)
    plt.draw()

# def check(x):
    # mixture = sess.run(T.exp(transition(T.to_float([x]))))[0]
    # print(mixture)
    # return np.einsum('k,kab,b->a', mixture, sess.run(A), x)
# # iter(10000)
