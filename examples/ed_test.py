import edward as ed
from deepx import T
import tensorflow as tf
from edward.models import Normal, Bernoulli

mu = Normal(tf.constant(0.0), tf.constant(0.00001))
x = Normal(mu, tf.constant(0.0001))

mu_x = ed.complete_conditional(mu)

sess = T.interactive_session()
