from deepx import T
import tensorflow as tf
from edward.models import Normal, Bernoulli

mu = Normal(tf.constant(0.0), tf.constant(0.00001))
x = Bernoulli(probs=mu - 5)

sess = T.interactive_session()
