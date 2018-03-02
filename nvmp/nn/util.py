from deepx import T

def log1pexp(x):
    return T.log(1 + T.exp(x))
