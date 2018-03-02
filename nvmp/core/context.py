from contextlib import contextmanager

CONTEXTS = [{}]
VARIABLE_SCOPE = ['global']

@contextmanager
def graph_context(ctx):
    CONTEXTS.append(ctx)
    yield
    CONTEXTS.pop(-1)

def context(x):
    return current_context().get(x, x)

def current_context():
    return CONTEXTS[-1]

@contextmanager
def scope(scope):
    assert scope in {'local', 'global'}
    VARIABLE_SCOPE.append(scope)
    yield
    VARIABLE_SCOPE.pop(-1)

def current_scope():
    return VARIABLE_SCOPE[-1]
