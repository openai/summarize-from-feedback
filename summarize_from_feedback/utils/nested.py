import collections.abc


def map_nested(d, f):
    new_d = {}
    for k, v in d.items():
        if isinstance(v, dict):
            new_d[k] = map_nested(v, f)
        else:
            new_d[k] = f(v)
    return new_d


def nested_reduce(list_of_nested, reducers):
    """Given list of nested dicts, returns a nested dict of reduced values

    f should be either a reducer function, or a nested dict, with values being reducer functions.
    """
    new_nested = {}
    for k in list_of_nested[0].keys():
        vals = [d[k] for d in list_of_nested]
        if isinstance(reducers, collections.abc.Callable):
            reducer = reducers
        else:
            assert isinstance(reducers, dict)
            reducer = reducers[k]
        if isinstance(list_of_nested[0][k], dict):
            new_nested[k] = nested_reduce(vals, reducer)
        else:
            new_nested[k] = reducer(vals)
    return new_nested
