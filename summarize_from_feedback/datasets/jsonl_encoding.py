import base64

import numpy as np


def encode_example(example):
    """
    Make numpy arrays in the example serializable.
    We use base64 since it compresses well and is JSON-compatible.
    """
    encoded_example = example.copy()
    for k in example:
        obj = example[k]
        if isinstance(obj, np.ndarray):
            # base64 encode ndarrays instead of having JSON them output in plaintext
            del encoded_example[k]
            encoded_example["__np_" + k] = dict(
                shape=obj.shape,
                dtype=obj.dtype.name,
                data=base64.b64encode(obj.tobytes()).decode("utf-8"),
            )
        elif isinstance(obj, bytes):
            if k.endswith("/utf8"):
                encoded_example[k] = obj.decode("utf-8")
            else:
                del encoded_example[k]
                encoded_example["__bytes_" + k] = base64.b64encode(obj).decode("utf-8")
        elif isinstance(obj, np.integer):
            # JSON refuses to serialize np.integer, so convert it to plain integer
            encoded_example[k] = int(obj)
        elif isinstance(obj, np.floating):
            # JSON refuses to serialize np.floating, convert it to plain float
            encoded_example[k] = float(obj)
        elif isinstance(obj, list):
            # Same as above, JSON doesn't know how to serialize np.integer so we convert each element to int
            if all(isinstance(i, np.integer) for i in obj):
                encoded_example[k] = [int(i) for i in obj]
    return encoded_example


def decode_example(example):
    """
    Decode all numpy arrays in the example.
    """
    decoded_example = example.copy()
    for k in example:
        if k.startswith("__np_"):
            to_decode = decoded_example.pop(k)
            decoded_example[k[5:]] = np.frombuffer(
                base64.b64decode(to_decode["data"]), dtype=to_decode["dtype"]
            ).reshape(to_decode["shape"])
        elif k.startswith("__bytes_"):
            to_decode = decoded_example.pop(k)
            decoded_example[k[8:]] = base64.b64decode(to_decode.encode("utf-8"))
    return decoded_example
