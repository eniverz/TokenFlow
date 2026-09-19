from collections.abc import Mapping, Sequence

import torch

from . import BaseTransform


# MARK: Repack keys

def flatten_dict(d, parent_key='', sep='/'):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, Mapping):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def unflatten_dict(d, sep='/'):
    result = {}
    for k, v in d.items():
        keys = k.split(sep)
        current = result
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        current[keys[-1]] = v
    return result


class RepackTransform(BaseTransform):
    """
    Repack the keys of the input data according to the provided mapping.

    :param repack: A dictionary specifying how to repack the keys. At least has these keys: "actions", "states", "images", "prompt". The values are the new keys in the output data.
    """
    def __init__(self, repack: dict):
        super().__init__()
        self.repack = repack

    def __call__(self, data):
        flatten_data = flatten_dict(data)
        flatten_keys = flatten_dict(self.repack)
        repacked_data = {k: flatten_data.get(v) for k, v in flatten_keys.items()}
        return unflatten_dict(repacked_data)
