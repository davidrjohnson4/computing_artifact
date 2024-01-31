"""
Utility functions
"""


def flatten_dict(d):    
    """
    Recursively flattens nested dictionaries such that the terminal, 
    non-dictionary objects are returned in a list.
    """
    out = []
    if isinstance(d, dict):
        for (k, v) in d.items():
            out.extend(flatten_dict(v))  
    else:
        out.append(d)
    return out


def flatten_dict_keys(d,
                      nest_level = 0,
                      nest_keys = [None] * 20,
                      idx_keys = []):
    """
    Recursively flattens nested dictionaries' keys; returns as a 
    list of strings representing the heirarchy, e.g.: 
    level0|level1|...). These strings match the indexes of the flattened
    dict list returned by `flatten_dict()`.
    """
    if isinstance(d, dict):
        nest_level += 1
        for (k, v) in d.items():
            nest_keys[nest_level - 1] = k
            flatten_dict_keys(v, nest_level, nest_keys, idx_keys)
    else:
        nest_keys_ = [x for x in nest_keys if x is not None]
        idx_key = "|".join(nest_keys_[:nest_level])
        idx_keys.append(idx_key)
        nest_level -= 1

    return idx_keys

