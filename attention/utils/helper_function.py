from attention.utils.config import AttrDict


def switch_dropout_off(params):
    for key, val in params.items():
        if isinstance(val, dict):
            params[key] = AttrDict.from_nested_dict(switch_dropout_off(val))
        if key == "dropout_rate":
            params[key] = 0.0
    return params
