from .models.lightgcn import *

__all__ = ['get_model', 'trained_model_metainfo_list']

_models = {
    'lightgcn': lightgcn
}

def get_model(name, **kwargs):
    """
    Get supported model.

    Parameters:
    ----------
    name : str
        Name of model.

    Returns
    -------
    Module
        Resulted model.
    """
    name = name.lower()
    if name not in _models:
        raise ValueError("Unsupported model: {}".format(name))
    net = _models[name](**kwargs)
    return net
