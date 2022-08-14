import torch
import torch.nn as nn

from .fcn import FCNResNet101


models = {
    'FCNResNet101': FCNResNet101
}


def load_model(model_type, state_dict):
    # todo(will.brennan) - improve this... might want to save a categories file with this instead
    category_prefix = '_categories.'
    categories = [k for k in state_dict.keys() if k.startswith(category_prefix)]
    categories = [k[len(category_prefix):] for k in categories]

    model = model_type(categories)
    model.load_state_dict(state_dict)

    return model


