## Initial Parameters
init_params = {
    'variational_distribution': {
        'mean_scale': 0.0,
        'variance_scale': 1.0
    },
}

def get_init_params(params: dict):
    """
        Returns params with any missing keys filled in with init_params
    """
    for key in init_params.keys():
        if key not in params.keys():
            params[key] = init_params[key]
    return params
