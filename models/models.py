models = {}


def register(name):
    def decorator(cls):
        models[name] = cls
        return cls
    return decorator


def make(model_name, cfg, sd=None):
    model = models[model_name](cfg)
    if sd is not None:
        model.load_state_dict(sd)
    return model