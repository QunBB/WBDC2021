from src.model.core.ple import PLE


def get_instance(name, config):
    return {"PLE": PLE}[name](**config)
