MODELS = {}


def register(func):
    """the decorator used to add network plugins"""
    MODELS[func.__name__] = func
    return func


__all__ = [MODELS]
