from .iresnet import iresnet18, iresnet34, iresnet50, iresnet100, iresnet200


def get_model(name, **kwargs):
    if name == "resnet18":
        return iresnet18(False, **kwargs)
    elif name == "resnet34":
        return iresnet34(False, **kwargs)
    elif name == "resnet50":
        return iresnet50(False, **kwargs)
    elif name == "resnet100":
        return iresnet100(False, **kwargs)
    elif name == "resnet200":
        return iresnet200(False, **kwargs)
    else:
        raise ValueError()
