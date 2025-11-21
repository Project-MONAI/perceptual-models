# Optional list of dependencies required by the package
dependencies = ["torch"]

from medicalnet.resnet import (  # noqa: E402
    medicalnet_resnet10_23datasets,
    medicalnet_resnet50_23datasets,
)

from radimagenet.resnet import (  # noqa: E402
    radimagenet_resnet50,
)

__all__ = [
    "medicalnet_resnet10_23datasets",
    "medicalnet_resnet50_23datasets",
    "radimagenet_resnet50",
]

def medicalnet_resnet10_23datasets_hub(*args, **kwargs):
    return medicalnet_resnet10_23datasets(*args, **kwargs)

def medicalnet_resnet50_23datasets_hub(*args, **kwargs):
    return medicalnet_resnet50_23datasets(*args, **kwargs)

def radimagenet_resnet50_hub(*args, **kwargs):
    return radimagenet_resnet50(*args, **kwargs)