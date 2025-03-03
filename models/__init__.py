from .mobilenetv2 import mobile_half
from .resnet import (resnet8, resnet8x4, resnet14, resnet20,
                      resnet32, resnet32x4, resnet44, resnet56, resnet110)
from .resnetv2 import (resnet18, resnet18x2, resnet34, resnet34x4,
                       resnext50_32x4d, wide_resnet50_2)
from .resnetv2_org import ResNet50
from .ShuffleNetv1 import ShuffleV1
from .ShuffleNetv2 import ShuffleV2
from .vgg import vgg8_bn, vgg11_bn, vgg13_bn, vgg16_bn, vgg19_bn
from .vggv2 import vgg11_bn as vgg11_imagenet
from .vggv2 import vgg13_bn as vgg13_imagenet
from .wrn import wrn_16_1, wrn_16_2, wrn_40_1, wrn_40_2

model_dict = {
    'resnet8': resnet8,
    'resnet14': resnet14,
    'resnet20': resnet20,
    'resnet32': resnet32,
    'resnet44': resnet44,
    'resnet56': resnet56,
    'ResNet18': resnet18,
    'ResNet18Double': resnet18x2,
    'ResNet34': resnet34,
    'ResNet50': ResNet50,
    'resnet110': resnet110,
    'resnet8x4': resnet8x4,
    'resnet32x4': resnet32x4,
    'resnext50_32x4d': resnext50_32x4d,
    'resnet34x4': resnet34x4,
    'wrn_16_1': wrn_16_1,
    'wrn_16_2': wrn_16_2,
    'wrn_40_1': wrn_40_1,
    'wrn_40_2': wrn_40_2,
    'wrn_50_2': wide_resnet50_2,
    'vgg8': vgg8_bn,
    'vgg11': vgg11_bn,
    'vgg13': vgg13_bn,
    'vgg16': vgg16_bn,
    'vgg19': vgg19_bn,
    'MobileNetV2': mobile_half,
    'ShuffleV1': ShuffleV1,
    'ShuffleV2': ShuffleV2,
}
