from .pyramidnet import *
from .densenet import *
from .resnet import *
from .wideresnet import *
from .resnext import *


def get_model(arch, dataset, num_aux_classes):
    valid_archs = ['wideresnet', 'resnext', 'pyramidnet', 'resnet']
    if arch not in valid_archs:
        raise ValueError('Supported models are: {} \n'
                         'but given {}'.format(valid_archs, arch))
    valid_datasets = ['cifar10', 'cifar100', 'fmnist']
    if dataset not in valid_datasets:
        raise ValueError('Supported datasets are: {} \n'
                         'but given {}'.format(valid_datasets, dataset))
    if dataset == 'cifar10':
        num_classes = 10
        num_colors = 3
    elif dataset == 'fmnist':
        num_classes = 10
        num_colors = 1
    else:
        num_classes = 100
        num_colors = 3

    if arch == 'wideresnet':
        depth = 28
        widen_factor = 10
        model = WideResNetAux(depth, num_classes,
                              num_aux_classes, num_colors, widen_factor)
        print('Using WideResNet-{}-{}'.format(depth, num_classes))
    elif arch == 'resnet':
        depth = 110
        model = ResNetAux(depth, num_classes, num_aux_classes,
                          num_colors, bottleneck=True)
        print('Using resnet-{}'.format(depth))
    elif arch == 'pyramidnet':
        depth = 272
        alpha = 200
        model = PyramidNetAux(depth, alpha, num_classes,
                              num_aux_classes, num_colors, bottleneck=True)
        print('Using PyramidNet-{}-{}'.format(depth, alpha))
    elif arch == 'resnext':
        cardinality = 8
        depth = 29
        widen_factor = 4
        model = resnextAux(cardinality, depth, num_classes,
                           num_aux_classes, widen_factor=4)
        print('Using resneXt-{}-{}-{}'.format(cardinality, depth, widen_factor))

    return model
