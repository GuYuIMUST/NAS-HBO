from collections import namedtuple

Genotype = namedtuple('Genotype', 'normal normal_concat reduce reduce_concat')

PRIMITIVES = [
    'MBConv_5x5x5_3',
    'MBConv_5x5x5_6',
    'max_pool_3x3x3',
    'avg_pool_3x3x3',
    'skip_connect',
    'sep_conv_3x3x3',
    'sep_conv_5x5x5',
    'dil_conv_3x3x3',
    'dil_conv_5x5x5'
]


PC_DARTS_luna_7 = Genotype(normal=[('dil_conv_5x5x5', 0), ('MBConv_5x5x5_3', 1), ('dil_conv_5x5x5', 0), ('MBConv_5x5x5_3', 2), ('MBConv_5x5x5_6', 2), ('MBConv_5x5x5_6', 3), ('dil_conv_3x3x3', 0), ('dil_conv_3x3x3', 4)], normal_concat=range(2, 6), reduce=[('skip_connect', 1), ('MBConv_5x5x5_6', 0), ('skip_connect', 1), ('MBConv_5x5x5_3', 0), ('dil_conv_3x3x3', 3), ('skip_connect', 2), ('sep_conv_3x3x3', 4), ('dil_conv_3x3x3', 3)], reduce_concat=range(2, 6))

PCDARTS = PC_DARTS_luna_7

