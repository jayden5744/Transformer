import torch.nn as nn


def xavier_uniform_initialize(submodule):
    nn.init.xavier_uniform_(submodule.weight)


def xavier_normal_initialize(submodule):
    nn.init.xavier_normal_(submodule.weight)


def he_uniform_initialize(submodule):
    nn.init.kaiming_uniform_(submodule.weight, nonlinearity="relu")


def he_normal_initialize(submodule):
    nn.init.kaiming_normal_(submodule.weight, nonlinearity="relu")