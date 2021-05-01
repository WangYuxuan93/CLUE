import torch
import torch.nn as nn

# Note: x is syntax input, y is vanilla input

class ConstantGateLayer(nn.Module):
    def __init__(self, fusion_gate_const=0.2):
        super(ConstantGateLayer, self).__init__()
        self.const = fusion_gate_const

    def forward(self, x, y):
        return self.const * x + (1 - self.const) * y


class ExtraConstantGateLayer(nn.Module):
    def __init__(self, fusion_gate_const=0.2):
        super(ExtraConstantGateLayer, self).__init__()
        self.const = fusion_gate_const

    def forward(self, x, y):
        return self.const * x + y


class ExtraScalarGateLayer(nn.Module):
    def __init__(self, in_out_size):
        super(ExtraScalarGateLayer, self).__init__()
        self.alpha = nn.Parameter(torch.tensor(0.))

    def forward(self, x, y):
        weight = torch.sigmoid(self.alpha)
        return weight * x + y


class HighwayGateLayer(nn.Module):
    def __init__(self, in_out_size, bias=True):
        super(HighwayGateLayer, self).__init__()
        self.transform = nn.Linear(in_out_size, in_out_size, bias=bias)

    def forward(self, x, y):
        out_transform = torch.sigmoid(self.transform(x))
        return out_transform * x + (1 - out_transform) * y


class InputGateLayer(nn.Module):
    def __init__(self, in_out_size, bias=True):
        super(InputGateLayer, self).__init__()
        self.transform = nn.Linear(in_out_size, in_out_size, bias=False)

    def forward(self, x, y):
        out_transform = torch.sigmoid(self.transform(x))
        return out_transform * x + y


class SigmoidTanhGateLayer(nn.Module):
    def __init__(self, in_out_size, bias=True):
        super(SigmoidTanhGateLayer, self).__init__()
        self.transform1 = nn.Linear(in_out_size, in_out_size, bias=True)
        self.transform2 = nn.Linear(in_out_size, in_out_size, bias=False)

    def forward(self, x, y):
        out_transform1 = torch.sigmoid(self.transform1(y))
        y = torch.tanh(self.transform2(y))
        return x + out_transform1 * y


class GruGateLayer(nn.Module):
    """
    We make use of the GRUCell in Pytorch.
    This formulation is somewhat different from the paper: "https://arxiv.org/pdf/1910.06764.pdf"
    Stabilizing Transformers for Reinforcement Learning
    """
    def __init__(self, in_out_size, bias=True):
        super(GruGateLayer, self).__init__()
        self.gru_cell = nn.GRUCell(in_out_size, in_out_size, bias=bias)

    def forward(self, x, y):
        batch, length, units = x.shape
        x = x.view(batch * length, units)
        y = y.view(batch * length, units)

        out_transform = self.gru_cell(y, x)
        out_units = out_transform.shape[1]
        out_transform = out_transform.view(batch, length, out_units)
        return out_transform


class MultiplicativeIntegrationGateLayer(nn.Module):
    def __init__(self):
        super(MultiplicativeIntegrationGateLayer, self).__init__()

    def forward(self, x, y):
        return x * y
