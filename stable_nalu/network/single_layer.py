import torch
from ..abstract import ExtendedTorchModule
from ..layer import GeneralizedLayer, BasicLayer


class SingleLayerNetwork(ExtendedTorchModule):
    UNIT_NAMES = GeneralizedLayer.UNIT_NAMES

    def __init__(self, unit_name, input_size=2, output_size=1, writer=None, nac_mul='none', eps=1e-7, **kwags):
        super().__init__('network', writer=writer, **kwags)
        self.unit_name = unit_name  # layer_type arg
        self.input_size = input_size
        self.nac_mul = nac_mul
        self.eps = eps

        # special case: NAC* requires a NAC+ as it's layer. The operation is dealt with in the forward pass of this class
        if unit_name == 'MNAC':
            unit_name = 'NAC'

        # self.norm_unit = GeneralizedLayer(input_size, input_size*2, 'NNU', writer=self.writer, name='norm_layer')

        self.layer_1 = GeneralizedLayer(input_size, output_size,
                                        'linear' if unit_name in BasicLayer.ACTIVATIONS else unit_name,
                                        writer=self.writer,
                                        name='layer_1',
                                        eps=eps, **kwags)
        # self.denorm_unit = GeneralizedLayer(input_size, output_size*2, 'NNU', writer=self.writer, name='norm_layer')

        self.reset_parameters()
        self.z_1_stored = None

    def reset_parameters(self):
        self.layer_1.reset_parameters()

    def regualizer(self):
        if self.nac_mul == 'max-safe':
            return super().regualizer({
                'z': torch.mean(torch.relu(1 - self.z_1_stored))
            })
        else:
            return super().regualizer()

    def normalise(self, input):
        return input / input.abs().sum(dim=1).unsqueeze(1)

    def forward(self, input):
        self.writer.add_summary('x', input)
        # inital_input = input
        # input = self.norm_unit((input, input))
        # input = self.normalise(inital_input)

        # do mulitplicative path (NAC*)
        if self.unit_name == 'MNAC':
            z_1 = torch.exp(self.layer_1(torch.log(torch.abs(input) + self.eps)))
        else:
            z_1 = self.layer_1(input)

        self.z_1_stored = z_1
        self.writer.add_summary('z_1', z_1)

        # z_1 = self.denorm_unit((z_1, inital_input))

        return z_1

    def extra_repr(self):
        return 'unit_name={}, input_size={}'.format(
            self.unit_name, self.input_size
        )
