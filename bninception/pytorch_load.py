import torch
import os
from torch import nn
from .layer_factory import get_basic_layer, parse_expr
import torch.utils.model_zoo as model_zoo
import yaml

_here = os.path.dirname(__file__)


class BNInception(nn.Module):
    def __init__(self, model_path=os.path.join(_here, 'bn_inception.yaml'),
                       weights=None):
        super(BNInception, self).__init__()

        manifest = yaml.load(open(model_path))

        layers = manifest['layers']

        self._channel_dict = dict()

        self._op_list = list()
        for layer in layers:
            out_var, op, in_var = parse_expr(layer['expr'])
            if op != 'Concat':
                id, out_name, module, out_channel, in_name = get_basic_layer(
                        layer, 3 if len(self._channel_dict) == 0 else self._channel_dict[in_var[0]],
                        conv_bias=True)

                self._channel_dict[out_name] = out_channel
                setattr(self, id, module)
                self._op_list.append((id, op, out_name, in_name))
            else:
                self._op_list.append((id, op, out_var[0], in_var))
                channel = sum([self._channel_dict[x] for x in in_var])
                self._channel_dict[out_var[0]] = channel

        if weights is None:
            self.load_state_dict(torch.utils.model_zoo.load_url('http://data.lip6.fr/cadene/pretrainedmodels/bn_inception-52deb4733.pth'))
        else:
            self.load_state_dict(weights)

    def forward(self, input, output=None):
        data_dict = dict()
        data_dict[self._op_list[0][-1]] = input

        for op in self._op_list:
            if op[1] != 'Concat' and op[1] != 'InnerProduct':
                data_dict[op[2]] = getattr(self, op[0])(data_dict[op[-1]])
            elif op[1] == 'InnerProduct':
                x = data_dict[op[-1]]
                data_dict[op[2]] = getattr(self, op[0])(x.view(x.size(0), -1))
            else:
                try:
                    data_dict[op[2]] = torch.cat(tuple(data_dict[x] for x in op[-1]), 1)
                except:
                    for x in op[-1]:
                        print(x,data_dict[x].size())
                    raise
        if output is None:
            return data_dict[self._op_list[-1][2]]
        else:
            return data_dict[output]


class InceptionV3(BNInception):
    def __init__(self, model_path='model_zoo/bninception/inceptionv3.yaml', num_classes=101,
                 weight_url='https://yjxiong.blob.core.windows.net/models/inceptionv3-cuhk-0e09b300b493bc74c.pth'):
        super(InceptionV3, self).__init__(model_path=model_path, weight_url=weight_url, num_classes=num_classes)
