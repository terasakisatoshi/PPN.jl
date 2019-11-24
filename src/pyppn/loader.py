import argparse
import configparser
import logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

import os

import chainer
import chainer.links as L
from chainer import initializers
import numpy as np

from .network_mobilenetv2 import MobilenetV2
from .network_resnet import ResNet50
from .network_resnet import ResNet
from .pose_utils import KEYPOINT_NAMES, EDGES

DEFAULT_MODEL_PATH = model_path = os.path.join(
    os.path.dirname(__file__),
    "../result/mobilenetv2_224x224_1.0_coco"
)


def parse_size(text):
    w, h = text.split('x')
    w = float(w)
    h = float(h)
    if w.is_integer():
        w = int(w)
    if h.is_integer():
        h = int(h)
    return w, h


def load_config(model):
    config = configparser.ConfigParser()
    config_path = os.path.join(model, 'src', 'config.ini')
    logger.info(config_path)
    config.read(config_path, 'UTF-8')
    return config


def get_network(model, **kwargs):
    if model == 'mv2':
        return MobilenetV2(**kwargs)
    elif model == 'resnet50':
        return ResNet50(**kwargs)
    elif model == 'resnet18':
        return ResNet(n_layers=18)
    elif model == 'resnet34':
        return ResNet(n_layers=34)
    else:
        raise Exception('Invalid model name')


class MyModel(chainer.Chain):

    def __init__(self, config):
        super(MyModel, self).__init__()

        with self.init_scope():
            dtype = np.float32
            self.feature_layer = get_network(config.get('model_param', 'model_name'), dtype=dtype, width_multiplier=1.0)
            ksize = self.feature_layer.last_ksize
            self.local_grid_size = parse_size(config.get('model_param', 'local_grid_size'))
            self.keypoint_names = KEYPOINT_NAMES
            self.edges = EDGES
            self.lastconv = L.Convolution2D(None,
                                            6 * len(self.keypoint_names) +
                                            self.local_grid_size[0] * self.local_grid_size[1] * len(self.edges),
                                            ksize=ksize, stride=1, pad=ksize // 2,
                                            initialW=initializers.HeNormal(1 / np.sqrt(2), dtype))

    def __call__(self, x):
        h = self.feature_layer(x)
        h = self.feature_layer.last_activation(self.lastconv(h))
        return h


def load(model_path=DEFAULT_MODEL_PATH):
    config = load_config(model_path)
    model = MyModel(config)
    chainer.serializers.load_npz(os.path.join(model_path, 'bestmodel.npz'), model)
    return model


def main():
    model_path = os.path.join(os.path.dirname(__file__), "../result/mobilenetv2_224x224_1.0_coco")
    print(model_path)
    load_model(model_path)


if __name__ == '__main__':
    main()
