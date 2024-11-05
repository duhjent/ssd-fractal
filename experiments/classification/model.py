from torchvision.models import vgg16

from models.fractalnet.fractal_net import FractalNet


def create_fractalnet():
    return FractalNet(
        data_shape=(3, 64, 64, 200),
        n_columns=4,
        init_channels=64,
        p_ldrop=0.15,
        dropout_probs=[0.0, 0.1, 0.2, 0.3, 0.4],
        gdrop_ratio=0.5,
    )


def create_vgg16():
    return vgg16(num_classes=200)
