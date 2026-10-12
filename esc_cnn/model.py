"""CNN architecture for log-mel spectrogram classification."""
import keras
from keras import layers

from . import config as C


BN_MOMENTUM = 0.9   # moving statistics follow the changing weights quickly


def conv_block(x, filters, name):
    """Conv(3x3) -> BatchNorm -> ReLU -> MaxPool(2x2).

    No dropout inside the convolutional blocks: dropout placed before a BatchNorm layer makes the
    activation variance differ between training and inference ("variance shift"), which corrupts the
    BN moving statistics. Regularisation comes from SpecAugment, mixup and the dropout in the head.
    """
    x = layers.Conv2D(filters, 3, padding="same", use_bias=False, name=f"{name}_conv")(x)
    x = layers.BatchNormalization(momentum=BN_MOMENTUM, name=f"{name}_bn")(x)
    x = layers.ReLU(name=f"{name}_relu")(x)
    return layers.MaxPooling2D(2, name=f"{name}_pool")(x)


def build_cnn(num_classes=50, input_shape=(C.N_MELS, C.N_FRAMES, 1), widths=(32, 64, 128, 256)):
    """Four convolutional blocks + global average pooling + dense classifier.

    Input  : 128 mel bands x 216 frames x 1 channel (log-mel "image" of a 5 s clip)
    Output : softmax over `num_classes` sound classes
    """
    inp = keras.Input(shape=input_shape, name="log_mel")
    x = inp
    for i, w in enumerate(widths, start=1):
        x = conv_block(x, w, name=f"block{i}")
    x = layers.Conv2D(widths[-1], 3, padding="same", use_bias=False, name="block5_conv")(x)
    x = layers.BatchNormalization(momentum=BN_MOMENTUM, name="block5_bn")(x)
    x = layers.ReLU(name="block5_relu")(x)
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dropout(0.4, name="head_drop1")(x)
    x = layers.Dense(256, activation="relu", name="fc1")(x)
    x = layers.Dropout(0.3, name="head_drop2")(x)
    out = layers.Dense(num_classes, activation="softmax", name="class_probs")(x)
    return keras.Model(inp, out, name="ESC_CNN")
