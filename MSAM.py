from keras.layers import Activation, Multiply, Conv3D, BatchNormalization, Add, Input
from keras.utils import plot_model
from keras.models import Model


def MSAM(inputs):
    # 第一层
    F3_1 = Conv3D(16, padding="same", kernel_size=(1, 1, 3), strides=(1, 1, 1))(inputs)
    F3_1_1 = BatchNormalization()(F3_1)
    F3_2 = Conv3D(16, padding="same", kernel_size=(3, 3, 1), strides=(1, 1, 1))(F3_1_1)
    F3_2_1 = BatchNormalization()(F3_2)

    # 第二层
    F7_1 = Conv3D(16, padding="same", kernel_size=(1, 1, 7), strides=(1, 1, 1))(inputs)
    F7_1_1 = BatchNormalization()(F7_1)
    F7_2 = Conv3D(16, padding="same", kernel_size=(7, 7, 1), strides=(1, 1, 1))(F7_1_1)
    F7_2_1 = BatchNormalization()(F7_2)

    # 第三次
    F11_1 = Conv3D(16, padding="same", kernel_size=(1, 1, 11), strides=(1, 1, 1))(inputs)
    F11_1_1 = BatchNormalization()(F11_1)
    F11_2 = Conv3D(16, padding="same", kernel_size=(11, 11, 1), strides=(1, 1, 1))(F11_1_1)
    F11_2_1 = BatchNormalization()(F11_2)

    F = Add()([F3_2_1, F7_2_1, F11_2_1])

    x = Activation("sigmoid")(F)
    outputs = Multiply()([inputs, x])
    return outputs










