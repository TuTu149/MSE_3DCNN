from keras.layers import Dense, Input, Reshape, Activation, Multiply, GlobalAvgPool3D
from keras.models import Model

def se_block(inputs, ratio=4):
    # inputs h,w,C
    channel = inputs._keras_shape[-1]
    #C
    x = GlobalAvgPool3D()(inputs)
    # 1，1，c
    x = Reshape([1, 1, -1])(x)
    x = Dense(channel // ratio)(x)
    x = Activation('relu')(x)
    x = Dense(channel)(x)
    x = Activation("sigmoid")(x)
    out = Multiply()([x, inputs])
    return out

# inputs = Input([5, 5, 200, 512])
# x = se_block(inputs)
# model = Model(inputs, x)
# model.summary()


"""
Total params: 131,712
Trainable params: 131,712
Non-trainable params: 0
"""
