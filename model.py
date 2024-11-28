from keras.models import Sequential
from keras.layers import (
    Dense,
    Dropout,
    Flatten,
    BatchNormalization,
    Conv2D,
    MaxPooling2D
)


def GAN_net(in_shape=(64, 64, 3)):
    """
    GAN_net: A CNN for binary classification of GAN-generated images.

    Args:
        in_shape (tuple): Input shape of the images (default is 64x64x3 for RGB).

    Returns:
        model (Sequential): Compiled Keras Sequential model.
    """
    model = Sequential()

    # First Convolutional Block
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=in_shape))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Second Convolutional Block
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Third Convolutional Block
    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Flatten and Fully Connected Layers
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.4))

    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.3))

    # Output Layer for Binary Classification
    model.add(Dense(1, activation='sigmoid'))

    return model
