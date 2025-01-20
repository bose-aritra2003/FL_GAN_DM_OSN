from keras.models import Sequential
from keras.layers import (
    Dense,
    Dropout,
    Flatten,
    BatchNormalization,
    Conv2D,
    MaxPooling2D,
    SpatialDropout2D
)
from keras.optimizers import Adam
from keras.regularizers import l2


def GAN_net(in_shape=(256, 256, 3)):
    """
    Improved GAN_net: A deeper CNN for binary classification of GAN-generated images.

    Args:
        in_shape (tuple): Input shape of the images (default is 64x64x3 for RGB).

    Returns:
        model (Sequential): Compiled Keras Sequential model.
    """
    model = Sequential()

    # First Convolutional Block
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=in_shape, kernel_regularizer=l2(0.001)))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(SpatialDropout2D(0.2))

    # Second Convolutional Block
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', kernel_regularizer=l2(0.001)))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(SpatialDropout2D(0.3))

    # Third Convolutional Block
    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu', kernel_regularizer=l2(0.001)))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(SpatialDropout2D(0.4))

    # Fourth Convolutional Block
    model.add(Conv2D(256, kernel_size=(3, 3), activation='relu', kernel_regularizer=l2(0.001)))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(SpatialDropout2D(0.5))

    # Flatten and Fully Connected Layers
    model.add(Flatten())
    model.add(Dense(256, activation='relu', kernel_regularizer=l2(0.001)))
    model.add(Dropout(0.5))

    model.add(Dense(128, activation='relu', kernel_regularizer=l2(0.001)))
    model.add(Dropout(0.5))

    # Output Layer for Binary Classification
    model.add(Dense(1, activation='sigmoid'))

    # Compile the Model
    optimizer = Adam(learning_rate=0.0001)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

    return model
