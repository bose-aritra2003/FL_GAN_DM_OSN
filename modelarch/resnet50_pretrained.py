import tensorflow as tf
from keras.applications.resnet import ResNet50
from keras.models import Model
from keras.layers import Dense, Flatten, Dropout

def Res50(input_shape=(64, 64, 3), num_classes=2):
    """
    Function to create a ResNet50 model with custom top layers

    Arguments:
    input_shape -- shape of the images of the dataset
    num_classes -- integer, number of classes

    Returns:
    model -- a Model() instance in Keras
    """

    # Load the ResNet50 model pre-trained on ImageNet without the top layer
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)

    # Freeze the layers of ResNet50
    base_model.trainable = False

    # Add custom layers on top of ResNet50
    x = Flatten()(base_model.output)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.3)(x)
    output = Dense(num_classes, activation='sigmoid')(x)

    # Create the final model
    model = Model(inputs=base_model.input, outputs=output)

    # Summary of the model
    model.summary()

    return model

Res50()