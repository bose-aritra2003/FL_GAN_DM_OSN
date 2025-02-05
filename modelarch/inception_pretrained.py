import tensorflow as tf
from keras.applications import InceptionV3
from keras.models import Model
from keras.layers import Dense, Flatten, Dropout

def IncV3(input_shape=(64,64,3), num_classes=2):    

    # Path to the checkpoint file (excluding the ".index" or ".data" suffix)
    ckpt_path = "./weights/inception_v3.ckpt"

    # Load the base InceptionV3 model without the top classification layer
    base_model = InceptionV3(weights=None, include_top=False, input_shape=input_shape)

    # Restore weights from the checkpoint
    base_model.load_weights(ckpt_path)

    # Freeze the base model layers
    base_model.trainable = False

    # Add custom layers 
    x = Flatten()(base_model.output)    
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.3)(x)
    output = Dense(num_classes, activation='sigmoid')(x)  # For binary classification

    # Create the final model
    model = Model(inputs=base_model.input, outputs=output)

    #   Compile the model
    # model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Print the model summary
    # model.summary()

    return model

