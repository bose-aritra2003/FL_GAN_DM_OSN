from keras.layers import Input, Conv2D 
from keras.layers import MaxPool2D, Flatten, Dense
from keras.models import Model
# input

def VGG16(input_shape= (224,224,3),classes =2):
    input = Input(shape =input_shape)
    # 1st Conv Block

    x = Conv2D (filters =64, kernel_size =3, padding ='same', activation='relu')(input)
    x = Conv2D (filters =64, kernel_size =3, padding ='same', activation='relu')(x)
    x = MaxPool2D(pool_size =2, strides =2, padding ='same')(x)
    # 2nd Conv Block

    x = Conv2D (filters =128, kernel_size =3, padding ='same', activation='relu')(x)
    x = Conv2D (filters =128, kernel_size =3, padding ='same', activation='relu')(x)
    x = MaxPool2D(pool_size =2, strides =2, padding ='same')(x)
    # 3rd Conv block

    x = Conv2D (filters =256, kernel_size =3, padding ='same', activation='relu')(x)
    x = Conv2D (filters =256, kernel_size =3, padding ='same', activation='relu')(x)
    x = Conv2D (filters =256, kernel_size =3, padding ='same', activation='relu')(x)
    x = MaxPool2D(pool_size =2, strides =2, padding ='same')(x)
    # 4th Conv block

    x = Conv2D (filters =512, kernel_size =3, padding ='same', activation='relu')(x)
    x = Conv2D (filters =512, kernel_size =3, padding ='same', activation='relu')(x)
    x = Conv2D (filters =512, kernel_size =3, padding ='same', activation='relu')(x)
    x = MaxPool2D(pool_size =2, strides =2, padding ='same')(x)

    # 5th Conv block

    x = Conv2D (filters =512, kernel_size =3, padding ='same', activation='relu')(x)
    x = Conv2D (filters =512, kernel_size =3, padding ='same', activation='relu')(x)
    x = Conv2D (filters =512, kernel_size =3, padding ='same', activation='relu')(x)
    x = MaxPool2D(pool_size =2, strides =2, padding ='same')(x)
    # Fully connected layers

    x = Flatten()(x)
    x   = Dense(units = 4096, activation ='relu')(x)
    x = Dense(units = 4096, activation ='relu')(x)
    output = Dense(classes, activation ='sigmoid')(x)
    # creating the model

    model = Model (inputs=input, outputs =output)
    return model


model = VGG16(input_shape=(256, 256, 3), classes=2)
model.compile(
    optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
)
model.summary()