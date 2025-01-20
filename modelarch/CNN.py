from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, BatchNormalization, Conv2D, MaxPooling2D, Input

def GAN_net(in_shape=(64, 64, 3), num_classes=2):
    model = Sequential()

    model.add(Conv2D(32, kernel_size=(3,3), activation='relu', input_shape=in_shape))
    model.add(MaxPooling2D((2,2)))

    model.add(Conv2D(64, kernel_size=(3,3), activation='relu')) 
    model.add(MaxPooling2D((2,2)))

    model.add(Flatten())

    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.2))

    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.2))

    model.add(Dense(5, activation='softmax'))
    return model
    # model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', name='convRes'))
    
    # model.add(Conv2D(32, kernel_size=(5, 5), padding='same', name='conv1'))
    # model.add(MaxPooling2D(pool_size=(3, 3), name='pool1'))
    # model.add(BatchNormalization())
    
    # model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same', name='conv2'))
    # model.add(Conv2D(64, kernel_size=(5, 5), padding='same', name='conv3'))
    # model.add(MaxPooling2D(pool_size=(3, 3), name='pool2'))
    # model.add(BatchNormalization())
    
    # model.add(Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same', name='conv4'))
    # model.add(Conv2D(128, kernel_size=(5, 5), padding='same', name='conv5'))
    # model.add(MaxPooling2D(pool_size=(3, 3), name='pool3'))
    # model.add(BatchNormalization())
    
    # model.add(Flatten())
    # model.add(Dense(256, activation='relu'))
    # model.add(Dropout(0.5))
    # model.add(Dense(256, activation='relu'))
    # model.add(Dropout(0.5))
    # model.add(Dense(num_classes, activation='sigmoid', name='predictions'))
    
    
