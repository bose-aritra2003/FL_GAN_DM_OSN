import numpy as np
import os

classes = ['sg2', "fb", 'ig', 'tg', 'wa']
class_labels = {classes: i for i, classes in enumerate(classes)}

def load_dataset(client_number):
    directory = f"COMPRESSED/client{client_number}"
    sub_directories = ["train", "val"]
    loaded_dataset = []
    for sub_directory in sub_directories:
        path = os.path.join(directory, sub_directory)
        images, labels = [], []
        print(f"Client dataset loading {sub_directory}")
        for folder in os.listdir(path):
            if folder not in class_labels:
                continue
            label = class_labels[folder]
            for file in os.listdir(os.path.join(path, folder)):
                img_path = os.path.join(os.path.join(path, folder), file)
                print(f"Path: {img_path}, Label: {label}")  # Print the path and label
                # image = cv2.imread(img_path)
                # if image is None:
                #     continue
                # image = cv2.resize(image, IMAGE_SIZE)
                # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                # images.append(image)

                labels.append(label)
        # images = np.array(images, dtype='float32')
        labels = np.array(labels, dtype='int32')
        loaded_dataset.append((images, labels))
    return loaded_dataset

# Example usage
load_dataset(2)