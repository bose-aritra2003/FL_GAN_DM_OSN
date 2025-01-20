import argparse
import flwr as fl
import tensorflow as tf
import os
import numpy as np
import cv2
from sklearn.utils import shuffle
from modelarch.resnet50 import ResNet50

# Server address
server_address = "10.24.109.224:5050"  # Update for production

# Define classes and image size
classes = ['0_real', '1_fake']
class_labels = {cls: i for i, cls in enumerate(classes)}
IMAGE_SIZE = (64,64)

# Define Flower client
class CifarClient(fl.client.NumPyClient):
    def __init__(self, model, training_images, training_labels, test_images, test_labels):
        self.model = model
        self.training_images = training_images
        self.training_labels = training_labels
        self.test_images = test_images
        self.test_labels = test_labels

    def get_parameters(self, config):
        """Get parameters of the local model."""
        return self.model.get_weights()

    def fit(self, parameters, config):
        """Train parameters on the locally held training set."""
        self.model.set_weights(parameters)
        batch_size = config["batch_size"]
        epochs = config["local_epochs"]

        best_val_accuracy = -1  # Track the best validation accuracy
        best_weights = None     # Store the weights corresponding to the best validation accuracy
        best_results = {}       # Store the results for the best epoch

        for epoch in range(epochs):
            # Train for one epoch
            history = self.model.fit(
                self.training_images,
                self.training_labels,
                batch_size=batch_size,
                epochs=epochs,
                validation_split=0.2,
                shuffle=True,
                verbose=0  # Suppress verbose output for each epoch
            )

            # Retrieve metrics for the current epoch
            loss = history.history["loss"][-1]
            accuracy = history.history["accuracy"][-1]
            val_loss = history.history["val_loss"][-1]
            val_accuracy = history.history["val_accuracy"][-1]

            # Check if the current epoch has the best validation accuracy
            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                best_weights = self.model.get_weights()
                best_results = {
                    "loss": loss,
                    "accuracy": accuracy,
                    "val_loss": val_loss,
                    "val_accuracy": val_accuracy,
                }

        # Use the best weights for reporting results
        self.model.set_weights(best_weights)
        num_examples_train = len(self.training_images)
        return best_weights, num_examples_train, best_results

    def evaluate(self, parameters, config):
        """Evaluate parameters on the locally held test set."""
        self.model.set_weights(parameters)
        loss, accuracy = self.model.evaluate(self.test_images, self.test_labels)
        num_examples_test = len(self.test_images)
        return loss, num_examples_test, {"accuracy": accuracy}


    def evaluate(self, parameters, config):
        """Evaluate parameters on the locally held test set."""
        self.model.set_weights(parameters)
        loss, accuracy = self.model.evaluate(self.test_images, self.test_labels)
        num_examples_test = len(self.test_images)
        return loss, num_examples_test, {"accuracy": accuracy}



def main():
    client_argumentparser = argparse.ArgumentParser()
    client_argumentparser.add_argument(
        '--client_number',
        dest='client_number',
        type=int,
        required=True,
        help='Used to load the dataset for the client'
    )
    client_argumentparser = client_argumentparser.parse_args()
    client_number = client_argumentparser.client_number

    # Validate dataset directory
    dataset_dir = f"datasets/client{client_number}"
    if not os.path.exists(dataset_dir):
        raise FileNotFoundError(f"Dataset directory for client {client_number} does not exist.")

    print(f"Client {client_number} has been connected!")

    model = ResNet50(input_shape=(64,64,3))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    (training_images, training_labels), (test_images, test_labels) = load_dataset(client_number)
    training_images, training_labels = shuffle(training_images, training_labels, random_state=25)

    client = CifarClient(model, training_images, training_labels, test_images, test_labels)
    fl.client.start_numpy_client(server_address=server_address, client=client)


def load_dataset(client_number):
    """Load and preprocess the dataset for the client."""
    directory = f"datasets/client{client_number}"
    sub_directories = ["train", "val"]
    images = []
    labels = []
    print(f"Loading dataset from {directory} for client {client_number}...")

    for sub_directory in sub_directories:
        path = os.path.join(directory, sub_directory)

        for folder in os.listdir(path):
            if folder not in class_labels:
                continue
            label = class_labels[folder]

            for file in os.listdir(os.path.join(path, folder)):
                img_path = os.path.join(path, folder, file)
                image = cv2.imread(img_path)
                if image is None:
                    continue
                image = cv2.resize(image, IMAGE_SIZE)
                images.append(image)
                labels.append(label)

    images = np.array(images, dtype='float32') / 255.0  # Normalize to [0, 1]
    labels = np.array(labels, dtype='int32')

    # Split the dataset into training and test datasets
    num_train = int(0.8 * len(images))
    training_images, test_images = images[:num_train], images[num_train:]
    training_labels, test_labels = labels[:num_train], labels[num_train:]
    training_labels = tf.keras.utils.to_categorical(training_labels, num_classes=2)
    test_labels = tf.keras.utils.to_categorical(test_labels, num_classes=2)

    return (training_images, training_labels), (test_images, test_labels)


if __name__ == "__main__":
    main()
