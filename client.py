import argparse
import flwr as fl
import tensorflow as tf
import os
import numpy as np
import cv2
from sklearn.utils import shuffle
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
from modelarch.resnet50_pretrained import Res50


# Server address
server_address = "10.24.41.216:5050"  # Update for production

# Define classes and image size
classes = ['0_real', '1_fake']
class_labels = {cls: i for i, cls in enumerate(classes)}
IMAGE_SIZE = (64, 64)

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
        history = self.model.fit(
            self.training_images,
            self.training_labels,
            batch_size=batch_size,
            epochs=epochs,
            validation_split=0.2,
            shuffle=True  # Ensures shuffling for each epoch
        )
        parameters_prime = self.model.get_weights()
        num_examples_train = len(self.training_images)

        return parameters_prime, num_examples_train, {
            "loss": history.history["loss"][-1],
            "accuracy": history.history["accuracy"][-1],
            "val_loss": history.history["val_loss"][-1],
            "val_accuracy": history.history["val_accuracy"][-1],
        }

    def evaluate(self, parameters, config):
        """Evaluate parameters on the locally held test set."""
        self.model.set_weights(parameters)
        predictions = self.model.predict(self.test_images)
        predicted_classes = np.argmax(predictions, axis=1)
        true_classes = np.argmax(self.test_labels, axis=1)

        loss, accuracy = self.model.evaluate(self.test_images, self.test_labels, verbose=0)
        auc = roc_auc_score(true_classes, predictions[:, 1])
        f1 = f1_score(true_classes, predicted_classes)
        precision = precision_score(true_classes, predicted_classes)
        recall = recall_score(true_classes, predicted_classes)

        return loss, len(self.test_images), {
            "accuracy": accuracy,
            "auc": auc,
            "f1_score": f1,
            "precision": precision,
            "recall": recall
        }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--client_number',
        dest='client_number',
        type=int,
        required=True,
        help='Used to load the dataset for the client'
    )
    args = parser.parse_args()
    client_number = args.client_number

    # Validate dataset directory
    dataset_dir = f"datasets/client{client_number}"
    if not os.path.exists(dataset_dir):
        raise FileNotFoundError(f"Dataset directory for client {client_number} does not exist.")

    print(f"Client {client_number} has been connected!")

    model = Res50(input_shape=(64,64,3))
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
