import os
import cv2
import numpy as np
import flwr as fl
from typing import Dict, Optional, Tuple
from model import GAN_net
import keras
from modelarch.resnet50 import ResNet50

# Server address
server_address = "0.0.0.0:5050"

# Define classes and image size
classes = ['0_real', '1_fake']
class_labels = {cls: i for i, cls in enumerate(classes)}
IMAGE_SIZE = (256, 256)

# Federated learning configuration
federatedLearningcounts = 30
local_client_epochs = 8
local_client_batch_size = 32


def main():
    model = ResNet50(input_shape=(256, 256, 3), classes=2)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Federated Averaging strategy with additional configurations
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=0.3,
        fraction_evaluate=0.2,
        min_fit_clients=3,
        min_evaluate_clients=3,
        min_available_clients=3,
        evaluate_fn=get_evaluate_fn(model),
        on_fit_config_fn=fit_config,
        on_evaluate_config_fn=evaluate_config,
        initial_parameters=fl.common.ndarrays_to_parameters(model.get_weights()),
    )

    # Start server with the federated learning configuration
    fl.server.start_server(
        server_address=server_address,
        config=fl.server.ServerConfig(num_rounds=federatedLearningcounts),
        strategy=strategy
    )


def load_dataset():
    """
    Load the test dataset for server-side evaluation
    """
    directory = 'datasets/server'
    sub_directory = "test"
    path = os.path.join(directory, sub_directory)
    images, labels = [], []
    print(f"Loading client dataset from {sub_directory}...")
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

    # Normalize images to [0, 1]
    
    images = np.array(images, dtype='float32') / 255.0
    labels = np.array(labels, dtype='int32')
    labels = keras.utils.to_categorical(labels, num_classes=2)

    return images, labels


def get_evaluate_fn(model):
    """
    Returns an evaluation function that will be called after each training round
    """
    test_images, test_labels = load_dataset()
    print("[Server] Test images shape:", test_images.shape)
    print("[Server] Test labels shape:", test_labels.shape)

    def evaluate(server_round: int, parameters: fl.common.NDArrays, config: Dict[str, fl.common.Scalar]) -> Optional[
        Tuple[float, Dict[str, fl.common.Scalar]]]:
        print(f"=== Server round {server_round}/{federatedLearningcounts} ===")

        # Set model weights from federated learning
        model.set_weights(parameters)

        # Evaluate the model on the test dataset
        loss, accuracy = model.evaluate(test_images, test_labels, verbose=0)
        print(f"Round {server_round}: Accuracy = {accuracy}")

        # Save the model at regular intervals (e.g., after every 10 rounds)
        if server_round % 10 == 0 or server_round == federatedLearningcounts:
            os.makedirs('Models', exist_ok=True)
            print("Saving model...")
            model.save(f'Models/gan_net_round_{server_round}.keras')

        return loss, {"accuracy": accuracy}

    return evaluate


def fit_config(server_round: int):
    """
    Custom configuration for the local training
    """
    return {
        "batch_size": local_client_batch_size,
        "local_epochs": local_client_epochs,
    }


def evaluate_config(server_round: int):
    """
    Configuration for evaluation on the server side
    """
    return {"val_steps": 4}


if __name__ == "__main__":
    main()
