# Server Code
from typing import Dict, Optional, Tuple
import flwr as fl
import tensorflow as tf
import os
import cv2
import numpy as np
from model import GAN_net

# Server address
server_address = "0.0.0.0:5050"
# Update to the server's actual IP address in production

# Define classes and image size
classes = ['real', 'fake']
class_labels = {cls: i for i, cls in enumerate(classes)}
IMAGE_SIZE = (64, 64)

# Federated learning configuration
federatedLearningcounts = 10
local_client_epochs = 5
local_client_batch_size = 32

def main():
    model = GAN_net()
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    strategy = fl.server.strategy.FedYogi(
        fraction_fit=0.3,
        fraction_evaluate=0.2,
        min_fit_clients=2,
        min_evaluate_clients=2,
        min_available_clients=2,
        evaluate_fn=get_evaluate_fn(model),
        on_fit_config_fn=fit_config,
        on_evaluate_config_fn=evaluate_config,
        initial_parameters=fl.common.ndarrays_to_parameters(model.get_weights()),
    )
    fl.server.start_server(
        server_address=server_address,
        config=fl.server.ServerConfig(num_rounds=federatedLearningcounts),
        strategy=strategy
    )

def load_dataset():
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
            img_path = os.path.join(os.path.join(path, folder), file)
            image = cv2.imread(img_path)
            if image is None:
                continue
            image = cv2.resize(image, IMAGE_SIZE)
            images.append(image)
            labels.append(label)
    images = np.array(images, dtype='float32') / 255.0  # Normalize to [0, 1]
    labels = np.array(labels, dtype='int32')
    return images, labels

def get_evaluate_fn(model):
    test_images, test_labels = load_dataset()
    print("[Server] Test images shape:", test_images.shape)
    print("[Server] Test labels shape:", test_labels.shape)
    def evaluate(
        server_round: int,
        parameters: fl.common.NDArrays,
        config: Dict[str, fl.common.Scalar]
    ) -> Optional[Tuple[float, Dict[str, fl.common.Scalar]]]:
        print(f"=== Server round {server_round}/{federatedLearningcounts} ===")
        model.set_weights(parameters)
        loss, accuracy = model.evaluate(test_images, test_labels, verbose=0)
        print(f"Round {server_round}: Accuracy = {accuracy}")
        if server_round == federatedLearningcounts:
            os.makedirs('Models', exist_ok=True)
            print("Saving final model...")
            model.save('Models/gan_net.keras')
        return loss, {"accuracy": accuracy}
    return evaluate

def fit_config(server_round: int):
    return {
        "batch_size": local_client_batch_size,
        "local_epochs": local_client_epochs,
    }

def evaluate_config(server_round: int):
    return {"val_steps": 4}

if __name__ == "__main__":
    main()
