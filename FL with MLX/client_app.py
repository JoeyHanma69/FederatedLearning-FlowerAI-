# client_app.py
import warnings
import torch

import flwr as fl
from flwr.client import NumPyClient

from transformers import logging
from huggingface_example.task import (
    train,
    test,
    load_data,
    set_params,
    get_params,
    get_model,
)

warnings.filterwarnings("ignore", category=FutureWarning)
logging.set_verbosity_error()

# Replace this with your own settings or command-line parameters
PARTITION_ID = 0
NUM_PARTITIONS = 1
MODEL_NAME = "distilbert-base-uncased"

class IMDBClient(NumPyClient):
    def __init__(self, model_name, partition_id, num_partitions) -> None:
        super().__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.trainloader, self.testloader = load_data(partition_id, num_partitions, model_name)
        self.net = get_model(model_name)
        self.net.to(self.device)

    def fit(self, parameters, config):
        # Load server-sent model parameters
        set_params(self.net, parameters)
        # Train
        train(self.net, self.trainloader, epochs=1, device=self.device)
        # Return updated parameters
        return get_params(self.net), len(self.trainloader), {}

    def evaluate(self, parameters, config):
        # Load server-sent model parameters
        set_params(self.net, parameters)
        # Evaluate
        loss, accuracy = test(self.net, self.testloader, device=self.device)
        return float(loss), len(self.testloader), {"accuracy": float(accuracy)}

def main() -> None:
    # Create the client
    client = IMDBClient(model_name=MODEL_NAME,
                        partition_id=PARTITION_ID,
                        num_partitions=NUM_PARTITIONS)
    # Start the client and connect to the server
    fl.client.start_numpy_client(
        server_address="127.0.0.1:8080",
        client=client,
    )

if __name__ == "__main__":
    main()
