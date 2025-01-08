from typing import List, Dict, Tuple, Optional
from flwr.common import Parameters, Scalar, ndarrays_to_parameters, parameters_to_ndarrays
from flwr.server.strategy import Strategy
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.common import FitIns, FitRes


class FedCyclic(Strategy):
    def __init__(
        self,
        num_rounds: int,
        num_local_epochs: int,
        learning_rate: float,
        batch_size: int,
        num_clients: int,
    ) -> None:
        """Initialize Fed-Cyclic strategy."""
        super().__init__()
        self.num_rounds = num_rounds
        self.num_local_epochs = num_local_epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_clients = num_clients
        self.global_weights = None  # Initialize global weights

    def initialize_parameters(self, client_manager: ClientManager) -> Optional[Parameters]:
        """Initialize global weights."""
        self.global_weights = self._initialize_global_weights()
        return ndarrays_to_parameters(self.global_weights)

    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """Configure the training for one round."""
        if self.global_weights is None:
            self.global_weights = parameters_to_ndarrays(parameters)

        # Identify the client index for this round (cyclic order)
        client_idx = (server_round - 1) % self.num_clients

        # Sample the specific client for this round
        sampled_clients = client_manager.sample(
            num_clients=1, min_num_clients=1, ids=[str(client_idx)]
        )

        # Fit configuration for the client
        fit_config = {
            "epochs": self.num_local_epochs,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
        }

        # Assign the global weights to the sampled client
        return [(client, FitIns(ndarrays_to_parameters(self.global_weights), fit_config))
                for client in sampled_clients]

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Tuple[ClientProxy, FitRes] or BaseException], # type: ignore
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate updates from one client and update global weights."""
        if not results:
            return None, {}

        # Extract the updated weights from the single client
        updated_weights = parameters_to_ndarrays(results[0][1].parameters)

        # Update global weights (cyclically)
        self.global_weights = updated_weights

        return ndarrays_to_parameters(self.global_weights), {}

    def evaluate(
        self, server_round: int, parameters: Parameters
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        """Global evaluation (not used in this case)."""
        return None

    def _initialize_global_weights(self):
        """Initialize global weights manually or randomly."""
        import numpy as np
        # Example: Replace this with initialization logic as per your model
        return [np.random.randn(10, 10), np.zeros(10)]


# Helper function to configure the client updates
def client_update(w: List, D_k: List, b: int, E: int, eta: float) -> List:
    """Perform client update as described in the algorithm."""
    for _ in range(E):  # For each local epoch
        for d in D_k:  # Split D_k into batches of size b
            gradient = compute_gradient(w, d)
            w = [w_i - eta * g_i for w_i, g_i in zip(w, gradient)]
    return w


def compute_gradient(w: List, batch: List) -> List:
    """Placeholder for gradient computation (replace with your logic)."""
    import numpy as np
    # Example: Random gradient generation for illustration
    return [np.random.randn(*w_i.shape) for w_i in w]