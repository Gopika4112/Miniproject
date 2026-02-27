import flwr as fl
from config import NUM_ROUNDS, LOCAL_EPOCHS, BATCH_SIZE, SERVER_ADDRESS, NUM_HOSPITALS

def fit_config(server_round: int):
    return {
        "local_epochs": LOCAL_EPOCHS,
        "batch_size": BATCH_SIZE,
    }

def main():
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=NUM_HOSPITALS,
        min_evaluate_clients=NUM_HOSPITALS,
        min_available_clients=NUM_HOSPITALS,
        on_fit_config_fn=fit_config,
    )

    fl.server.start_server(
        server_address=SERVER_ADDRESS,
        config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
        strategy=strategy,
    )

if __name__ == "__main__":
    main()