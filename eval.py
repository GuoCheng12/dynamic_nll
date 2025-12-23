import hydra
import torch
from omegaconf import DictConfig

from src.data import build_dataloaders
from src.modules import GaussianLogLikelihoodLoss
from src.models import MLPRegressor
from src.utils import expected_calibration_error, mae, nll, rmse, set_seed


@hydra.main(config_path="configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, _, test_loader = build_dataloaders(cfg.hyperparameters.batch_size, seed=cfg.seed)

    model = MLPRegressor(
        input_dim=cfg.model.get("input_dim", 1),
        hidden_sizes=cfg.model.hidden_sizes,
        activation=cfg.model.activation,
        dropout=cfg.model.get("dropout", 0.0),
    ).to(device)

    if cfg.get("checkpoint"):
        state = torch.load(cfg.checkpoint, map_location=device)
        model.load_state_dict(state["model"])

    criterion = GaussianLogLikelihoodLoss()
    model.eval()
    metrics = {"mse": 0.0, "mae": 0.0, "rmse": 0.0, "nll": 0.0, "ece": 0.0}
    count = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            mean, variance = model(data)
            loss = criterion(mean, target, variance=variance, interpolate=False)
            batch_size = len(data)
            metrics["mse"] += torch.mean((mean - target) ** 2).item() * batch_size
            metrics["mae"] += mae(mean, target) * batch_size
            metrics["rmse"] += rmse(mean, target) * batch_size
            metrics["nll"] += nll(mean, target, variance) * batch_size
            metrics["ece"] += expected_calibration_error(mean, variance, target) * batch_size
            count += batch_size
            print(f"Loss: {loss.item():.4f} beta={criterion.beta:.4f}")

    for k in metrics:
        metrics[k] /= max(count, 1)
    print("Test metrics:", metrics)


if __name__ == "__main__":
    main()
