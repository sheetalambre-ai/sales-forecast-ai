"""
LSTM forecasting model wrapper with fallback option.
"""

import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

from models.base import BaseForecastModel
from models.utils import save_model, load_model

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


class LSTMModel(BaseForecastModel):
    """
    LSTM/Neural Network forecasting model.
    Falls back to sklearn's MLPRegressor if PyTorch is not available.
    """

    def __init__(self, epochs: int = 5, batch_size: int = 64, lr: float = 0.001, **kwargs):
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.kwargs = kwargs
        self.is_fallback = not HAS_TORCH
        self.model = None
        self.scaler_x = StandardScaler()
        self.scaler_y = StandardScaler()

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
    ) -> None:
        """
        Train the model.
        """
        # Fill missing values if any
        X_scaled = self.scaler_x.fit_transform(X_train.fillna(0))
        y_scaled = self.scaler_y.fit_transform(y_train.values.reshape(-1, 1)).ravel()

        if not self.is_fallback:
            # PyTorch implementation
            input_dim = X_train.shape[1]
            # Simple PyTorch MLP/RNN
            self.model = PyTorchNNRegressor(input_dim=input_dim, lr=self.lr)
            self.model.fit(X_scaled, y_scaled, epochs=self.epochs, batch_size=self.batch_size)
        else:
            # Fallback to MLPRegressor
            self.model = MLPRegressor(
                hidden_layer_sizes=(64, 32),
                activation="relu",
                solver="adam",
                max_iter=50,
                random_state=42,
                **self.kwargs
            )
            self.model.fit(X_scaled, y_scaled)

    def predict(
        self,
        X_test: pd.DataFrame,
    ) -> np.ndarray:
        """
        Generate predictions.
        """
        X_scaled = self.scaler_x.transform(X_test.fillna(0))

        if not self.is_fallback:
            preds_scaled = self.model.predict(X_scaled)
        else:
            preds_scaled = self.model.predict(X_scaled)

        # Inverse transform target values
        preds = self.scaler_y.inverse_transform(preds_scaled.reshape(-1, 1)).ravel()
        return preds

    def save(self, filepath: str) -> None:
        """
        Save the model.
        """
        save_model(
            {
                "model": self.model,
                "is_fallback": self.is_fallback,
                "scaler_x": self.scaler_x,
                "scaler_y": self.scaler_y,
                "epochs": self.epochs,
                "batch_size": self.batch_size,
                "lr": self.lr,
                "kwargs": self.kwargs
            },
            filepath
        )

    def load(self, filepath: str) -> None:
        """
        Load the model.
        """
        data = load_model(filepath)
        self.model = data["model"]
        self.is_fallback = data["is_fallback"]
        self.scaler_x = data["scaler_x"]
        self.scaler_y = data["scaler_y"]
        self.epochs = data["epochs"]
        self.batch_size = data["batch_size"]
        self.lr = data["lr"]
        self.kwargs = data["kwargs"]


if HAS_TORCH:
    class PyTorchNNModel(nn.Module):
        """
        Simple PyTorch Feedforward neural network for regression.
        Can act as a neural network baseline.
        """
        def __init__(self, input_dim: int):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, 1)
            )

        def forward(self, x):
            return self.net(x)

    class PyTorchNNRegressor:
        """
        Scikit-learn wrapper for our PyTorch neural network.
        """
        def __init__(self, input_dim: int, lr: float = 0.001):
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model = PyTorchNNModel(input_dim).to(self.device)
            self.criterion = nn.MSELoss()
            self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

        def fit(self, X: np.ndarray, y: np.ndarray, epochs: int, batch_size: int):
            self.model.train()
            X_tensor = torch.tensor(X, dtype=torch.float32)
            y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
            dataset = TensorDataset(X_tensor, y_tensor)
            loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

            for epoch in range(epochs):
                for batch_x, batch_y in loader:
                    batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                    self.optimizer.zero_grad()
                    output = self.model(batch_x)
                    loss = self.criterion(output, batch_y)
                    loss.backward()
                    self.optimizer.step()

        def predict(self, X: np.ndarray) -> np.ndarray:
            self.model.eval()
            X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
            with torch.no_grad():
                preds = self.model(X_tensor)
            return preds.cpu().numpy().ravel()
else:
    # Set PyTorch placeholders
    PyTorchNNModel = None
    PyTorchNNRegressor = None
