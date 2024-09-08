from sklearn.base import BaseEstimator, RegressorMixin
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd


class PyTorchRegressor(BaseEstimator, RegressorMixin):
    def __init__(
        self,
        input_size,
        hidden_sizes=[256, 128, 64],
        lr=0.001,
        batch_size=32,
        epochs=100,
        device="cpu",
        optimizer_type="adam",
        criterion_type="mse",
        dropout_rate=0.5,
    ):
        """
        PyTorch regressor with customizable network architecture.
        """
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs
        self.device = device
        self.optimizer_type = optimizer_type
        self.criterion_type = criterion_type
        self.dropout_rate = dropout_rate
        self.model = None
        self.optimizer = None
        self.criterion = None
        self.train_loss_history = []
        self.val_loss_history = []

        if len(hidden_sizes) == 0:
            raise ValueError(
                "hidden_sizes must contain at least one hidden layer configuration."
            )

    def _initialize_model(self):
        """
        Initializes the model with sequential layers and explicit weight initialization.
        """
        layers = []
        input_dim = self.input_size

        # Создаем слои, чередуя ReLU и Tanh
        for i, hidden_size in enumerate(self.hidden_sizes):
            layer = nn.Linear(input_dim, hidden_size)
            nn.init.kaiming_normal_(layer.weight)
            nn.init.constant_(layer.bias, 0)
            layers.append(layer)

            # Чередование ReLU и Tanh
            if i % 2 == 0:
                layers.append(nn.ReLU())
            else:
                layers.append(nn.Tanh())

            layers.append(nn.Dropout(p=self.dropout_rate))
            input_dim = hidden_size

        # Выходной слой
        output_layer = nn.Linear(input_dim, 1)
        nn.init.kaiming_normal_(output_layer.weight)
        nn.init.constant_(output_layer.bias, 0)
        layers.append(output_layer)

        self.model = nn.Sequential(*layers)
        self.model.to(self.device)

        # Настройка оптимизатора
        if self.optimizer_type == "adam":
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        elif self.optimizer_type == "sgd":
            self.optimizer = optim.SGD(self.model.parameters(), lr=self.lr)
        else:
            raise ValueError(f"Unsupported optimizer type: {self.optimizer_type}")

        # Настройка функции потерь
        if self.criterion_type == "mse":
            self.criterion = nn.MSELoss()
        elif self.criterion_type == "mae":
            self.criterion = nn.L1Loss()
        else:
            raise ValueError(f"Unsupported criterion type: {self.criterion_type}")

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        """
        Trains the model on the data. Validation data is optional.

        Parameters
        ----------
        X_train : np.ndarray or pd.DataFrame
            Training data.
        y_train : np.ndarray or pd.Series
            Target values for the training data.
        X_val : np.ndarray or pd.DataFrame, optional
            Validation data. If not provided, training proceeds without validation.
        y_val : np.ndarray or pd.Series, optional
            Target values for the validation data. If not provided, training proceeds without validation.
        """

        X_train_tensor = torch.FloatTensor(X_train).to(self.device)
        y_train_tensor = (
            torch.FloatTensor(y_train.values).reshape(-1, 1).to(self.device)
        )

        if X_val is not None and y_val is not None:
            X_val_tensor = torch.FloatTensor(X_val).to(self.device)
            y_val_tensor = (
                torch.FloatTensor(y_val.values).reshape(-1, 1).to(self.device)
            )

        train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True
        )

        self._initialize_model()
        self.model.train()

        for epoch in range(self.epochs):
            running_train_loss = 0.0
            for inputs, targets in train_loader:
                self.optimizer.zero_grad()
                outputs = self.model(inputs)

                if torch.isnan(outputs).any():
                    print(f"NaN in predictions during epoch {epoch}")

                loss = self.criterion(outputs, targets)

                if torch.isnan(loss).any():
                    print(f"NaN in loss during epoch {epoch}")

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                running_train_loss += loss.item()

            # Calculate and save training loss
            train_loss = running_train_loss / len(train_loader)
            self.train_loss_history.append(train_loss)

            # If validation data is provided, calculate validation loss
            if X_val is not None and y_val is not None:
                self.model.eval()
                with torch.no_grad():
                    val_outputs = self.model(X_val_tensor)
                    val_loss = self.criterion(val_outputs, y_val_tensor).item()

                    if torch.isnan(val_outputs).any():
                        print(f"NaN in validation predictions during epoch {epoch}")

                    self.val_loss_history.append(val_loss)

                if epoch % 10 == 0:
                    print(
                        f"Epoch {epoch+1}/{self.epochs}, Training Loss: {train_loss}, Validation Loss: {val_loss}"
                    )
                self.model.train()
            else:
                print(f"Epoch {epoch+1}/{self.epochs}, Training Loss: {train_loss}")

    def predict(self, X):
        """
        Predicts on new data.
        """
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            predictions = self.model(X_tensor).cpu().numpy()

        return predictions
