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
        PyTorch регрессор с возможностью задавать архитектуру сети.
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
        Инициализация модели с последовательными слоями и явной инициализацией весов.
        """
        layers = []
        input_dim = self.input_size

        for hidden_size in self.hidden_sizes:
            layer = nn.Linear(input_dim, hidden_size)
            nn.init.kaiming_normal_(layer.weight)
            nn.init.constant_(layer.bias, 0)
            layers.append(layer)
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=self.dropout_rate))
            input_dim = hidden_size

        output_layer = nn.Linear(input_dim, 1)
        nn.init.kaiming_normal_(output_layer.weight)
        nn.init.constant_(output_layer.bias, 0)
        layers.append(output_layer)

        self.model = nn.Sequential(*layers)
        self.model.to(self.device)

        if self.optimizer_type == "adam":
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        elif self.optimizer_type == "sgd":
            self.optimizer = optim.SGD(self.model.parameters(), lr=self.lr)
        else:
            raise ValueError(f"Unsupported optimizer type: {self.optimizer_type}")

        if self.criterion_type == "mse":
            self.criterion = nn.MSELoss()
        elif self.criterion_type == "mae":
            self.criterion = nn.L1Loss()
        else:
            raise ValueError(f"Unsupported criterion type: {self.criterion_type}")

    def fit(self, X_train, y_train, X_val, y_val):
        """
        Обучает модель на данных.
        """
        if isinstance(X_train, pd.DataFrame):
            X_train = X_train.to_numpy()
        if isinstance(y_train, pd.Series):
            y_train = y_train.to_numpy()
        if isinstance(X_val, pd.DataFrame):
            X_val = X_val.to_numpy()
        if isinstance(y_val, pd.Series):
            y_val = y_val.to_numpy()

        X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(self.device)
        y_train_tensor = (
            torch.tensor(y_train, dtype=torch.float32).view(-1, 1).to(self.device)
        )
        X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(self.device)
        y_val_tensor = (
            torch.tensor(y_val, dtype=torch.float32).view(-1, 1).to(self.device)
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
                    print(f"NaN в предсказаниях на эпохе {epoch}")

                loss = self.criterion(outputs, targets)

                if torch.isnan(loss).any():
                    print(f"NaN в потере на эпохе {epoch}")

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                self.optimizer.step()
                running_train_loss += loss.item()

            self.model.eval()
            with torch.no_grad():
                val_outputs = self.model(X_val_tensor)
                val_loss = self.criterion(val_outputs, y_val_tensor).item()

                if torch.isnan(val_outputs).any():
                    print(f"NaN в валидационных предсказаниях на эпохе {epoch}")

            train_loss = running_train_loss / len(train_loader)
            self.train_loss_history.append(train_loss)
            self.val_loss_history.append(val_loss)

            # if epoch % 10 == 0:
            #     print(
            #         f"Epoch {epoch+1}/{self.epochs}, Training Loss: {train_loss}, Validation Loss: {val_loss}"
            #     )

            self.model.train()

    def predict(self, X):
        """
        Прогнозирование на новых данных.
        """
        if isinstance(X, pd.DataFrame):
            X = X.to_numpy()

        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
            predictions = self.model(X_tensor).cpu().numpy()

        return predictions
