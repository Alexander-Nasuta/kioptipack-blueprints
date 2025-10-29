import asyncio
import logging
import torch
import mlflow
import pandas as pd
import psutil

from mlflow import MlflowClient

from fastiot.core import FastIoTService, loop

from torch import nn, optim
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from kio.banner import KIOptiPack_banner

from kio.ml_lifecycle_broker_facade import request_get_all_processed_data_points


class MyNeuralNet(nn.Module):
    """
    A simple neural network for demonstration purposes.

    :ivar layer_1: The first linear layer.
    :type layer_1: torch.nn.Linear
    :ivar layer_2: The second linear layer.
    :type layer_2: torch.nn.Linear
    :ivar layer_3: The third linear layer.
    :type layer_3: torch.nn.Linear

    :meth forward(x): Forward pass through the network.
    """

    def __init__(self, input_dim, hidden_dim, output_dim, *args, **kwargs):
        """
        Initialize the network.

        :param input_dim: Input dimension of the network.
        :type input_dim: int
        :param hidden_dim: Hidden layer dimension.
        :type hidden_dim: int
        :param output_dim: Output dimension of the network.
        :type output_dim: int
        :param args: Zusätzliche Positionsargumente.
        :type args: tuple
        :param kwargs: Zusätzliche Schlüsselwortargumente.
        :type kwargs: dict
        """
        super().__init__(*args, **kwargs)
        self.layer_1 = nn.Linear(input_dim, hidden_dim)
        self.layer_2 = nn.Linear(hidden_dim, hidden_dim)
        self.layer_3 = nn.Linear(hidden_dim, hidden_dim)
        self.layer_4 = nn.Linear(hidden_dim, hidden_dim)
        self.layer_5 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        """
        Forward pass through the network.

        :param x: The input to the network.
        :type x: torch.Tensor
        :return: The output of the network.
        :rtype: torch.Tensor
        """
        x = torch.relu(self.layer_1(x))
        x = torch.relu(self.layer_2(x))
        x = torch.relu(self.layer_3(x))
        x = torch.relu(self.layer_4(x))
        x = self.layer_5(x)
        return x


class MyDataset(Dataset):
    """
    A custom dataset for the pytorch regression service.

    :param df: The dataframe containing the data.
    :type df: pd.DataFrame

    :param prediction_columns: The columns to be used as prediction targets.
    :type prediction_columns: list

    """

    def __init__(self, df: pd.DataFrame, prediction_columns: list):
        """
        Initialize the dataset.
        """
        self.df = df
        self.prediction_columns = prediction_columns

    def __len__(self):
        """
        Return the length of the dataset.

        :return: The length of the dataset.
        :rtype: int

        """
        return len(self.df)

    def __getitem__(self, idx):
        """
        Get an item from the dataset.

        :param idx: The index of the item to get.
        :type idx: int

        :return: The input and output data.
        :rtype: tuple

        """

        temp = self.df.iloc[idx].copy()
        y_data = temp[self.prediction_columns].to_numpy()
        # Remove the prediction columns from the data
        # somehow drop does not work properly on a single datapoint
        for col in self.prediction_columns:
            temp.pop(col)
        x_data = temp.to_numpy()

        return x_data, y_data


class PytorchTrainingService(FastIoTService):

    MLFLOW_TRACKING_URI = "http://127.0.0.1:8080" # Adjust to your MLflow tracking server URI if you do not use the default one
    MLFLOW_EXPERIMENT_NAME = "fiot-pytorch-training" # Name of the MLflow experiment
    TRAINING_INTERVAL = 60 * 15 # 10 minutes # Interval between training runs in seconds


    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        mlflow.set_tracking_uri(self.MLFLOW_TRACKING_URI)
        mlflow.set_experiment(self.MLFLOW_EXPERIMENT_NAME)

    async def _start(self):
        print(KIOptiPack_banner)  # Display the KIOptiPack banner on service start
        self._logger.info("MotivSensorService started.")

    @loop
    async def produce(self):

        await self.train_model()

        return asyncio.sleep(self.TRAINING_INTERVAL)

    async def train_model(self, **kwargs):
        self._logger.info("Training model.")
        self._logger.info("Fetching labeled dataset.")
        labeled_dataset = await request_get_all_processed_data_points(fiot_service=self)
        df = pd.DataFrame(labeled_dataset)

        input_dim = 6  # Emodul, M_K000055,  M_K000057,  M_K000034,  M_K000035,  M_K000141
        output_dim = 3  # Temp, Zeit, Druck

        default_params = {
            "learning_rate": 0.001,
            "batch_size": 32,
            "num_epochs": 100,
            "input_dim": input_dim,
            "output_dim": output_dim,
            "hidden_dim": 64
        }
        params = {**default_params, **kwargs}

        model = MyNeuralNet(params["input_dim"], params["hidden_dim"], params["output_dim"])
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=params["learning_rate"])

        # Convert integer columns to float64 to handle missing values
        df = df.astype({col: 'float64' for col in df.select_dtypes(include='int').columns})

        dataset = MyDataset(df, ["Temp", "Zeit", "Druck"])
        dataloader = DataLoader(dataset, batch_size=params["batch_size"], shuffle=True)


        with mlflow.start_run(
                description="Demonstration PyTorch model for KioModel",
                tags={"model_type": "pytorch", "project": "KioModel Demo"},
                log_system_metrics=True
                # requires psutil package. This will log CPU and memory usage automatically for long runs (10s interval)
        ) as run:
            # Log parameters
            for key, value in params.items():
                mlflow.log_param(key, value)

            # Save dataset
            df.to_csv("dataset.csv", index=False)
            mlflow.log_artifact("dataset.csv", artifact_path="data")

            # Log dataset to MLflow (shows up in Dataset column)
            dataset = mlflow.data.from_pandas(df, source="dataset.csv", name="kio-example-dataset")
            mlflow.log_input(dataset)

            for epoch in range(params["num_epochs"]):
                running_loss = 0.0
                for inputs, targets in dataloader:
                    inputs = inputs.float()
                    targets = targets.float()

                    optimizer.zero_grad()
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    loss.backward()
                    optimizer.step()
                    running_loss += loss.item()

                epoch_loss = running_loss / len(dataloader)
                print(f"Epoch {epoch + 1}/{params['num_epochs']}, Loss: {epoch_loss}")

                # Log metrics
                mlflow.log_metric("training-loss", float(epoch_loss), step=epoch)

                # Log Resource Metrics
                mlflow.log_metric("system.cpu.percent", psutil.cpu_percent(), step=epoch)
                mlflow.log_metric("system.memory.percent", psutil.virtual_memory().percent, step=epoch)

            # Log model with input example
            input_example = torch.randn(1, input_dim).float().numpy()
            # get an example input form the dataset
            mlflow.pytorch.log_model(model, name="KioModel", input_example=input_example)

            model_uri = f"runs:/{run.info.run_id}/KioModel"
            model_details = mlflow.register_model(model_uri=model_uri, name="KioModel")

            # Create an MLflow client
            client = MlflowClient()

            # Add tags to this specific model version
            client.set_model_version_tag(
                name="KioModel",
                version=model_details.version,
                key="framework",
                value="pytorch"
            )

            client.set_model_version_tag(
                name="KioModel",
                version=model_details.version,
                key="dataset",
                value="kio-example-dataset"
            )



if __name__ == '__main__':
    # Change this to reduce verbosity or remove completely to use `FASTIOT_LOG_LEVEL` environment variable to configure
    # logging.
    logging.basicConfig(level=logging.DEBUG)
    PytorchTrainingService.main()
