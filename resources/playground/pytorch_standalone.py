# install psutil

import pprint
import sys

import psutil
import pandas as pd
from mlflow import MlflowClient
from sklearn.pipeline import Pipeline

from kio.pipeline_operations import ColumnDropper, NormalizeCols, OneHotEncodePd, ReplaceNoneValues, \
    NumericOneHotEncodePd

from kio.dataset import example_dataset

import torch
from torch import nn, optim
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import mlflow

MLFLOW_TRACKING_URI = "http://127.0.0.1:8080"
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)




df = pd.DataFrame(example_dataset)

preprocessor = Pipeline(
        verbose=False,
        steps=[
            (
                "Drop irrelevant columns",
                ColumnDropper([
                    "CAD", "Verpackungstyp", "Masse", "Geometrie", "Kopfraumatmosphäre", "Flächenanteilmodifiziert",
                    # dropping also columns for simplicity. These could be normalized instead
                    "RauheitRz", "Trübung", "Glanz", "Dicke", "MaximaleZugspannung", "MaximaleLängenänderung", "RauheitRa",
                    # dropping also 'Kaltverfo'. This field could be one-hot encoded instead
                    'Kaltverfo',
                    # dropping 'Ausformung' which could be one-hot encoded instead
                    "Ausformung"
                ])
            ),
            (
                "Fill missing 'Massenanteile'",
                ReplaceNoneValues(target='Massenanteile', replacement_value=[1.0])
            ),
            (
                "Encode 'ListeKomponenten' and 'Massenanteile' in an one-hot format",
                NumericOneHotEncodePd(
                    targets=['ListeKomponenten', 'Massenanteile'],
                    prefix="M",
                    sep="_",
                    required_columns=["M_K000055", "M_K000057", "M_K000034", "M_K000035", "M_K000141"]
                )
            ),
            (
                "Normalise Emodul",
                NormalizeCols(
                    target="Emodul",
                    column_range=(775.2626646454261, 923.5297844703941),
                    feature_range=(0, 1)
                )
            ),
            ("Normalise Temp", NormalizeCols(
                target="Temp",
                column_range=(0, 500),
                feature_range=(0, 1)
            )),
            ("Normalise Druck", NormalizeCols(
                target="Druck",
                column_range=(0, 6),
                feature_range=(0, 1)
            )),
            ("Normalise Zeit", NormalizeCols(
                target="Zeit",
                column_range=(0, 40),
                feature_range=(0, 1)
            ))
        ]
    )

class DemonstratorNeuralNet(nn.Module):
    """
    A simple neural network for demonstration purposes.

    Attributes
    ----------
    layer_1 : torch.nn.Linear
        The first linear layer.
    layer_2 : torch.nn.Linear
        The second linear layer.
    layer_3 : torch.nn.Linear
        The third linear layer.

    Methods
    -------
    forward(x)
        Forward pass through the network.
    """

    def __init__(self, input_dim, hidden_dim, output_dim, *args, **kwargs):
        """
        Initialize the network.

        Parameters
        ----------
        input_dim
        hidden_dim
        output_dim
        args
        kwargs
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

        Parameters
        ----------
        x
            The input to the network.

        Returns
        -------
        torch.Tensor
            The output of the network.
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

    Attributes
    ----------
    df : pd.DataFrame
        The dataframe containing the data.
    prediction_columns : list
        The columns to be used as prediction targets.

    Methods
    -------
    __len__()
        Return the length of the dataset.
    __getitem__(idx)
        Get an item from the dataset.
    """

    def __init__(self, df: pd.DataFrame, prediction_columns: list):
        """
        Initialize the dataset.

        Parameters
        ----------
        df : pd.DataFrame
            The dataframe containing the data.
        prediction_columns : list
            The columns to be used as prediction targets.
        """
        self.df = df
        self.prediction_columns = prediction_columns

    def __len__(self):
        """
        Return the length of the dataset.

        Returns
        -------
        int
            The length of the dataset.
        """
        return len(self.df)

    def __getitem__(self, idx):
        """
        Get an item from the dataset.

        Parameters
        ----------
        idx : int
            The index of the item.

        Returns
        -------
        tuple
            The input and output data.
        """

        temp = self.df.iloc[idx].copy()
        y_data = temp[self.prediction_columns].to_numpy()
        # Remove the prediction columns from the data
        # somehow drop does not work properly on a single datapoint
        for col in self.prediction_columns:
            temp.pop(col)
        x_data = temp.to_numpy()

        return x_data, y_data

def train_model(df, num_epochs=100, batch_size=32, hidden_dim=288, learning_rate=0.001, **kwargs):
    """
    Train a PyTorch model.

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe containing the data.
    num_epochs : int, optional
        The number of epochs to train for (default is 25).
    batch_size : int, optional
        The batch size (default is 32).

    Returns
    -------
    None
    """

    input_dim = 6 # Emodul, M_K000055,  M_K000057,  M_K000034,  M_K000035,  M_K000141
    output_dim = 3 # Temp, Zeit, Druck
    hidden_dim = 64

    default_params = {
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "num_epochs": num_epochs,
        "hidden_dim": hidden_dim
    }
    params = {**default_params, **kwargs}

    model = DemonstratorNeuralNet(input_dim, hidden_dim, output_dim)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=params["learning_rate"])

    # Convert integer columns to float64 to handle missing values
    df = df.astype({col: 'float64' for col in df.select_dtypes(include='int').columns})

    dataset = MyDataset(df, ["Temp", "Zeit", "Druck"])
    dataloader = DataLoader(dataset, batch_size=params["batch_size"], shuffle=True)



    with mlflow.start_run(
        description="Demonstration PyTorch model for KioModel",
        tags={"model_type": "pytorch", "project": "KioModel Demo"},
        log_system_metrics=True # requires psutil package. This will log CPU and memory usage automatically for long runs (10s interval)
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

        # Log a dictionary as an artifact
        raw_datapoint = {  # FolieVP28
            "ListeKomponenten": ["K000055"],  # id or material name
            "Massenanteile": None,  # unit g/g
            "Flächenanteilmodifiziert": 0,  # unit %
            "Geometrie": "Quader",  # unit: list of types
            "Kopfraumatmosphäre": None,  # unit list of (pa)
            "Masse": None,  # unit g
            "Verpackungstyp": "Folie",  # type
            "CAD": None,  # link to CAD file
            "RauheitRa": 0.729,  # unit µm
            "RauheitRz": 3.33,  # unit µm
            "Trübung": 450.7,  # unit HLog
            "Glanz": 46.9,  # unit GE
            "Dicke": 777,  # unit µm
            "Emodul": 923.5297844703941,  # unit MPa
            "MaximaleZugspannung": 39.27389962516748,  # unit MPa
            "MaximaleLängenänderung": 24.74862718628088,  # unit %
            # Qulaity Labels
            "Ausformung": 1,
            "Kaltverfo": 3,
            # Training Label
            "Temp": 300,
            "Zeit": 8,
            "Druck": 1,
        },

        raw_prediction_payload = {
            "ListeKomponenten": ["K000055"],  # id or material name
            "Massenanteile": None,  # unit g/g
            "Flächenanteilmodifiziert": 0,  # unit %
            "Geometrie": "Quader",  # unit: list of types
            "Kopfraumatmosphäre": None,  # unit list of (pa)
            "Masse": None,  # unit g
            "Verpackungstyp": "Folie",  # type
            "CAD": None,  # link to CAD file
            "RauheitRa": 0.729,  # unit µm
            "RauheitRz": 3.33,  # unit µm
            "Trübung": 450.7,  # unit HLog
            "Glanz": 46.9,  # unit GE
            "Dicke": 777,  # unit µm
            "Emodul": 923.5297844703941,  # unit MPa
            "MaximaleZugspannung": 39.27389962516748,  # unit MPa
            "MaximaleLängenänderung": 24.74862718628088,  # unit %
            # Qulaity Labels
            "Ausformung": 1,
            "Kaltverfo": 3,
        }

        #mlflow.log_dict(raw_datapoint, "raw_label_datapoint.json")
        #mlflow.log_dict(raw_prediction_payload, "raw_prediction_payload.json")

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
    df = preprocessor.fit_transform(df)
    print(df.head())

    mlflow.set_experiment("fiot-pytorch-training")

    # Train the model
    train_model(df, num_epochs=100, batch_size=32)

    sys.exit(0)

    # Load the model from the MLflow repository
    model_uri = "models:/KioModel/1"
    model = mlflow.pytorch.load_model(model_uri)

    input_data = {
        "ListeKomponenten": ["K000055", "K000057"],  # id or material name
        "Massenanteile": [0.75, 0.25],  # unit g/g
        "Flächenanteilmodifiziert": 0,  # unit %
        "Geometrie": "Quader",  # unit: list of types
        "Kopfraumatmosphäre": None,  # unit list of (pa)
        "Masse": None,  # unit g
        "Verpackungstyp": "Folie",  # type
        "CAD": None,  # link to CAD file
        "RauheitRa": 0.08966666666666667,  # unit µm
        "RauheitRz": 0.7366666666666667,  # unit µm
        "Trübung": 176.6,  # unit HLog
        "Glanz": 39,  # unit GE
        "Dicke": 769.6666666666666,  # unit µm
        "Emodul": 878.7979886112262,  # unit MPa
        "MaximaleZugspannung": 37.156951742990245,  # unit MPa
        "MaximaleLängenänderung": 19.73276680651324,  # unit %
        # Quality Labels
        "Ausformung": 6,
        "Kaltverfo": 3,
        # Training Label
        "Temp": 0,  # Note these are included here pass the data through the pipeline and removed afterwards
        "Zeit": 0,
        "Druck": 0,
    }

    # Convert the input data to a DataFrame
    input_df = pd.DataFrame([input_data])
    # Preprocess the input data
    preprocessed_input = preprocessor.transform(input_df)
    # drop the prediction columns
    for col in ["Temp", "Zeit", "Druck"]:
        preprocessed_input.pop(col)

    # Convert the preprocessed input data to a PyTorch tensor
    input_tensor = torch.tensor(preprocessed_input.to_numpy()).float()

    # Perform a forward pass through the model
    with torch.no_grad():
        output_tensor = model(input_tensor)
        # to pandas df with column names Temp, Zeit, Druck
        output_df = pd.DataFrame(output_tensor.numpy(), columns=["Temp", "Zeit", "Druck"])
        print(output_df.head())
        # rescale
        # scale temp by 500
        output_df["Temp"] = output_df["Temp"] * 500
        # scale Zeit by 40
        output_df["Zeit"] = output_df["Zeit"] * 40
        # scale Druck by 6
        output_df["Druck"] = output_df["Druck"] * 6
        print(output_df.head())


