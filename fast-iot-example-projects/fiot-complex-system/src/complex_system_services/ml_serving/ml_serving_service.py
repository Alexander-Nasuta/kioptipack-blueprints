import logging
from asyncio import Future

from fastiot.core import FastIoTService, reply
from fastiot.msg.thing import Thing

import torch
import numpy as np
import pandas as pd

from torch import nn, optim

from kio.ml_lifecycle_broker_facade import \
    request_get_processed_data_points_from_raw_data, ok_response_thing, error_response_thing
from kio.ml_lifecycle_subjects_name import ML_SERVING_SUBJECT

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
        self.layer_3 = nn.Linear(hidden_dim, output_dim)

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
        x = self.layer_3(x)
        return x



class MlServingService(FastIoTService):
    """
    This service is responsible for serving predictions using a PyTorch model.

    The model is a simple neural network that takes 15 input features and outputs a single value.

    Attributes
    ----------
    _regression_model : DemonstratorNeuralNet
        The regression model to be used for predictions.
    _example_raw_payload : dict
        An example raw payload to be used for testing the model.

    Methods
    -------
    _start()
        Start the service.
    _setup_model()
        Initialize the regression model.
    _load_model_weights_wandb()
        Load the model weights from wandb.
    _process_raw_data_points(data: list[dict]) -> Future[list[dict]]
        Process raw data points.
    _get_prediction(raw_datapoints: list[dict]) -> Future[list[list[float]]]
        Get predictions for raw data points.
    prediction(topic: str, msg: Thing) -> Thing
        Serve predictions for raw data points.
    """
    _regression_model = None

    _example_raw_payload = {
        'laborant': "TK",
        'material_id': "11111111",
        'datum': "15.03.2024, 16:14:48",
        'rohwert_1_labormessung': 22.64237723051251,
        'rohwert_2_labormessung': 0.55194,
        'rohwert_3_labormessung': -0.472279,
        'aufbereiteter_wert': 0.287696
    }

    async def _start(self):
        """
        Runs when the service starts.
        """
        self._logger.info("MlPytorchRegressionService started")
        await self._setup_model()

    async def _setup_model(self):
        """
        Initialize the regression model.

        Returns
        -------
        None
        """
        self._logger.info("Setting up Demonstrator Regression model")

        self._regression_model = DemonstratorNeuralNet(
            input_dim=15,
            hidden_dim=10,
            output_dim=1
        )

        # Load model weights
        await self._load_model_weights_wandb()

        # test model with example payload
        _ = await self._get_prediction(
            raw_datapoints=[
                self._example_raw_payload,
                self._example_raw_payload,
                self._example_raw_payload,
            ]
        )

    async def _load_model_weights_wandb(self):
        """
        Load the model weights from wandb.

        Returns
        -------
        None
        """
        self._logger.info("Loading model weights from wandb")

        import wandb

        # Create a new API object
        api = wandb.Api()

        # Get all the artifacts of a specific project
        wandb_config = {
            "entity": "querry",  # Replace with your username
            "project": "KIOptipack-dev",  # Replace with your project name
            "name": "model_4c7eb0ae-2dc6-49f5-a179-605a89",  # Replace with your artifact name
            "version": "v6",
            "group": "MVDP-pytorch-regression",
            "model_type": "pytorch-regression-model",
        }
        artifact = api.artifact("querry/KIOptipack-dev/DemonstratorNeuralNet:latest")
        self._logger.info(f"Downloading model weights for model '{artifact.metadata['model_name']}'")
        artifact.download()

        model_name = artifact.metadata["model_name"]
        model_version = artifact.version
        path = f"./artifacts/DemonstratorNeuralNet:{model_version}/{model_name}"
        if self._regression_model is None:
            raise ValueError("Regression model not initialized. Please call _setup_model() first.")
        self._regression_model.load_state_dict(
            torch.load(f"./artifacts/DemonstratorNeuralNet:{model_version}/{model_name}"))

    async def _process_raw_data_points(self, data: list[dict]) -> Future[list[dict]]:
        """
        Process raw data points.

        Parameters
        ----------
        data
            Raw data points.

        Returns
        -------
        Future[list[dict]]
            Processed data points.
        """
        self._logger.info(f"Processing raw data points, received")
        return await request_get_processed_data_points_from_raw_data(
            fiot_service=self,
            data=data,
        )

    async def _get_prediction(self, raw_datapoints: list[dict]) -> Future[list[list[float]]]:
        """
        Get predictions for raw data points.

        Parameters
        ----------
        raw_datapoints
            Raw data points.

        Returns
        -------
        Future[list[list[float]]]
            Predictions for raw data points.
        """
        if self._regression_model is None:
            raise ValueError("Regression model not initialized. Please call _setup_model() first.")

        processed_data = await self._process_raw_data_points(data=raw_datapoints)
        temp = pd.DataFrame(processed_data)
        _ = np.array([temp.pop("aufbereiteter_wert")])
        x_data = temp.to_numpy()

        prediction = self._regression_model(torch.tensor(x_data, dtype=torch.float32))
        return prediction.tolist()

    @reply(ML_SERVING_SUBJECT)
    async def prediction(self,  _: str, msg: Thing) -> Thing:
        """
        Serve predictions for raw data points.

        Parameters
        ----------
        _
        msg
            The message containing the raw data points.
        msg
            The message containing the raw data points.

        Returns
        -------
        Thing
            The response message.
        """
        if not isinstance(msg.value, list):
            self._logger.error(f"Payload (the 'value' field of the msg Thing) must be of type list, "
                      f"but received: {type(msg.value)}")
            raise ValueError("Payload must be a list of raw data points")

        raw_data_points: list[dict] = msg.value

        try:
            res:list[list[float]] = await self._get_prediction(raw_datapoints=raw_data_points)

            return ok_response_thing(payload=res, fiot_service=self)

        except Exception as e:
            self._logger.error(f"Error while processing raw data points: {e}")
            return error_response_thing(exception=e, fiot_service=self)


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    MlServingService.main()
