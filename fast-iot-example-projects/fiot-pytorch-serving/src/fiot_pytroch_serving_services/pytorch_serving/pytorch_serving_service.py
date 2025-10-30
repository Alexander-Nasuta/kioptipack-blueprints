import asyncio
import logging
import random
from typing import Any, Coroutine

import torch
from asyncio import Future

import mlflow

import pandas as pd
import numpy as np

from fastiot.core import FastIoTService, Subject, subscribe, loop, reply
from fastiot.core.core_uuid import get_uuid
from fastiot.core.time import get_time_now
from fastiot.msg.thing import Thing

from kio.banner import KIOptiPack_banner

from kio.ml_lifecycle_broker_facade import request_get_processed_data_points_from_raw_data, ok_response_thing, \
    error_response_thing

from kio.ml_lifecycle_subjects_name import ML_SERVING_SUBJECT


class PytorchServingService(FastIoTService):

    # Example model uri. 1 can be replaced with the desired model version.
    # Note: mlflow considers using the latest version as deprecated.
    MODEL_URI = "models:/KioModel/1"

    MLFLOW_TRACKING_URI = "http://127.0.0.1:8080"  # Adjust to your MLflow tracking server URI if you do not use the default one
    MLFLOW_EXPERIMENT_NAME = "fiot-pytorch-serving"  # Name of the MLflow experiment

    _example_raw_payload = {
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
        # Training Label. Not used for prediction, only for passing through the pipeline
        #"Temp": 0,  # Note these are included here pass the data through the pipeline and removed afterwards
        #"Zeit": 0, # Note these are included here pass the data through the pipeline and removed afterwards
        #"Druck": 0, # Note these are included here pass the data through the pipeline and removed afterwards
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        mlflow.set_tracking_uri(self.MLFLOW_TRACKING_URI)
        mlflow.set_experiment(self.MLFLOW_EXPERIMENT_NAME)

    async def _start(self):
        """
        Runs when the service starts.
        """
        print(KIOptiPack_banner)  # Display the KIOptiPack banner on service start
        self._logger.info("PytorchServingService started")
        await self._setup_model()

    async def _setup_model(self):
        """
        Initialize the regression model.

        :returns: None
        :rtype: None
        """
        self._logger.info("Setting up Demonstrator Regression model")

        # Load model weights
        await self._load_model_weights_mlfow()

        # test model with example payload
        _ = await self._get_prediction(
            raw_datapoints=[
                self._example_raw_payload,
                self._example_raw_payload,
                self._example_raw_payload,
            ]
        )

    async def _load_model_weights_mlfow(self):
        """
        Load the model weights from mlflow.

        :returns: None
        :rtype: None
        """
        self._logger.info("Loading model weights from mlfow")
        model = mlflow.pytorch.load_model(model_uri=self.MODEL_URI)
        self._logger.info(f"Model loaded from mlflow: \n{model}")
        self._regression_model = model

    async def _process_raw_data_points(self, data: list[dict]) -> Future[list[dict]]:
        """
        Process raw data points.

        :param data: Raw data points to be processed.
        :type data: list[dict]
        :returns: Processed data points.
        :rtype: Future[list[dict]]
        """
        self._logger.info(f"Processing raw data points")
        return await request_get_processed_data_points_from_raw_data(
            fiot_service=self,
            data=data,
        )

    async def _get_prediction(self, raw_datapoints: list[dict]) -> list[dict]:
        """
        Get predictions for raw data points.

        :param raw_datapoints: Raw data points to get predictions for.
        :type raw_datapoints: list[dict]
        :returns: Predictions for raw data points.
        :rtype: Future[list[list[float]]]
        """
        if self._regression_model is None:
            raise ValueError("Regression model not initialized. Please call _setup_model() first.")

        self._logger.info(
            f"Integration test: getting a processed data point from Data Processing Service and performing a prediction.")
        # add labels with dummy values to pass the data through the processing pipeline
        for e in raw_datapoints:
            e["Temp"] = 0
            e["Zeit"] = 0
            e["Druck"] = 0

        processed_data = await self._process_raw_data_points(data=raw_datapoints)
        temp = pd.DataFrame(processed_data)

        # drop the prediction columns
        for col in ["Temp", "Zeit", "Druck"]:
            temp.pop(col)

        x_data = temp.to_numpy()

        # Perform a forward pass through the model
        with torch.no_grad():
            prediction = self._regression_model(torch.tensor(x_data, dtype=torch.float32))
            self._logger.info(f"Integration test passed.")

            output_df = pd.DataFrame(prediction.numpy(), columns=["Temp", "Zeit", "Druck"])
            # scale temp by 500
            output_df["Temp"] = output_df["Temp"] * 500
            # scale Zeit by 40
            output_df["Zeit"] = output_df["Zeit"] * 40
            # scale Druck by 6
            output_df["Druck"] = output_df["Druck"] * 6

        return output_df.to_dict(orient="records")

    @reply(ML_SERVING_SUBJECT)
    async def prediction(self, _: str, msg: Thing) -> Thing:
        """
        Serve predictions for raw data points.

        :param _: Nicht verwendet.
        :param msg: Die Nachricht, die die Rohdatenpunkte enthält.
        :type msg: Thing
        :returns: Die Antwortnachricht mit den Vorhersagen.
        :rtype: Thing
        """
        if not isinstance(msg.value, list):
            self._logger.error(f"Payload (the 'value' field of the msg Thing) must be of type list, "
                      f"but received: {type(msg.value)}")
            raise ValueError("Payload must be a list of raw data points")

        raw_data_points: list[dict] = msg.value

        try:
            res: list[dict] = await self._get_prediction(raw_datapoints=raw_data_points)

            return ok_response_thing(payload=res, fiot_service=self)

        except Exception as e:
            self._logger.error(f"Error while processing raw data points: {e}")
            return error_response_thing(exception=e, fiot_service=self)


if __name__ == '__main__':
    # Change this to reduce verbosity or remove completely to use `FASTIOT_LOG_LEVEL` environment variable to configure
    # logging.
    logging.basicConfig(level=logging.DEBUG)
    PytorchServingService.main()
