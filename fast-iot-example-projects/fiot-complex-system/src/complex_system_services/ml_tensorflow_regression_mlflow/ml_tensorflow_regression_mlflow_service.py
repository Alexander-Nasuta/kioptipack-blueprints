import logging
import asyncio
import pprint

import mlflow

import pandas as pd
import numpy as np

import tensorflow as tf
from keras.api.layers import Dense, InputLayer
from keras.api.utils import PyDataset
from keras.api.models import Model

from fastiot.core import FastIoTService, loop
from rich.progress import Progress


from kio.ml_lifecycle_broker_facade import request_get_processed_data_points_count, request_get_processed_data_points_page


class DemonstratorTFModel(Model):
    """
    A simple Tensorflow model for demonstration purposes.

    Attributes
    ----------
    input_layer : InputLayer
        An InputLayer to ensure the correct shape of data is handed to the model
    dense1 : Dense
        A Dense layer as the first weighted layer
    dense2 : Dense
        A Dense layer as the second weighted layer
    output_layer : Dense
        A Dense layer as the output layer

    Methods
    -------
    call(x)
        Forward pass through the Model.
    """

    def __init__(self, data_shape: tuple = (15, 1), dense_layer1: int = 64, dense_layer2: int = 64,
                 output_shape: int = 1):
        super(DemonstratorTFModel, self).__init__()
        self.input_layer = InputLayer(shape=data_shape)
        self.dense1 = Dense(dense_layer1, activation='relu')
        self.dense2 = Dense(dense_layer2, activation='relu')
        self.output_layer = Dense(output_shape)

    def call(self, x):
        x = self.dense1(x)
        x = self.dense2(x)
        return self.output_layer(x)


class PageSequence(PyDataset):
    _page_size: int
    _total_pages: int
    _num_entries_in_db: int
    _current_page: int

    _fast_iot_service: FastIoTService
    _broker_timeout: float

    _page_df: pd.DataFrame

    def __init__(self, fast_iot_service: FastIoTService, page_size: int, broker_timeout=10,
                 batch_size: int = 5 ,**kwargs):
        """
        Initialize the sequence.

        Parameters
        ----------
        fast_iot_service
        page_size
        broker_timeout
        """
        super().__init__(**kwargs)
        self._fast_iot_service = fast_iot_service
        self._broker_timeout = broker_timeout
        self._page_size = page_size
        self._batch_size = batch_size

    def __len__(self):
        """
        Return the length of the dataset.

        Returns
        -------
        int
            The length of the dataset.
        """
        return len(self._page_df)

    async def _init_total_pages(self):
        """
        Initialize the total number of pages.

        Returns
        -------
        int
            The total number of pages.
        """
        # count
        count: int = await request_get_processed_data_points_count(fiot_service=self._fast_iot_service)
        self._num_entries_in_db = count
        self._total_pages = int(np.ceil(self._num_entries_in_db / self._page_size))

    async def _get_page_df(self, page: int) -> pd.DataFrame:
        """
        Get the dataframe for a page.

        Parameters
        ----------
        page
            The page. (A slice of the data present in the database.)

        Returns
        -------
        pd.DataFrame
            The dataframe for the page.
        """
        # query the db_service for the processed data points
        page: list[dict] = await request_get_processed_data_points_page(
            fiot_service=self._fast_iot_service,
            page=page,
            page_size=self._page_size
        )
        return pd.DataFrame(page)

    async def init_dataset(self):
        """
        Initialize the dataset.

        Returns
        -------
        None
        """
        # init total number of pages
        await self._init_total_pages()
        df = await self._get_page_df(page=0)
        self._page_df = df
        self._current_page = 0

    def has_next_page(self):
        """
        Check if there is a next page.

        Returns
        -------
        bool
            True if there is a next page, False otherwise.
        """
        return self._current_page < self._total_pages

    @property
    def num_pages(self):
        """
        The total number of pages.

        Returns
        -------
        """
        if self._total_pages is None:
            self._logger.warn("total pages not initialized. init_page() needs to called and awaited first.")
        return self._total_pages

    async def load_next_page(self):
        """
        Load the next page.

        Returns
        -------
        None
        """
        if self._current_page is None:
            self._logger.error("page not initialized. init_page() needs to called and awaited first.")
            raise ValueError("page not initialized. init_page() needs to called and awaited first.")

        if self._current_page >= self._total_pages:
            self._logger.error("no more pages available")
            raise ValueError("no more pages available")

        self._current_page += 1
        df = await self._get_page_df(page=self._current_page)
        self._page_df = df

    def __getitem__(self, idx):  # idx means index of the chunk.
        """
        Get an item from the dataset.

        Parameters
        ----------
        idx

        Returns
        -------
        tuple
            The input and output data.
        """
        # drop index column
        temp = self._page_df
        sample_size = min(self._batch_size, len(temp))
        temp = temp.sample(n=sample_size, replace=False)

        y_df = temp.pop("aufbereiteter_wert")
        x_df = temp

        y_npa = np.array(y_df)
        x_npa = x_df.to_numpy()

        if idx == self.__len__():
            raise IndexError

        return x_npa, y_npa


class MlTensorflowRegressionMlflowService(FastIoTService):
    """
    A service for training a tensorflow model with optional mlflow experiment tracking.

    Attributes
    ----------
    MLFLOW_TRACKING_URI : str
        The mlflow tracking uri.

    Methods
    -------
    _start()
        Start the service.
    _stop()
        Stop the service.
    get_model()
        Get the model.
    training_loop
        Initializes a training loop of models
    train_model_without_experiment_tracking
        Trains a model without tracking the process
    train_model_with_experiment_tracking
        Trains a model while tracking the process with MLFlow
    """
    MLFLOW_TRACKING_URI = "http://127.0.0.1:8080"
    MLFLOW_EXPERIMENT_ID = "498700715529195410"

    async def _start(self):
        """
        Runs when the service starts.
        """
        self._logger.info("MlPytorchRegressionService started")
        self._logger.info(f"Setting MLFlow tracking uri to {self.MLFLOW_TRACKING_URI}")
        mlflow.set_tracking_uri(self.MLFLOW_TRACKING_URI)

    async def _stop(self):
        """
        Runs when the service stops.
        """
        self._logger.info("MlPytorchRegressionService stopped")

    def get_model(self) -> DemonstratorTFModel:
        """
        Create a model instance.

        Returns
        -------
        DemonstratorTFModel
            A model instance.
        """
        return DemonstratorTFModel(
            data_shape=(15, 1),
            dense_layer1=64,
            dense_layer2=64,
            output_shape=1
        )

    @loop
    async def training_loop(self):
        """
        Loops the training process.
        """
        model = self.get_model()
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        loss_fn = tf.keras.losses.MeanSquaredError()
        metrics = [tf.keras.metrics.MeanAbsoluteError()]
        model.compile(optimizer=optimizer, loss=loss_fn, metrics=metrics)
        dataset = PageSequence(fast_iot_service=self, page_size=10)

        await self.train_model_with_experiment_tracking(dataset, model)

        return asyncio.sleep(24 * 60 * 60)

    async def train_model_without_experiment_tracking(self, dataset: PageSequence, model: DemonstratorTFModel,
                                                      epochs: int = 5):
        """
        Train the model without experiment tracking.

        Parameters
        ----------
        dataset
        model
        epochs

        Returns
        -------
        """
        self._logger.info("Starting training loop without experiment tracking.")
        await dataset.init_dataset()
        total_steps = dataset.num_pages * epochs
        with Progress() as progress:
            task = progress.add_task("[cyan]Training...", total=total_steps)
            for page in range(dataset.num_pages):
                for epoch in range(epochs):
                    for batch_idx in range(len(dataset)):
                        x, y = dataset[batch_idx]
                        with tf.GradientTape() as tape:
                            y_pred = model(x, training=True)
                            loss = tf.keras.losses.MeanSquaredError()(y, y_pred)
                        gradients = tape.gradient(loss, model.trainable_variables)
                        temp = model.evaluate(x, y, return_dict=True)
                        model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))
                        progress.update(task, advance=1, )
                await dataset.load_next_page()

        self._logger.info("Training loop without experiment tracking completed.")

    async def train_model_with_experiment_tracking(self, dataset: PageSequence, model: DemonstratorTFModel,
                                                   epochs: int = 5):
        """
        Trains the model with experiment tracking.

        Parameters
        ----------
        dataset
        model
        epochs

        Returns
        -------
        """
        self._logger.info("Starting training loop with experiment tracking.")
        await dataset.init_dataset()
        total_steps = dataset.num_pages * epochs
        with mlflow.start_run(experiment_id=self.MLFLOW_EXPERIMENT_ID) as run:
            with Progress() as progress:
                optimizer_step = 0
                task = progress.add_task("[cyan]Training...", total=total_steps)
                for page in range(dataset.num_pages):
                    for epoch in range(epochs):
                        for batch_idx in range(len(dataset)):
                            x, y = dataset[batch_idx]
                            with tf.GradientTape() as tape:
                                y_pred = model(x, training=True)
                                loss = tf.keras.losses.MeanSquaredError()(y, y_pred)
                            gradients = tape.gradient(loss, model.trainable_variables)
                            model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))
                            temp = model.evaluate(x, y, return_dict=True)
                            metrics = {
                                "loss": temp["loss"],
                                "MAE": temp["mean_absolute_error"]
                            }
                            mlflow.log_metrics(metrics=metrics, step=optimizer_step)
                            progress.update(task, advance=1, )
                            optimizer_step += 1
                    await dataset.load_next_page()

            self._logger.info("Training loop with MLFlow experiment tracking completed.")
            mlflow.tensorflow.log_model(model=model, artifact_path="model")

            model_uri = f"runs:/{run.info.run_id}/model"
            model_details = mlflow.register_model(model_uri=model_uri, name="MyModel")
            self._logger.info(f"registered model in mlfow model registry. Details: \n {pprint.pformat(dict(model_details))}")


if __name__ == '__main__':
    # Change this to reduce verbosity or remove completely to use `FASTIOT_LOG_LEVEL` environment variable to configure
    # logging.
    logging.basicConfig(level=logging.DEBUG)
    MlTensorflowRegressionMlflowService.main()
