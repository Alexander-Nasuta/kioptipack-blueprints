import asyncio
import logging
import pprint
import uuid

import mlflow
import random

import torch

import numpy as np
import pandas as pd

from fastiot.core import FastIoTService, Subject, subscribe, loop
from fastiot.core.core_uuid import get_uuid
from fastiot.core.time import get_time_now
from fastiot.msg.thing import Thing
from rich.progress import Progress
from torch.utils.data import Dataset

from kio.ml_lifecycle_broker_facade import request_get_processed_data_points_count, \
    request_get_all_raw_data_points, request_get_processed_data_points_page


from torch import nn, optim


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


class PageDataset(Dataset):
    """
    A custom dataset for the pytorch regression service.

    Attributes
    ----------
    _page_size : int
        The size of a page.
    _total_pages : int
        The total number of pages.
    _num_entries_in_db : int
        The total number of entries in the database.
    _current_page : int
        The current page.
    _fast_iot_service : FastIoTService
        The fast iot service.
    _broker_timeout : float
        The broker timeout.
    _page_df : pd.DataFrame
        The page dataframe.

    Methods
    -------
    __len__()
        Return the length of the dataset.
    _init_total_pages()
        Initialize the total number of pages.
    _get_page_df(page)
        Get the dataframe for a page.
    init_dataset()
        Initialize the dataset.
    has_next_page()
        Check if there is a next page.
    load_next_page()
        Load the next page.
    __getitem__(idx)
        Get an item from the dataset.
    """
    _page_size: int
    _total_pages: int
    _num_entries_in_db: int
    _current_page: int

    _fast_iot_service: FastIoTService
    _broker_timeout: float

    _page_df: pd.DataFrame

    def __init__(self, fast_iot_service: FastIoTService, page_size: int, broker_timeout=10):
        """
        Initialize the dataset.

        Parameters
        ----------
        fast_iot_service
        page_size
        broker_timeout
        """
        self._fast_iot_service = fast_iot_service
        self._broker_timeout = broker_timeout

        self._page_size = page_size

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
        # query the db_service for the number of raw data points
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
        temp = temp.iloc[idx]

        y_data = np.array([temp.pop("aufbereiteter_wert")])
        x_data = temp.to_numpy()

        # The following condition is actually needed in Pytorch. Otherwise, for our particular example,
        # the iterator will be an infinite loop.
        # Readers can verify this by removing this condition.
        if idx == self.__len__():
            raise IndexError

        return x_data, y_data


class MlPytorchRegressionMlflowService(FastIoTService):
    """
    A service for training a pytorch model with mlflow experiment tracking.

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
    training_loop()
        The training loop.
    train_model_without_experiment_tracking(dataset, model, loss_fn, optimizer, epochs, batch_size, shuffle)
        Train the model without experiment tracking.
    train_model_with_wandb_tracking(dataset, model, loss_fn, optimizer, epochs, batch_size, shuffle)
        Train the model with wandb tracking.
    """
    MLFLOW_TRACKING_URI = "http://127.0.0.1:8080"

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

    def get_model(self) -> DemonstratorNeuralNet:
        """
        create a model instance.

        Returns
        -------
        DemonstratorNeuralNet
            A model instance.
        """
        return DemonstratorNeuralNet(
            input_dim=15,
            hidden_dim=10,
            output_dim=1
        )

    @loop
    async def training_loop(self):
        """
        The training loop.
        Returns
        -------

        """
        model = self.get_model()
        loss_fn = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        dataset = PageDataset(fast_iot_service=self, page_size=10)

        # await self.train_model_without_experiment_tracking(dataset, model, loss_fn, optimizer)
        await self.train_model_with_mlflow_tracking(dataset, model, loss_fn, optimizer)

        return asyncio.sleep(24 * 60 * 60)

    async def train_model_without_experiment_tracking(self, dataset: PageDataset, model: DemonstratorNeuralNet,
                                                      loss_fn: nn.MSELoss,
                                                      optimizer: optim.Adam, epochs: int = 5, batch_size: int = 5,
                                                      shuffle: bool = True):
        """
        Train the model without experiment tracking.

        Parameters
        ----------
        dataset
        model
        loss_fn
        optimizer
        epochs
        batch_size
        shuffle

        Returns
        -------
        """
        self._logger.info("Starting training loop without experiment tracking.")
        await dataset.init_dataset()
        progress = Progress()
        total_steps = dataset.num_pages * epochs
        task_id = progress.add_task("[cyan]Training...", total=total_steps)

        with progress:
            for page in range(dataset.num_pages):
                # define pytorch data loader
                data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

                # define pytorch training loop
                for epoch in range(epochs):
                    for batch_idx, (x, y) in enumerate(data_loader):
                        optimizer.zero_grad()
                        y_pred = model(x.to(torch.float32)).to(torch.float32)
                        loss = loss_fn(y_pred, y.to(torch.float32))
                        loss.backward()
                        optimizer.step()
                        # self._logger.info(f"page: {page}, epoch: {epoch}, batch_idx: {batch_idx}, loss: {loss.item()}")
                    progress.update(task_id, advance=1)

                await dataset.load_next_page()

        self._logger.info("Training loop without experiment tracking completed.")
        # save model
        # here you can implement a saving mechanism for the model

    async def train_model_with_mlflow_tracking(self, dataset: PageDataset, model: DemonstratorNeuralNet,
                                               loss_fn: nn.MSELoss, optimizer: optim.Adam, epochs: int = 5,
                                               batch_size: int = 5, shuffle: bool = True):
        """
        Train the model with mlflow tracking.

        Parameters
        ----------
        dataset
        model
        loss_fn
        optimizer
        epochs
        batch_size
        shuffle

        Returns
        -------
        """
        self._logger.info("Starting training loop with wandb tracking.")
        await dataset.init_dataset()
        progress = Progress()
        total_steps = dataset.num_pages * epochs
        task_id = progress.add_task("[cyan]Training", total=total_steps)

        with mlflow.start_run() as run:

            with progress:
                optimizer_step = 0
                for page in range(dataset.num_pages):
                    # define pytorch data loader
                    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

                    # define pytorch training loop
                    for epoch in range(epochs):
                        for batch_idx, (x, y) in enumerate(data_loader):
                            optimizer.zero_grad()
                            y_pred = model(x.to(torch.float32)).to(torch.float32)
                            loss = loss_fn(y_pred, y.to(torch.float32))
                            loss.backward()
                            optimizer.step()
                            # Log metrics with wandb
                            metrics = {
                                "loss": loss.item(),
                                "epoch": epoch,
                                "page": page,
                                "optimizer_step": optimizer_step
                            }
                            mlflow.log_metrics(metrics, step=optimizer_step)
                            self._logger.debug(f"page: {page}, epoch: {epoch}, batch_idx: {batch_idx}, loss: {loss.item()}")

                            optimizer_step += 1
                        progress.update(task_id, advance=1, )

                    await dataset.load_next_page()

            self._logger.info("Training loop with wandb tracking completed.")

            mlflow.pytorch.log_model(pytorch_model=model, artifact_path="model")

            model_uri = f"runs:/{run.info.run_id}/model"
            model_details = mlflow.register_model(model_uri=model_uri, name="MyModel")
            self._logger.info(f"registered model in mlfow model regestry. Details: \n {pprint.pformat(dict(model_details))}")


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    MlPytorchRegressionMlflowService.main()
