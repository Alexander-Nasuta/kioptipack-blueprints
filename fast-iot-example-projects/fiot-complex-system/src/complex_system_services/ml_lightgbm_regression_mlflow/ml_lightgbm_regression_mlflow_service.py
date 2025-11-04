import logging
import asyncio
from typing import Tuple

import mlflow
import lightgbm as lgbm
import numpy as np
import pandas as pd

from fastiot.core import FastIoTService, loop
from pandas import DataFrame
from rich.progress import Progress
from lightgbm import LGBMRegressor, Dataset
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

from kio.ml_lifecycle_broker_facade import (request_get_processed_data_points_count,
                                                                            request_get_processed_data_points_page)


class PageSequence():
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
        self._logger.info("Loading initial page. . .")
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
        self._logger.info(f"Loading next page {self._current_page}. . .")
        df = await self._get_page_df(page=self._current_page)
        self._page_df = df

    async def get_complete_sequence(self) -> tuple[DataFrame, DataFrame, DataFrame, DataFrame]:
        """
        Get the complete sequence of pages.
        Returns
        -------
        tuple(pd.Dataframe, pd.Dataframe, pd.Dataframe, pd.Dataframe)
            tuple containing the complete data sequence split into x and y as well as train and test data.
        """
        x_train: pd.DataFrame
        x_test: pd.DataFrame
        y_train: pd.DataFrame
        y_test: pd.DataFrame
        page_list = []
        for page in range(self._total_pages):
            temp = self._page_df.copy()
            page_list.append(temp)
            await self.load_next_page()

        data_set = pd.concat(page_list)

        y_df = data_set.pop("aufbereiteter_wert")
        x_df = data_set

        x_train, x_test, y_train, y_test = train_test_split(x_df, y_df, test_size=0.15)

        return x_train, x_test, y_train, y_test


class MlLightgbmRegressionMlflowService(FastIoTService):
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
    create_model()
        Create the model.
    training_loop
        Initializes training of models
    train_model_without_experiment_tracking
        Trains a model without tracking the process
    train_model_with_experiment_tracking
        Trains a model while tracking the process with MLFlow
    """
    MLFLOW_TRACKING_URI = "http://127.0.0.1:8080"
    MLFLOW_EXPERIEMENT_ID = "497440626014835498"

    async def _start(self):
        """
        Runs when the service starts.
        """
        self._logger.info("MlLGBMRegressionService started")
        self._logger.info(f"Setting MLFlow tracking uri to {self.MLFLOW_TRACKING_URI}")
        mlflow.set_tracking_uri(self.MLFLOW_TRACKING_URI)

    async def _stop(self):
        """
        Runs when the service stops.
        """
        self._logger.info("MlLGBMRegressionService stopped")

    @loop
    async def training_loop(self):
        """
        Loops the training process.
        """
        params = {
            'task': 'train',
            'boosting': 'gbdt',
            'objective': 'regression',
            'num_leaves': 10,
            'learning_rate': 0.05,
            'metric': {'l2', 'l1'},
            'verbose': -1,
            'early_stopping_rounds': 30
        }

        data_sequence = PageSequence(fast_iot_service=self, page_size=10)
        await data_sequence.init_dataset()

        await self.train_model_with_experiment_tracking(data_sequence=data_sequence, paramters=params)

        return asyncio.sleep(24 * 60 * 60)

    async def train_model_without_experiment_tracking(self, data_sequence: PageSequence, paramters: dict):
        """
        Trains the model without experiment tracking.
        Parameters
        ----------
        data_sequence
        paramters

        Returns
        -------

        """
        x_train, x_test, y_train, y_test = await data_sequence.get_complete_sequence()
        lgbm_ds_train = lgbm.Dataset(x_train, y_train)
        lgbm_ds_test = lgbm.Dataset(x_test, y_test, reference=lgbm_ds_train)

        model = lgbm.train(paramters, train_set=lgbm_ds_train, valid_sets=[lgbm_ds_test])

        y_pred = model.predict(x_test)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)

        self._logger.info("MSE: %.2f" % mse)
        self._logger.info("RMSE: %.2f" % rmse)

        self._logger.info("Training loop without experiment tracking completed.")

    async def train_model_with_experiment_tracking(self, data_sequence: PageSequence, paramters: dict):
        """
        Train model with experiment tracking.
        Parameters
        ----------
        data_sequence
        paramters

        Returns
        -------

        """
        mlflow.lightgbm.autolog()
        with mlflow.start_run(experiment_id=self.MLFLOW_EXPERIEMENT_ID) as run:

            x_train, x_test, y_train, y_test = await data_sequence.get_complete_sequence()
            lgbm_ds_train = lgbm.Dataset(x_train, y_train)
            lgbm_ds_test = lgbm.Dataset(x_test, y_test, reference=lgbm_ds_train)

            model = lgbm.train(paramters, train_set=lgbm_ds_train, valid_sets=[lgbm_ds_test])

            y_pred = model.predict(x_test)
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)

            self._logger.info("MSE: %.2f" % mse)
            self._logger.info("RMSE: %.2f" % rmse)


if __name__ == '__main__':
    # Change this to reduce verbosity or remove completely to use `FASTIOT_LOG_LEVEL` environment variable to configure
    # logging.
    logging.basicConfig(level=logging.DEBUG)
    MlLightgbmRegressionMlflowService.main()
