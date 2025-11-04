import asyncio
import logging
import random
import pandas as pd
import sklearn

from fastiot.core import FastIoTService, Subject, subscribe, loop, reply
from fastiot.core.core_uuid import get_uuid
from fastiot.core.time import get_time_now
from fastiot.msg.thing import Thing
from sklearn.pipeline import Pipeline

from kio.pipeline_operations import (
    Discretisation,
    OneHotEncodePd,
    NormalizeCols
)
from kio.pipeline_operations import (
    ColumnDropper,
    DropIncompleteRow,
    FillNaNWithMean,
    FillNaNWithMedian,
    FillNaNWithValue
)
from kio.ml_lifecycle_broker_facade import request_get_all_raw_data_points, \
    request_upsert_many_processed_data_points, ok_response_thing, error_response_thing
from kio.ml_lifecycle_subjects_name import DATA_PROCESSING_PROCESS_RAW_DATA_SUBJECT


class DataProcessingService(FastIoTService):
    _preprocessor_pipeline: Pipeline = None

    @loop
    async def process_data_loop(self):

        if self._preprocessor_pipeline is None:
            self._logger.info("preprocessor pipeline not yet set up. waiting for 5 seconds")
            return asyncio.sleep(5)

        self._logger.info("processing data loop started")
        # Fetch raw data from database
        raw_entries: list[dict] = await request_get_all_raw_data_points(fiot_service=self)
        # convert to pandas dataframe
        df = pd.DataFrame(raw_entries)
        self._logger.info(f"received {len(df)} raw data entries from database. first 5 entries:")
        print(df.head(n=5))
        # process data
        processed_data: pd.DataFrame = self.process_raw_db_data(df=df)
        print("processed data (first 5 entries:): ")
        print(processed_data.head(n=5))

        # convert dataframe to list of dicts
        data_points = processed_data.to_dict(orient="records")
        # push processed data to db via broker
        db_service_response: dict = await request_upsert_many_processed_data_points(fiot_service=self, data=data_points)
        self._logger.info(f"received response from db-service: {db_service_response}")

        return asyncio.sleep(24 * 60 * 60)  # 24h

    async def _start(self):
        self._logger.info("DataProcessingService started")
        await self.setup_pipeline()

    async def setup_pipeline(self):
        self._logger.info("setting up pipeline")

        # this is a list of functions that will be applied to the dataframe
        # it uses sklearn transformers
        #
        # NOTE: sklearn transformers are not designed to work with pandas dataframes, but with numpy arrays
        #       we rewrote the transformers to work with pandas dataframes
        #       if you want to use a transformer that is not implemented yet, you can write your own transformer
        #       look into the feature_engineering.py file to see how to write your own transformer

        steps = []

        # the following steps are just examples
        # feel free to mix and match or add your own

        ################################################################
        #   ___       _           ___ _               _
        #  |   \ __ _| |_ __ _   / __| |___ __ _ _ _ (_)_ _  __ _
        #  | |) / _` |  _/ _` | | (__| / -_) _` | ' \| | ' \/ _` |
        #  |___/\__,_|\__\__,_|  \___|_\___\__,_|_||_|_|_||_\__, |
        #                                                   |___/
        ################################################################

        # this is a function that drops a column from the dataframe
        # some columns may be useless for a ml model, so you can drop them here
        data_clean_op_column_drop = (
            "DATA_CEANING_Drop_datum_col",
            ColumnDropper(target=["datum"])
        )
        steps.append(data_clean_op_column_drop)

        ################################################################

        # example to drop rows that have NaN values in a specific column
        data_clean_op_strat_drop = (
            "DATA_CEANING_Drop_laborant_NaN_Rows",
            DropIncompleteRow(['laborant'])
        )
        steps.append(data_clean_op_strat_drop)

        ################################################################

        # example to fill NaN values with a mean value
        #
        # NOTE: the mean value is calculated from the given dataframe
        #       in case the dataframe is a slice of the db, so the mean value is calculated from a subset of the db,
        #       the mean value will be different from the mean value calculated from the whole db
        #
        #       So, if your db is large, you may want to calculate/query the mean value from the whole db
        #       and then pass it example below that fills NaN values with a specific value

        data_clean_op_start_mean = (
            "DATA_CEANING_fill_rohwert_1_labormessung_NaN_with_mean",
            FillNaNWithMean("rohwert_1_labormessung")
        )
        steps.append(data_clean_op_start_mean)

        ################################################################

        # example to fill NaN values with a median value
        # NOTE: the median value is calculated from the given dataframe
        #       in case the dataframe is a slice of the db, so the median value is calculated from a subset of the db,
        #       the median value will be different from the mean value calculated from the whole db
        #
        #       So, if your db is large, you may want to calculate/query the median value from the whole db
        #       and then pass it example below that fills NaN values with a specific value

        data_clean_op_start_median = (
            "DATA_CEANING_fill_rohwert_2_labormessung_NaN_with_median",
            FillNaNWithMedian("rohwert_2_labormessung")
        )
        steps.append(data_clean_op_start_median)

        data_clean_op_start_median = (
            "DATA_CEANING_fill_aufbereiteter_wert_NaN_with_median",
            FillNaNWithMedian("aufbereiteter_wert")
        )
        steps.append(data_clean_op_start_median)

        ################################################################

        # example to fill NaN values with a specific value
        data_clean_op_start_fill_val = (
            "DATA_CEANING_fill_rohwert_3_labormessung_NaN_with_0.5",
            FillNaNWithValue(target="rohwert_3_labormessung", value=0.5)
        )
        steps.append(data_clean_op_start_fill_val)

        ################################################################
        #   ___       _          _____                  __                    _   _
        #  |   \ __ _| |_ __ _  |_   _| _ __ _ _ _  ___/ _|___ _ _ _ __  __ _| |_(_)___ _ _
        #  | |) / _` |  _/ _` |   | || '_/ _` | ' \(_-<  _/ _ \ '_| '  \/ _` |  _| / _ \ ' \
        #  |___/\__,_|\__\__,_|   |_||_| \__,_|_||_/__/_| \___/_| |_|_|_\__,_|\__|_\___/_||_|
        #
        ################################################################

        # example to discrete a column with a specific number of bins

        data_transform_discretisation = (
            "DATA_TRANSFORM_Discretisation_rohwert_1_labormessung",
            Discretisation(target="rohwert_1_labormessung",
                           bins=5,
                           labels=["very_low", "low", "medium", "high", "very_high"])
        )
        steps.append(data_transform_discretisation)

        ################################################################

        # example to one-hot encode a column
        data_transform_op_ohe_1 = (
            "Data_Transformation_Auto_One_Hot_Encode_Rohwert_1_Labormessung",
            OneHotEncodePd(target="rohwert_1_labormessung", prefix="rohwert_1", sep="_")
        )
        steps.append(data_transform_op_ohe_1)
        data_transform_op_ohe_2 = (
            "DATA_TRANSFORM_Auto_One_Hot_Encode_Material_Id",
            OneHotEncodePd(
                target="material_id",
                prefix="material_id",
                sep="_",
                required_columns=[
                    "material_id_00000000",
                    "material_id_11111111",
                    "material_id_22222222",
                    "material_id_33333333"
                ]
            )
        )
        steps.append(data_transform_op_ohe_2)
        data_transform_op_ohe_3 = (
            "DATA_TRANSFORM_Auto_One_Hot_Encode_Laborant",
            OneHotEncodePd(
                target="laborant",
                prefix="laborant",
                sep="_",
                required_columns=["laborant_AN", "laborant_HANS", "laborant_SO", "laborant_TK"]
            )
        )
        steps.append(data_transform_op_ohe_3)

        ################################################################

        # example to normalize a column
        #
        # NOTE: the min and max values for scaling are calculated from the given dataframe
        # TODO: add version that queries DB for min and max values
        data_transform_op_norm_query_1 = (
            "DATA_TRANSFORM_Normalize_aufbereiteter_wert",
            NormalizeCols(target="aufbereiteter_wert", feature_range=(0, 1))
        )
        steps.append(data_transform_op_norm_query_1)

        data_transform_op_norm_query_2 = (
            "DATA_TRANSFORM_Normalize_rohwert_2_labormessung",
            NormalizeCols(target="rohwert_2_labormessung", feature_range=(0, 1))
        )
        steps.append(data_transform_op_norm_query_2)

        data_transform_op_norm_query_3 = (
            "DATA_TRANSFORM_Normalize_rohwert_3_labormessung",
            NormalizeCols(target="rohwert_3_labormessung", feature_range=(-1, 1))
        )
        steps.append(data_transform_op_norm_query_3)

        ################################################################

        preprocessor = Pipeline(steps=steps, verbose=False)
        self._preprocessor_pipeline = preprocessor

    def process_raw_db_data(self, df: pd.DataFrame) -> pd.DataFrame:
        self._logger.info(f"processing raw data from database")
        processed_dataframe = self._preprocessor_pipeline.fit_transform(df)
        return processed_dataframe

    def process_raw_data_points(self, data: list[dict]) -> list[dict]:
        self._logger.info(f"processing raw data from database")
        df = pd.DataFrame(data)
        processed_dataframe = self._preprocessor_pipeline.fit_transform(df)
        print(processed_dataframe.to_dict(orient="records"))
        return processed_dataframe.to_dict(orient="records")

    @reply(DATA_PROCESSING_PROCESS_RAW_DATA_SUBJECT)
    async def process_many_raw_datapoints(self, topic: str, msg: Thing) -> Thing:
        print("process....")
        if not isinstance(msg.value, list):
            self._logger.error(f"Payload (the 'value' field of the msg Thing) must be of type list, "
                      f"but received: {type(msg.value)}")
            raise ValueError("Payload must be a list of raw data points")

        data_points: list[dict] = msg.value

        self._logger.info(f"Received {len(data_points)} raw data points to be processed")

        try:
            df = pd.DataFrame(data_points)
            processed_data: pd.DataFrame = self.process_raw_db_data(df=df)
            data_points = processed_data.to_dict(orient="records")

            return ok_response_thing(payload=data_points, fiot_service=self)

        except Exception as e:
            self._logger.error(f"Error while processing raw data points: {e}")
            return error_response_thing(exception=e, fiot_service=self)


if __name__ == '__main__':
    # Change this to reduce verbosity or remove completely to use `FASTIOT_LOG_LEVEL` environment variable to configure
    # logging.
    logging.basicConfig(level=logging.DEBUG)
    DataProcessingService.main()
