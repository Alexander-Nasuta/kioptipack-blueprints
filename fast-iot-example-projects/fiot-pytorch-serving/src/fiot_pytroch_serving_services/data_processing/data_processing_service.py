import asyncio
import logging

import pandas as pd

from fastiot.core import FastIoTService, loop, reply
from fastiot.msg import Thing
from kio.ml_lifecycle_subjects_name import DATA_PROCESSING_PROCESS_RAW_DATA_SUBJECT

from sklearn.pipeline import Pipeline

from kio.banner import KIOptiPack_banner
from kio.pipeline_operations import ColumnDropper, NormalizeCols, OneHotEncodePd, NumericOneHotEncodePd, ReplaceNoneValues

from kio.ml_lifecycle_broker_facade import request_get_all_raw_data_points, request_upsert_many_processed_data_points, \
    ok_response_thing, error_response_thing


class DataProcessingService(FastIoTService):
    _preprocessor_pipeline = Pipeline(
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

    async def _start(self):
        print(KIOptiPack_banner)
        self._logger.info("Data Processing Service started.")


    def process_raw_db_data(self, df: pd.DataFrame) -> pd.DataFrame:
        self._logger.info(f"processing raw data from database")
        processed_dataframe = self._preprocessor_pipeline.fit_transform(df)
        return processed_dataframe

    @loop
    async def process_data_loop(self):
        # Fetch raw data from database
        try:
            raw_entries: list[dict] = await request_get_all_raw_data_points(fiot_service=self)
        except Exception as e:
            self._logger.error(f"error while requesting raw data from db-service: {e}")
            return asyncio.sleep(60)  # retry after 1 min

        # convert to pandas dataframe
        df = pd.DataFrame(raw_entries)
        self._logger.info(f"received {len(df)} raw data entries from database. first 5 entries:")
        print(df.head(n=5))

        processed_data: pd.DataFrame = self.process_raw_db_data(df=df)
        self._logger.info("processed data (first 5 entries:): ")
        print(processed_data.head(n=500))

        # print all dtypes of processed data for each column
        self._logger.info("processed data dtypes:")
        for col, dtype in processed_data.dtypes.items():
            self._logger.info(f"  {col}: {dtype}")


        # convert dataframe to list of dicts
        data_points = processed_data.to_dict(orient="records")
        # push processed data to db via broker
        try:
            db_service_response: dict = await request_upsert_many_processed_data_points(fiot_service=self, data=data_points)
            self._logger.info(f"received response from db-service: {db_service_response}")

        except Exception as e:
            # Log any exceptions that occur during the request to save data points.
            self._logger.error(f"Error occurred: {e}.")
            self._logger.error(f"Check if the database service is running.")

        return asyncio.sleep(24 * 60 * 60)  # 24h

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