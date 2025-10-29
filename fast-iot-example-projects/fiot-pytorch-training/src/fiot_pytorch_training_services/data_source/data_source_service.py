import asyncio
import logging

from fastiot.core import FastIoTService, loop
from kio.banner import KIOptiPack_banner
from kio.dataset import example_dataset
from kio.ml_lifecycle_broker_facade import request_save_many_raw_data_points


class DataSourceService(FastIoTService):

    async def _start(self):
        print(KIOptiPack_banner) # Display the KIOptiPack banner on service start
        self._logger.info("MotivSensorService started.")

        # Save an initial dataset when the service starts
        await self.save_initial_dataset()

    async def save_initial_dataset(self):

        dataset = example_dataset

        try:
            response = await request_save_many_raw_data_points(fiot_service=self, data=dataset)
            self._logger.info(f"received response from db-service: {response}")

        except Exception as e:
            # Log any exceptions that occur during the request to save data points.
            self._logger.error(f"Error occurred: {e}.")
            self._logger.error(f"Check if the database service is running.")


    # comment the loop
    # @loop
    async def produce(self):
        # you could add functionality that generates labeld data points here.
        # The following is just an example with two hardcoded data points.
        # This is just to give you an idea how to send data points to the database service.
        labeled_data = [
            {
                "ListeKomponenten": ["K000055", "K000057"],
                "Massenanteile": [0.5, 0.5],
                "Flächenanteilmodifiziert": 0,
                "Geometrie": "Quader",
                "Kopfraumatmosphäre": None,
                "Masse": None,
                "Verpackungstyp": "Folie",
                "CAD": None,
                "RauheitRa": 0.08666666666666667,
                "RauheitRz": 0.924,
                "Trübung": 216.1,
                "Glanz": 36.7,
                "Dicke": 738.6666666666666,
                "Emodul": 807.9225728004443,
                "MaximaleZugspannung": 33.22942107172407,
                "MaximaleLängenänderung": 14.57795412214027,
                "Ausformung": 3,
                "Kaltverfo": 3,
                "Temp": 420,
                "Zeit": 32,
                "Druck": 1,
            },
            {
                "ListeKomponenten": ["K000055", "K000057"],
                "Massenanteile": [0.5, 0.5],
                "Flächenanteilmodifiziert": 0,
                "Geometrie": "Quader",
                "Kopfraumatmosphäre": None,
                "Masse": None,
                "Verpackungstyp": "Folie",
                "CAD": None,
                "RauheitRa": 0.08666666666666667,
                "RauheitRz": 0.924,
                "Trübung": 216.1,
                "Glanz": 36.7,
                "Dicke": 738.6666666666666,
                "Emodul": 807.9225728004443,
                "MaximaleZugspannung": 33.22942107172407,
                "MaximaleLängenänderung": 14.57795412214027,
                "Ausformung": 4,
                "Kaltverfo": 3,
                "Temp": 460,
                "Zeit": 24,
                "Druck": 4.33,
            }
        ]

        # The try-except block is used to handle any exceptions that may occur during the request to save data points.
        # This prevents the service from crashing and provides informative logging.
        try:
            response = await request_save_many_raw_data_points(fiot_service=self, data=labeled_data)
            self._logger.info(f"received response from db-service: {response}")

        except Exception as e:
            # Log any exceptions that occur during the request to save data points.
            self._logger.error(f"Error occurred: {e}.")
            self._logger.error(f"Check if the database service is running.")

        return asyncio.sleep(5) # wait for 5 seconds before producing next data


if __name__ == '__main__':
    # Change this to reduce verbosity or remove completely to use `FASTIOT_LOG_LEVEL` environment variable to configure
    # logging.
    logging.basicConfig(level=logging.DEBUG)
    DataSourceService.main()