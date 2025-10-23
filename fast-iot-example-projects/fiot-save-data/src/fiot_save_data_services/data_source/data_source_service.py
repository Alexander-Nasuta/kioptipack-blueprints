import asyncio
import logging
import random

from fastiot.core import FastIoTService, loop
from kio.banner import KIOptiPack_banner
from kio.ml_lifecycle_broker_facade import request_save_many_raw_data_points


class DataSourceService(FastIoTService):

    async def _start(self):
        print(KIOptiPack_banner) # Display the KIOptiPack banner on service start
        self._logger.info("MotivSensorService started.")

    @loop
    async def produce(self):
        """
        This method creates some dummy data and publishes it at regular intervals.
        Two data points are generated with random sensor names and values.
        """

        # The data can be any dictionary structure. Here we create two example data points with random values.
        # Feel free to modify the structure as per your requirements.
        data = [{
            'sensor_name': f'example_sensor_{random.randint(1, 5)}',
            'value': random.randint(20, 30)
        }, {
            'sensor_name': f'example_sensor_{random.randint(1, 5)}',
            'value': random.randint(20, 30)
        }]
        self._logger.info(f"Publishing: {data}")

        # The try-except block is used to handle any exceptions that may occur during the request to save data points.
        # This prevents the service from crashing and provides informative logging.
        try:
            # These are predefined functions that handle communication with the database service.
            # They are essentially wrapper for the native fastiot request-reply mechanism.
            # They hide the complexity of message creation, sending, and receiving and error handling.
            # Blueprints provide a set of such functions to realize common ml lifecycle tasks.
            # For more complex scenarios, you can also directly use fastiot's broker mechanisms or create your own wrapper functions.
            response = await request_save_many_raw_data_points(fiot_service=self, data=data)
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
