import asyncio
import logging
import pprint
import random

from fastiot.core import FastIoTService, loop
from datetime import datetime

from kio.ml_lifecycle_broker_facade import request_get_prediction


class MlConsumerService(FastIoTService):

    @staticmethod
    def _get_random_raw_datapoint() -> dict:
        return {
            'laborant': ["TK", "HANS", "AN", "SO"][random.randint(0, 3)],
            'material_id': ["00000000", "11111111", "22222222", "33333333"][random.randint(0, 3)],
            'datum':  datetime.now().strftime("%d.%m.%Y, %H:%M:%S"),
            'rohwert_1_labormessung': random.uniform(0, 30),
            'rohwert_2_labormessung': random.uniform(0, 30),
            'rohwert_3_labormessung': random.uniform(0, 2),
            'aufbereiteter_wert''':  0.1,
        }

    @loop
    async def request_prediction(self):
        self._logger.info("Requesting prediction")
        raw_unlabeled_datapoints = [self._get_random_raw_datapoint() for _ in range(2)]
        self._logger.info(f"Requesting predictions for: \n{pprint.pformat(raw_unlabeled_datapoints)}")
        predictions = await request_get_prediction(fiot_service=self, data=raw_unlabeled_datapoints)
        self._logger.info(f"Received predictions: \n{pprint.pformat(predictions)}")
        return asyncio.sleep(5)


if __name__ == '__main__':
    # Change this to reduce verbosity or remove completely to use `FASTIOT_LOG_LEVEL` environment variable to configure
    # logging.
    logging.basicConfig(level=logging.DEBUG)
    MlConsumerService.main()
