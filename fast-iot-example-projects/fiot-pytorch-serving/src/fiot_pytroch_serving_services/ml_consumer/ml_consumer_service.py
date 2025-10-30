import asyncio
import logging
import random
import pprint

from fastiot.core import FastIoTService, Subject, subscribe, loop
from kio.ml_lifecycle_broker_facade import request_get_prediction


class MlConsumerService(FastIoTService):

    @staticmethod
    def _get_random_raw_datapoint() -> dict:
        random_num_range_0_1 = random.random()
        return {
            "ListeKomponenten": ["K000055", "K000057"],  # id or material name
            "Massenanteile": [random_num_range_0_1, 1 - random_num_range_0_1],  # unit g/g
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
            "Emodul": random.random() * (923.5297844703941-775.2626646454261) + 775.2626646454261,  # unit MPa
            "MaximaleZugspannung": 37.156951742990245,  # unit MPa
            "MaximaleLängenänderung": 19.73276680651324,  # unit %
            # Quality Labels
            "Ausformung": 6,
            "Kaltverfo": 3,
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
