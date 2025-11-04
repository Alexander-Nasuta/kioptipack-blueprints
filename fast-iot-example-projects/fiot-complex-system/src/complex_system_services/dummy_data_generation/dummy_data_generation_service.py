import logging
import random
import math

from fastiot.core import FastIoTService

from kio.ml_lifecycle_broker_facade import request_save_many_raw_data_points

from datetime import datetime


class DummyDataGenerationService(FastIoTService):

    async def _start(self):
        self._logger.info("DummyDataGenerationServiceService started")
        await self.generate_dummy_data(num_data_points=100)

    async def generate_dummy_data(self, num_data_points=10):
        self._logger.info(f"generating {num_data_points} dummy data entries. "
                 f"some entries will be intentionally invalid/missing values.")

        material_ids = [
            "00000000",
            "11111111",
            "22222222",
            "33333333",
        ]

        laboratory_technicians = ["TK", "AN", "SO", "HANS", None]
        # laboratory_technicians = ["TK"*1000] # use this line to prove errors in MariaDb. This line will cause an error

        data_points: list = [
            {
                "material_id": material_ids[random.randint(0, len(material_ids) - 1)],
                "datum": datetime.now().strftime("%d.%m.%Y, %H:%M:%S"),
                "laborant": laboratory_technicians[random.randint(0, len(laboratory_technicians) - 1)],
                "rohwert_1_labormessung": random.uniform(0, 30),
                "rohwert_2_labormessung": random.uniform(0, 30),
                "rohwert_3_labormessung": random.uniform(0, 2),
                "aufbereiteter_wert": None,
            } for _ in range(num_data_points)
        ]

        for i in data_points:

            if random.random() > 0.5:
                i["aufbereiteter_wert"] = (i["rohwert_1_labormessung"] + i["rohwert_2_labormessung"]) ** 1.5 \
                                          - math.exp(-0.25 * i["rohwert_3_labormessung"])

            elif random.random() > 0.5:
                i["rohwert_1_labormessung"] = None

            elif random.random() > 0.5:
                i["rohwert_2_labormessung"] = None

            elif random.random() > 0.5:
                i["rohwert_3_labormessung"] = None

        db_service_response: dict = await request_save_many_raw_data_points(fiot_service=self, data=data_points)
        self._logger.info(f"received response from db-service: {db_service_response}")


if __name__ == '__main__':
    # Change this to reduce verbosity or remove completely to use `FASTIOT_LOG_LEVEL` environment variable to configure
    # logging.
    logging.basicConfig(level=logging.DEBUG)
    DummyDataGenerationService.main()
