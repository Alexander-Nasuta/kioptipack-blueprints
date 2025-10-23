import logging
import uuid

from fastiot.core import FastIoTService, reply
from fastiot.msg.thing import Thing

import pymongo
from kio.ml_lifecycle_broker_facade import ok_response_thing
from kio.ml_lifecycle_subjects_name import DB_SAVE_MANY_RAW_DATAPOINTS_SUBJECT
from pymongo import MongoClient
from pymongo.results import InsertManyResult

from kio.banner import KIOptiPack_banner

class MongoDatabaseService(FastIoTService):
    # Note: for blueprints we hardcode all configuration parameters.
    #       In a real application, you would want to load these from a configuration file or environment variables.
    #       For that you can use fastiot's built-in configuration management. Those can be set in the individual
    #       deployments in the deployment folder using the .env files.
    #       See fastiot docs for more info.
    #
    #       The hardcoded parameters are used for a lower barrier to entry for trying out the blueprints.

    # Note in order to use authentication, please first create a user in your mongo database. See mongo docs for more info.
    # This is optional. If no username and password are provided, no authentication will be used.
    _db_username = None # example :'fiot'
    _db_password = None # example 'fiotdev123'
    _db_port = '27017' # default mongo port
    _db_host = 'localhost' # example 'localhost'. This can be an IP or domain name

    _DB_NAME = "mongodb" # this is just a label for logging purposes. This is present since fastiot has built-in support for different databases.
    _MONGO_DB = "fiot-save-data-blueprint" # name of the mongo database
    _MONGO_EXAMPLE_DATA_COLLECTION = "example_data" # name of the collection where example data will be stored

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if self._db_username and self._db_password:
            # Using authentication
            connection_string = f"mongodb://{self._db_username}:{self._db_password}@{self._db_host}:{self._db_port}/?authMechanism=SCRAM-SHA-1"
        else:
            # Without authentication
            connection_string = f"mongodb://{self._db_host}:{self._db_port}/"

        self._mongodb_client = MongoClient(connection_string)
        self._db = self._mongodb_client[self._MONGO_DB]
        self._raw_data_collection = self._db[self._MONGO_EXAMPLE_DATA_COLLECTION]


    async def _start(self):
        print(KIOptiPack_banner)
        self._logger.info("MotivSensorService started.")
        self._logger.info("Feel free to check your mongo database in a GUI (like MongoDB Compass) or via command line.")


    @reply(DB_SAVE_MANY_RAW_DATAPOINTS_SUBJECT)
    async def db_save_many_raw_datapoints(self, _: str, msg: Thing) -> Thing:
        """
        Saves many raw data points to the database.

        :param _: The topic of the message. This is not used in this method.
        :type _: str
        :param msg: The message that contains the raw data points to be saved to the database.
        :return: A Thing object that contains the result of the operation. This is either an acknowledgement or an error.
                Acknowledgements contain the number of raw data points that were saved to the database.
        :rtype: Thing

        """
        if not isinstance(msg.value, list):
            self._logger.error(f"Payload (the 'value' field of the msg Thing) must be of type list, "
                               f"but received: {type(msg.value)}")
            raise ValueError("Payload must be a list of raw data points")

        data_points: list[dict] = msg.value
        self._logger.info(f"Received {len(data_points)} raw data points to be inserted into mongodb")

        # add uuids to data points
        for data_point in data_points:
            data_point["_id"] = str(uuid.uuid4())

        self._logger.info(f"Inserting data points into mongodb")
        res: InsertManyResult = self._raw_data_collection.insert_many(data_points)
        self._logger.info(f"DB transaction result: {res.acknowledged}")

        self._logger.info(f"Inserted {len(res.inserted_ids)} raw data points into mongodb")

        # feel free to include whatever information you want to return here.
        db_specific_info = {
            "acknowledged": True,
            "db": "MongoDB",
        }

        # in principle one does not need to return information here.
        # However, some infos are return here, so that the requesting service can log the information.
        return ok_response_thing(payload=db_specific_info, fiot_service=self)


if __name__ == '__main__':
    # Change this to reduce verbosity or remove completely to use `FASTIOT_LOG_LEVEL` environment variable to configure
    # logging.
    logging.basicConfig(level=logging.DEBUG)
    MongoDatabaseService.main()
