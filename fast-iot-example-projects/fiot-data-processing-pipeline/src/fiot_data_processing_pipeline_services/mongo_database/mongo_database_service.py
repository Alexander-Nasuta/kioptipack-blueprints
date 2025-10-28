import logging
import pprint
import uuid

from fastiot.core import FastIoTService, reply
from fastiot.msg.thing import Thing

import pymongo
from kio.ml_lifecycle_broker_facade import ok_response_thing, error_response_thing
from kio.ml_lifecycle_subjects_name import DB_SAVE_MANY_RAW_DATAPOINTS_SUBJECT, \
    DB_UPSERT_MANY_PROCESSED_DATAPOINTS_SUBJECT, DB_GET_ALL_RAW_DATA_SUBJECT
from pymongo import MongoClient, UpdateOne
from pymongo.results import InsertManyResult, BulkWriteResult

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
    _MONGO_DB = "fiot-data-processing-blueprint" # name of the mongo database
    _MONGO_EXAMPLE_DATA_COLLECTION = "KIOptiPackRaw" # name of the collection where example data will be stored
    _MONGO_PROCESSED_DATA_COLLECTION = "KIOptiPackProcessed"

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
        self._processed_data_collection = self._db[self._MONGO_PROCESSED_DATA_COLLECTION]


    async def _start(self):
        print(KIOptiPack_banner)
        self._logger.info("MotivSensorService started.")
        self._logger.info("Feel free to check your mongo database in a GUI (like MongoDB Compass) or via command line.")

        # drop collection on start for demo purposes
        # in a real application you probably don't want this behavior
        # this is just to avoid duplicate data points when restarting the service multiple times during development
        # alternatively, you can add id fields to your data points to avoid duplicates and youse upserts instead of inserts
        self._db.drop_collection(self._MONGO_EXAMPLE_DATA_COLLECTION)


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
            if "_id" not in data_point:
                data_point["_id"] = str(uuid.uuid4())

        # log the data points to be inserted
        self._logger.debug(f"Data points to be inserted: {pprint.pformat(data_points)}")

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

    @reply(DB_GET_ALL_RAW_DATA_SUBJECT)
    async def get_all_raw_data(self, _: str, __: Thing) -> Thing:
        """
        Gets all raw data from the database.
        Parameters
        ----------
        _
            The topic of the message. This is not used in this method.
        __
            The message that contains the request to get all raw data from the database.

        Returns
        -------
        Thing
            A Thing object that contains the result of the operation. This is either an acknowledgement or an error.
            Acknowledgements contain all the raw data from the database.

        """

        self._logger.info("Received request to get all raw data from mongodb")

        filter_kwargs = {}  # empty dict means no filter
        raw_data_entries = self._raw_data_collection.find()
        raw_data_entries = [dict(data) for data in raw_data_entries]

        # the native mongo ID is not serializable
        # so we convert it to a string
        for data in raw_data_entries:
            data["_id"] = str(data["_id"])

        return ok_response_thing(payload=raw_data_entries, fiot_service=self)

    @reply(DB_UPSERT_MANY_PROCESSED_DATAPOINTS_SUBJECT)
    async def db_save_upsert_processed_datapoints(self, _: str, msg: Thing) -> Thing:
        """
        Upserts many processed data points to the database.

        Parameters
        ----------
        _
            The topic of the message. This is not used in this method.
        msg
            The message that contains the processed data points to be upserted to the database.

        Returns
        -------
        Thing
            A Thing object that contains the result of the operation. This is either an acknowledgement or an error.
            Acknowledgements contain the number of processed data points that were upserted to the database.
        """

        if not isinstance(msg.value, list):
            self._logger.error(f"Payload (the 'value' field of the msg Thing) must be of type list, "
                               f"but received: {type(msg.value)}")
            raise ValueError("Payload must be a list of processed data points")

        data_points: list[dict] = msg.value
        self._logger.info(f"Received {len(data_points)} processed data points to be inserted into mongodb")

        #print(pprint.pformat(data_points))

        self._logger.info(f"Upserting data points into mongodb using bulk wirte")
        # Prepare a list of UpdateOne operations
        operations = [
            UpdateOne(
                {"_id": data_point["_id"]},  # filter
                {"$set": data_point},  # update
                upsert=True  # upsert
            )
            for data_point in data_points
        ]
        try:
            res: BulkWriteResult = self._processed_data_collection.bulk_write(operations)
        except Exception as e:
            self._logger.error(f"Error while upserting processed data points into mongodb: {e}")
            return error_response_thing(exception=e, fiot_service=self)
        self._logger.info(f"DB transaction result: {res.acknowledged}")


        all_processed_data_entries: list[dict] = self._processed_data_collection.find()
        for element in all_processed_data_entries:
            #print(element)
            pass

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
