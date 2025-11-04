import logging
import uuid
import os

from fastiot.core import FastIoTService, reply
from fastiot.db.mongodb_helper_fn import get_mongodb_client_from_env
from fastiot.msg.thing import Thing
from pymongo import UpdateOne, MongoClient
from pymongo.results import InsertManyResult, BulkWriteResult

from kio.ml_lifecycle_broker_facade import ok_response_thing, error_response_thing
from kio.ml_lifecycle_subjects_name import (
    DB_SAVE_MANY_RAW_DATAPOINTS_SUBJECT,
    DB_GET_ALL_RAW_DATA_SUBJECT,
    DB_UPSERT_MANY_PROCESSED_DATAPOINTS_SUBJECT,
    DB_GET_PROCESSED_DATA_COUNT_SUBJECT,
    DB_GET_PROCESSED_DATA_PAGE_SUBJECT
)


class DatabaseMongoService(FastIoTService):
    """
        This is the DatabaseMongoService class.

        This service is responsible for storing and retrieving data from the database.
        It implements the CRUD operations (create, read, update, delete) and provides these functionalities to other
        services via the broker.

        Attributes
        ----------
        _DB_NAME : str
            The name of the database.
        _MONGO_DB : str
            The name of the MongoDB database.
        _MONGO_RAW_DATA_COLLECTION : str
            The name of the MongoDB collection for raw data.
        _MONGO_PROCESSED_DATA_COLLECTION : str
            The name of the MongoDB collection for processed data.
        _mongodb_client : MongoClient
            The MongoDB client.
        _db_username : str
            From environment variables inferred username, used to construct the connection string.
        _db_password : str
            From environment variables inferred password for the according user, used to
            construct the connection string.
        _db_port : str
            From environment variables inferred port, used to construct the connection string.
        _db_host : str
            From environment variables inferred host, used to construct the connection string.
        _db : Database
            The MongoDB database.
        _raw_data_collection : Collection
            The MongoDB collection for raw data.
        _processed_data_collection : Collection
            The MongoDB collection for processed data.

        Methods
        -------
        __init__(**kwargs)
            Constructs all the necessary attributes for the DatabaseMongoService object.
        db_save_many_raw_datapoints(topic: str, msg: Thing) -> Thing
            Saves many raw data points to the database.
        db_save_upsert_processed_datapoints(topic: str, msg: Thing) -> Thing
            Upserts many processed data points to the database.
        get_all_raw_data(topic: str, msg: Thing) -> Thing
            Gets all raw data from the database.
        get_processed_data_count(topic: str, msg: Thing) -> Thing
            Gets the count of processed data points in the database.
        get_processed_data_page(topic: str, msg: Thing) -> Thing
            Gets a page of processed data points from the database.
        """

    _DB_NAME = "mongodb"
    _MONGO_DB = "KIOptiPackDb"
    _MONGO_RAW_DATA_COLLECTION = "KIOptiPackRaw"
    _MONGO_PROCESSED_DATA_COLLECTION = "KIOptiPackProcessed"

    def __init__(self, **kwargs):
        """
        Constructs all the necessary attributes for the DatabaseMongoService object.

        Parameters
        ------------
        kwargs : dict
             keyword arguments that are passed to the FastIoTService constructor.
        """
        super().__init__(**kwargs)
        self._db_username = os.environ.get("FASTIOT_MONGO_DB_USERNAME")
        self._db_password = os.environ.get("FASTIOT_MONGO_DB_PASSWORD")
        self._db_port = os.environ.get("FASTIOT_MONGO_DB_PORT")
        self._db_host = os.environ.get("FASTIOT_MONGO_DB_HOST")
        connection_string = f"mongodb://{self._db_username}:{self._db_password}@{self._db_host}:{self._db_port}/?authMechanism=SCRAM-SHA-1"
        self._mongodb_client = MongoClient(connection_string)
        self._db = self._mongodb_client[self._MONGO_DB]

        self._raw_data_collection = self._db[self._MONGO_RAW_DATA_COLLECTION]
        self._processed_data_collection = self._db[self._MONGO_PROCESSED_DATA_COLLECTION]
        # self._trained_model_collection = self._db[self._MONGO_TRAINED_MODEL_COLLECTION]

    @reply(DB_SAVE_MANY_RAW_DATAPOINTS_SUBJECT)
    async def db_save_many_raw_datapoints(self, _: str, msg: Thing) -> Thing:
        """
        Saves many raw data points to the database.

        Parameters
        ----------
        _
            The topic of the message. This is not used in this method.
        msg
            The message that contains the raw data points to be saved to the database.
        Returns
        -------
        Thing
            A Thing object that contains the result of the operation. This is either an acknowledgement or an error.
            Acknowledgements contain the number of raw data points that were saved to the database.
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

        self._logger.info(f"Insering data points into mongodb")
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

        # the native mongo ID is not serializable to json
        # so we convert it to a string
        for data in raw_data_entries:
            data["_id"] = str(data["_id"])

        return ok_response_thing(payload=raw_data_entries, fiot_service=self)

    @reply(DB_GET_PROCESSED_DATA_COUNT_SUBJECT)
    async def get_processed_data_count(self, _: str, __: Thing) -> Thing:
        """
        Gets the count of processed data points in the database.
        Parameters
        ----------
        _
            The topic of the message. This is not used in this method.
        __
            The message that contains the request to get the count of processed data points in the database.

        Returns
        -------
        Thing
            A Thing object that contains the result of the operation. This is either an acknowledgement or an error.
            Acknowledgements contain the count of processed data points in the database.
        """
        self._logger.info("Received request to get the number of processed data points from mongodb")

        try:
            count = self._processed_data_collection.count_documents({})
        except Exception as e:
            self._logger.error(f"Error while counting processed data points in mongodb: {e}")
            return error_response_thing(exception=e, fiot_service=self)

        return ok_response_thing(payload=count, fiot_service=self)

    @reply(DB_GET_PROCESSED_DATA_PAGE_SUBJECT)
    async def get_processed_data_page(self, _: str, msg: Thing) -> Thing:
        """
        Gets a page of processed data points from the database.

        Parameters
        ----------
        _
        msg
            The message that contains the request to get a page of processed data points from the database.

        Returns
        -------
        Thing
            A Thing object that contains the result of the operation. This is either an acknowledgement or an error.
            Acknowledgements contain a page of processed data points from the database.
        """
        self._logger.debug(f"Received request to get a page of processed data points from {self._DB_NAME}")
        default_params = {
            "page": 0,
            "page_size": 10,
        }

        params: dict = msg.value

        # warning if unexpected parameters are present
        for k in params.keys():
            if k not in default_params.keys():
                self._logger.warning(f"Unexpected parameter '{k}' in request. Ignoring it.")

        # merge default and user parameters
        params = {**default_params, **params}

        # check 'page' and 'page_size' are in the params dict
        try:
            if "page" not in params or "page_size" not in params:
                raise ValueError("params must contain 'page' and 'page_size'")
            if params["page"] < 0:
                raise ValueError("page must be >= 0")
            if params["page_size"] < 0:
                raise ValueError("page_size must be >= 0")

            page_documents = self._processed_data_collection.find() \
                .skip(params["page"] * params["page_size"]) \
                .limit(params["page_size"])
            res = [dict(doc) for doc in page_documents]
            # drop the native mongo ID
            for doc in res:
                doc.pop("_id", None)

        except Exception as e:
            self._logger.error(f"Error while counting processed data points in mongodb: {e}")
            return error_response_thing(exception=e, fiot_service=self)

        return ok_response_thing(payload=res, fiot_service=self)


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    DatabaseMongoService.main()
