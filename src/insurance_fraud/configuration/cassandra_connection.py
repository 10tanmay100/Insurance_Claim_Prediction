from cassandra.cluster import Cluster
from cassandra.auth import PlainTextAuthProvider
from src.insurance_fraud.exception import InsuranceFraudException
from src.insurance_fraud.utils.main_utils import pandas_factory
from src.insurance_fraud.constant import *
import os,sys
from src.insurance_fraud.logger import logging


class CassandraClient:
          def __init__(self,cloud_config,auth_provider):
                    try:
                              cluster = Cluster(cloud=cloud_config, auth_provider=auth_provider)
                              logging.info("Getting cluster for cassandra..")
                              session = cluster.connect("data")
                              logging.info("Connected with the keyspace..")
                              row = session.execute("SELECT * FROM data.train_data").one()
                              session.row_factory = pandas_factory
                              session.default_fetch_size = None
                              query = "SELECT * FROM data.train_data"
                              logging.info('Getting the query...')
                              rslt = session.execute(query, timeout=None)
                              logging.info("Execute the query to gettig the data from cassandra!!")
                              self.df = rslt._current_rows
                    except Exception as e:
                              logging.error("Issue While connecting with the data!!")
                              raise InsuranceFraudException(e,sys) from e
          def get_dataframe(self):
                    return self.df





