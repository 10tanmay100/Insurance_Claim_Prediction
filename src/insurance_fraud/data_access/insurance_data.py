from src.insurance_fraud.logger import logging
from src.insurance_fraud.exception import InsuranceFraudException
from src.insurance_fraud.configuration.cassandra_connection import CassandraClient
from src.insurance_fraud.constant.database import *
import os,sys
from src.insurance_fraud.logger import logging


class InsuranceFraudData:
          def __init__(self,):
                    """
                    Help to export data from cassandra as dataframe
                    """
                    try:
                              self.cassandra_df=CassandraClient(cloud_config=CLOUD_CONFIG,auth_provider=AUTH_PROVIDER).get_dataframe()
                    except Exception as e:
                              raise InsuranceFraudException(e,sys) from e
          
          def export_data_as_dataframe(self):
                    try:
                              if "policy_id" in self.cassandra_df.columns.to_list():
                                        df=self.cassandra_df.drop(columns=["policy_id"],axis=1)
                                        return df
                              else:
                                        return df
                    except Exception as e:
                              raise InsuranceFraudException(e,sys) from e







