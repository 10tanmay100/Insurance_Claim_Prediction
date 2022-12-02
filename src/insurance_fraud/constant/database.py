from cassandra.cluster import Cluster
from cassandra.auth import PlainTextAuthProvider

CLIENT_ID="PMIjYMHgaJINxbWCLwUkpESO"
CLIENT_SECRET="5LDlZ-TMtP0Hw,v9+5mfHZpOwCOt4r,veKDJ8csuGzZzWqEP7C3Q2GHFZ,fSTY5,TQI9Dt6gtjHJ5d2Y3,,C1R_hwMLKUrQ8pYZHeFU5JEAQdS4OsN1TkRAc,U4GvZeu"


CLOUD_CONFIG={'secure_connect_bundle': "F:\Project Ineuron\secure-connect-analytics-vidhya.zip"}

AUTH_PROVIDER = PlainTextAuthProvider(CLIENT_ID,CLIENT_SECRET)