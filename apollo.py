import ds.helper_functions as hf 
import pandas as pd 
from datetime import datetime

_apollo_conn = hf.connect(db="Apollo",server="USLSACASQL1",engine=True)

_specialties = list(pd.read_sql("SELECT taxonomy_specialty from [Apollo].[dbo].[config_specialties]", con=_apollo_conn)["taxonomy_specialty"])

def _is_in_apollo(npi):

    query = """Select npi from Apollo.dbo.hcp_master where npi = '{0}'""".format(npi)
    res = pd.read_sql(query,_apollo_conn)
    if len(res) == 1:
        return True
    else:
        False

def insert_record(npi, specialty, requested_by, hcp_first_name='', hcp_last_name='', sub_specialty=''):
    if not _is_in_apollo(npi) and (hcp_first_name == '' or hcp_last_name==''):
        raise ValueError("HCP not in apollo, Please provide first and last name")
    elif specialty not in _specialties:
        raise ValueError("Specialty is not recognized: please provide one from taxonomy_specialty on Apollo.dbo.config_specialties")
    else:
        pd.DataFrame(data={
            "npi":[npi],
            "firstname":[hcp_first_name],
            "lastname":[hcp_last_name],
            "specialty":[specialty],
            "sub_specialty":[sub_specialty],
            "taxonomy_code":[pd.read_sql("select taxonomy_code from [Apollo].[dbo].[config_specialties] where taxonomy_specialty='{0}'".format(specialty),con=_apollo_conn).iloc[0].taxonomy_code],
            "requested_by":[requested_by],
            "requested_date":[datetime.now()],
            "processed_date":[None]}).to_sql(name="manual_changes", if_exists="append", con=_apollo_conn, index=False)