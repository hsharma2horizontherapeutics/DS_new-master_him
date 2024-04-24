# from tqdm import tqdm
import pyodbc
import sqlalchemy
import urllib
import pymongo
import json
from pathlib import Path, WindowsPath
import pandas as pd
from datetime import datetime
from time import sleep
from fastprogress.fastprogress import master_bar, progress_bar
import hashlib
import re
import numpy as np
from ds.helper_functions import connect
import os

pd.options.display.max_rows = 50

con = connect(mongodb=True)
db = con["apollo"]
pd.set_option('display.float_format', lambda x: '%.f' % x)
collection = db["master"]

fail = []

conn = connect(db="Apollo",server="USLSACASQL1")

def fields():
    # df = pd.read_excel("C:/projects/djosephs/Apollo/apollo_config.xlsx", sheet_name="master_columns")
    df = pd.read_sql("select * from config_master_columns", con=conn)
    return df

# def calculated_fields():
#     df = pd.read_excel("C:/projects/djosephs/Apollo/apollo_config.xlsx", sheet_name="calculated_fields")
#     return df

fields = fields()

def cwd():
    return os.getcwd()

def pull_mongo_data(npi_list=[], colls=[], collection=collection):
    if len(npi_list) == 0:
        return pd.DataFrame()
    else:
        npi_list = [str(x) for x in npi_list]
        master_columns = fields

        mongo_filter = {"npi":{"$in":npi_list}}

        temp_att = []
        for attribute in master_columns[master_columns["attribute"]=="yes"].index:
            temp_att.append(master_columns.loc[attribute,"field_name"])

        if len(colls) == 0:
            colls = list(master_columns["field_name"].to_numpy())
            colls.remove('last_mod_date')
            attributes = temp_att
        else:
            attributes = []
            for att in temp_att:
                if att in colls:
                    attributes.append(att)
        non_att = []
        for coll in colls:
            if coll not in attributes and coll != "npi":
                non_att.append(coll)

        total = collection.count_documents(mongo_filter)

        cursor = collection.find(mongo_filter)
        data = []
        fail = []
        for document in cursor:
            try:
                insert_att = []
                insert_non_att = []
                if "attributes" in document["master"] and document["master"]["attributes"] is not None:
                    for att in attributes:
                        if att in document["master"]["attributes"] and document["master"]["attributes"][att] is not None:
                            if "value" in document["master"]["attributes"][att]:
                                insert_att.append(document["master"]["attributes"][att]["value"])
                            else:
                                insert_att.append(None)
                        else:
                            insert_att.append(None)
                else:
                    insert_att = [None] * len(attributes)
            except:
                print(document["npi"])
            try:

                for non in non_att:
                    insert_non_att.append(document["master"][non]["value"])
                vals = [document["npi"],document["last_mod_date"]] + insert_non_att + insert_att
                data.append(vals)
            except Exception as e:

                fail.append(document["npi"])

        colls = ["npi","last_mod_date"]

        colls += non_att + attributes

        return pd.DataFrame(data, columns=colls)