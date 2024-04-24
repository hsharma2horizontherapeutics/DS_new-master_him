import pandas as pd
import numpy as np
from ds.helper_functions import connect
from dateutil.relativedelta import relativedelta
import warnings
warnings.filterwarnings('ignore')
import sys
import ds.mongo_apollo as mg
import re ## for strings

import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style = 'whitegrid')
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
pd.options.mode.chained_assignment = None  # default='warn'
import datetime
from datetime import timedelta

import os
pd.set_option('precision', 0)

conn = connect(db="DS_Claims", server="USLSACASQL2", engine=True)
data_dir = "Z:/temp_data/sql_push/current/"

import upz
# import ds.claims_processing.tep as tep
# import ds.claims_processing.kxx as kxx

def write_extra_tables(email=False):
    new_tep_surgery_table()
    new_kxx_course_therapy_table()
    new_tep_course_therapy_table().to_sql(name="tep_course_therapy", index=False, if_exists="replace",con=conn)
    TreaterReferrer().to_sql(name="tep_treater_referrer", index=False, if_exists="replace",con=conn)
    upz.run()
    #tep.run()
    # kxx.run()

    if email:

        print(email)

def get_tep_features():

    df = pd.read_parquet(data_dir + "tep_patients.parquet")

    df = df.merge(_tep_features(), "left", "patient_id")
    df = df.merge(_hub_payer_info(), "left", "patient_id")
    # df = df.merge(new_tep_course_therapy_table(), "left", "patient_id")

#     listCols = ['ted_dx_date','tep_start_date','hub_enroll_date','tep_proc_first_date','graves_dx_date','e0500_ophtho_dx_date','ted_icd9_dx_date','proptosis_dx_date','vis_impair_dx_date','conf_graves_dx_date']
#     df[listCols] =  df[listCols].apply(pd.to_datetime)


    df = df.reset_index()
    for column in df.columns:
        if "date" in column:
            try:
                df[column] = pd.to_datetime(df[column]).dt.date
            except Exception as e:
                print(e)
                print(column,"failed to date")
    return df

def get_kxx_features():
    df = pd.read_parquet(data_dir + "kxx_patients.parquet")
    return df

def time_diff_patient(x):
    time_diff_list=x.diff()/np.timedelta64(1, 'D')
    return time_diff_list

def new_tep_course_therapy_table():
    #### Calculate Course of Therapy Table TEP (02/16/2021)
    #### Read Claims Data:
    conn = connect(server = 'USLSACASQL2', db='DS_Claims', engine=True)
    query = """SELECT *  FROM tep_claims"""
    tep_claims = pd.read_sql(query, conn)

    ## Add end of claims as column
    tep_claims['svc_date'] = pd.to_datetime(tep_claims['svc_date'])
    tep_claims['end_data_collection'] = [tep_claims['svc_date'].max()]*len(tep_claims)

    ## Keep only infusions claims (procedures Only):
    #tep_claims = tep_claims.loc[(tep_claims['code_group2']=='tep'] (this takes both Rx & procedures)
    tep_claims = tep_claims.loc[(tep_claims['code']=='J3241') | ((tep_claims["code_group3"]=="tep_rx") & (tep_claims["code_category"]!="rx"))] #Only Tep Procedures/claims from dx+proc+remit
    print('There are '+ str(tep_claims['patient_id'].nunique())+' patients w tpp claims')

    #### Get Courses of Therapy:
    ## no duplicated proc code but do have more than one hcp id at the same day
    tep_claims_dt_deduped=tep_claims[['patient_id','svc_date']].drop_duplicates().sort_values(['patient_id' ,'svc_date'])
    tep_claims_dt_deduped['time_diff']=tep_claims_dt_deduped.groupby(['patient_id'])[['svc_date']].apply(lambda x: time_diff_patient(x))
    tep_claims_dt_deduped=tep_claims_dt_deduped.reset_index().drop(['index'], axis=1)
    tep_claims_dt_deduped['course_id']=0

    ##index where we have date gap >60 or switch to another patient
    index_list=tep_claims_dt_deduped[(tep_claims_dt_deduped['time_diff']>60)|(tep_claims_dt_deduped['time_diff'].isnull())].index
    l = index_list.tolist()
    l_mod = [0] + l + [max(l)+1]
    list_of_dfs = [tep_claims_dt_deduped.iloc[l_mod[n]:l_mod[n+1]] for n in range(len(l_mod)-1)]
    for i in range(len(list_of_dfs)):
        list_of_dfs[i]['course_id']=i

    ## Get course treatment STR id
    tep_claims_dt_deduped['course_id_str'] = tep_claims_dt_deduped.apply(lambda x: x['patient_id']+'_'+str(x['course_id']),axis=1)

    #### Aggregate at course of therapy level:
    ## Get first and last date and count per CT:
    tep_ct = tep_claims_dt_deduped.sort_values(by = ['course_id_str','svc_date'], ascending = True).groupby('course_id_str', as_index = False).agg({"patient_id":"first","svc_date":["min", "max", "nunique"]})
    tep_ct.columns = ["_".join(x) for x in tep_ct.columns.ravel()]
    tep_ct.columns = ['course_id_str','patient_id','start_date_ct','end_date_ct','count_tep_infusion_ct']
    tep_ct['start_date_ct'] = pd.to_datetime(tep_ct['start_date_ct']); tep_ct['end_date_ct'] = pd.to_datetime(tep_ct['end_date_ct']);
    tep_ct['total_days_ct'] = (tep_ct['end_date_ct'] - tep_ct['start_date_ct']).astype('timedelta64[D]')
    ## Add NPI to each infusion:
    tep_ct = pd.merge(tep_ct,tep_claims[['patient_id','svc_date','npi','end_data_collection']].rename(columns = {"svc_date":"start_date_ct"}), on = ['patient_id','start_date_ct'], how = 'left')
    tep_ct['end_data_collection'] = pd.to_datetime(tep_ct['end_data_collection'])
    tep_ct['completed_therapy'] = np.ones((len(tep_ct),1))
    tep_ct.loc[(tep_ct['end_data_collection'] - tep_ct['start_date_ct']).astype('timedelta64[D]') < 60,'completed_therapy'] = 0
    print('There are '+ str(tep_ct['patient_id'].nunique())+ ' patients and '+ str(tep_ct['course_id_str'].nunique())+' courses of therapy and length tep_ct: '+ str(len(tep_ct)))

    tep_ct.drop_duplicates(subset = ['course_id_str'], inplace=True)
    writeDf = tep_ct.drop(columns = ['end_data_collection'])
    print('There are '+ str(writeDf['patient_id'].nunique())+ ' patients and '+ str(writeDf['course_id_str'].nunique())+' courses of therapy')

    # writeDf.to_sql(name="tep_course_therapy", index=False, if_exists="replace",con=conn)
    return writeDf

def new_kxx_course_therapy_table():
#### Calculate Course of Therapy Table KXX (11/20/2020)    
    #### Read Claims Data: 
    conn = connect(server = 'USLSACASQL2', db='DS_Claims', engine=True)
    query_pat = """SELECT *  FROM [DS_Claims].[dbo].[kxx_claims]"""
    kxx_claim = pd.read_sql(query_pat, conn)

    ## Add end of claims as column 
    kxx_claim['svc_date'] = pd.to_datetime(kxx_claim['svc_date'])
    kxx_claim['end_data_collection'] = [kxx_claim['svc_date'].max()]*len(kxx_claim)

    ## Keep only infusions claims (procedures Only):
    #kxx_claim = kxx_claim.loc[kxx_claim['code_group2']=='kxx'] (this takes both Rx & procedures)
    kxx_claim = kxx_claim.loc[kxx_claim['code']=='J2507'] ## only KXX procedures
    print('There are '+ str(kxx_claim['patient_id'].nunique())+' patients w kxx claims')
    
    #### Read Immuno Claims data: 
    query_claims = """SELECT *  FROM [DS_Claims].[dbo].[kxx_detail_immuno_claims]"""
    immuno_claims = pd.read_sql(query_claims, conn)
    print('There are '+ str(immuno_claims['patient_id'].nunique())+ ' patients with Immuno')
    immuno_claims['immuno_binary'] = np.ones((len(immuno_claims),1))

    ## Drop duplicates same immuno, same day, same patient
    immuno_claims.drop_duplicates(subset = ['patient_id','svc_date','ndc'], inplace = True)
    
    #### Get Courses of Therapy:
    ##no duplicated proc code but do have more than one hcp id at the same day
    kxx_claim_dt_deduped=kxx_claim[['patient_id','svc_date']].drop_duplicates().sort_values(['patient_id' ,'svc_date'])
    kxx_claim_dt_deduped['time_diff']=kxx_claim_dt_deduped.groupby(['patient_id'])[['svc_date']].apply(lambda x: time_diff_patient(x))
    kxx_claim_dt_deduped=kxx_claim_dt_deduped.reset_index().drop(['index'], axis=1)
    kxx_claim_dt_deduped['course_id']=0
    
    ##index where we have date gap >60 or switch to another patient
    index_list=kxx_claim_dt_deduped[(kxx_claim_dt_deduped['time_diff']>60)|(kxx_claim_dt_deduped['time_diff'].isnull())].index
    l = index_list.tolist()
    l_mod = [0] + l + [max(l)+1]
    list_of_dfs = [kxx_claim_dt_deduped.iloc[l_mod[n]:l_mod[n+1]] for n in range(len(l_mod)-1)]
    for i in range(len(list_of_dfs)):
        list_of_dfs[i]['course_id']=i
    
    ## Get course treatment STR id
    kxx_claim_dt_deduped['course_id_str'] = kxx_claim_dt_deduped.apply(lambda x: x['patient_id']+'_'+str(x['course_id']),axis=1)

    #### Aggregate at course of therapy level:
    ## Get first and last date and count per CT: 
    kxx_ct = kxx_claim_dt_deduped.sort_values(by = ['course_id_str','svc_date'], ascending = True).groupby('course_id_str', as_index = False).agg({"patient_id":"first","svc_date":["min", "max", "nunique"]})
    kxx_ct.columns = ["_".join(x) for x in kxx_ct.columns.ravel()]
    kxx_ct.columns = ['course_id_str','patient_id','start_date_ct','end_date_ct','count_kxx_vials_ct']
    kxx_ct['start_date_ct'] = pd.to_datetime(kxx_ct['start_date_ct']); kxx_ct['end_date_ct'] = pd.to_datetime(kxx_ct['end_date_ct']); 
    kxx_ct['total_days_ct'] = (kxx_ct['end_date_ct'] - kxx_ct['start_date_ct']).astype('timedelta64[D]')

    ## Add NPI to each infusion: 
    kxx_ct = pd.merge(kxx_ct,kxx_claim[['patient_id','svc_date','npi','end_data_collection']].rename(columns = {"svc_date":"start_date_ct"}), on = ['patient_id','start_date_ct'], how = 'left')
    kxx_ct['end_data_collection'] = pd.to_datetime(kxx_ct['end_data_collection'])
    kxx_ct['completed_therapy'] = np.ones((len(kxx_ct),1))
    kxx_ct.loc[(kxx_ct['end_data_collection'] - kxx_ct['start_date_ct']).astype('timedelta64[D]') < 60,'completed_therapy'] = 0
    print('There are '+ str(kxx_ct['patient_id'].nunique())+ ' patients and '+ str(kxx_ct['course_id_str'].nunique())+' courses of therapy and length kxx_ct: '+ str(len(kxx_ct)))
    
    #### Get Closest Immuno 60D before/after 1st Infusion: 
    immuno_claims2 = pd.merge(immuno_claims[['patient_id','svc_date','ndc','brand']].drop_duplicates(subset = ['patient_id','svc_date']).rename(columns= {"svc_date":"immuno_date"}),kxx_ct[['patient_id','course_id_str','start_date_ct']], on = 'patient_id')
    immuno_claims2['immuno_date'] = pd.to_datetime(immuno_claims2['immuno_date']); immuno_claims2['start_date_ct'] = pd.to_datetime(immuno_claims2['start_date_ct']) 
    immuno_claims2['days_immuno_kxx'] = (immuno_claims2['immuno_date'] - immuno_claims2['start_date_ct']).astype('timedelta64[D]')
    immuno_claims2['abs_days_immuno_kxx'] = immuno_claims2['days_immuno_kxx'].abs()

    ## Get closest immuno to start date of Course of Therapy
    immuno_ct = immuno_claims2.sort_values(by = ['course_id_str','start_date_ct','abs_days_immuno_kxx'], ascending = True).drop_duplicates(subset = ['course_id_str'], keep = 'first')
    immuno_ct.drop_duplicates(subset = ['course_id_str'], inplace = True)
    
    ## Merge back into main Course of Therapy Table:
    kxx_ct2 = pd.merge(kxx_ct,immuno_ct[['course_id_str','immuno_date','days_immuno_kxx','abs_days_immuno_kxx','ndc','brand']], on = ['course_id_str'], how = 'left')
    print('There are '+ str(kxx_ct2['patient_id'].nunique())+ ' patients and '+ str(kxx_ct2['course_id_str'].nunique())+' courses of therapy and length kxx_ct2: '+ str(len(kxx_ct2)))

    ## Get qualifying immuno:
    kxx_ct2['qualifying_immuno'] = np.zeros((len(kxx_ct2),1))
    kxx_ct2['immuno_date'] = pd.to_datetime(kxx_ct2['immuno_date']); kxx_ct2['end_date_ct'] = pd.to_datetime(kxx_ct2['end_date_ct']) 
#     kxx_ct2.loc[(kxx_ct2['abs_days_immuno_kxx']<=60)&(kxx_ct2['immuno_date'].notna()),'qualifying_immuno']=1
    kxx_ct2.loc[(kxx_ct2['immuno_date']<kxx_ct2['end_date_ct'])&(kxx_ct2['abs_days_immuno_kxx']<=60)&(kxx_ct2['immuno_date'].notna()),'qualifying_immuno']=1 ## New definition

    kxx_ct2.drop_duplicates(subset = ['course_id_str'], inplace = True)
    print('There are '+ str(kxx_ct2['patient_id'].nunique())+ ' patients and '+ str(kxx_ct2['course_id_str'].nunique())+' courses of therapy and length kxx_ct2: '+ str(len(kxx_ct2)))
    ## Add Start Month:
    kxx_ct2['start_month_ct'] = pd.to_datetime(kxx_ct2['start_date_ct']).dt.to_period('M')
    ## Add Start Type (get 1st ct date per patient)
    firstCT = kxx_ct2.sort_values(by = ['patient_id','start_date_ct','course_id_str']).drop_duplicates(subset = ['patient_id'])[['course_id_str','patient_id','start_date_ct']]
    kxx_ct2['start_type'] = 'restart'
    kxx_ct2.loc[kxx_ct2['course_id_str'].isin(firstCT['course_id_str'].unique()),'start_type'] = 'start'
    
    #### Get final DF (drop irrelevant columns)
    writeDf = kxx_ct2.drop(columns = ['end_data_collection','immuno_date','abs_days_immuno_kxx'])
    print('There are '+ str(writeDf['patient_id'].nunique())+ ' patients and '+ str(writeDf['course_id_str'].nunique())+' courses of therapy and : '+ str(writeDf.loc[writeDf['days_immuno_kxx'].notna(),'patient_id'].nunique())+' patients with Immuno Overall and '+ str(writeDf.loc[writeDf['qualifying_immuno']==1,'patient_id'].nunique())+' patients with Immuno 60D before/after KXX')    
    writeDf['start_month_ct'] = writeDf['start_month_ct'].astype(str)
    
    writeDf.to_sql(name="kxx_course_therapy", index=False, if_exists="replace",con=conn)
    
def new_tep_surgery_table():
    ###### Read in Surgery Table: 
    pathIn = 'Z:/Projects/Adhoc/Claudia/tep_patient_journey/input/'
    surgeryTep = pd.read_csv(pathIn + 'ted_surgery_code_groups.csv')
    surgeryTep.to_sql(name="tep_surgery_type_table", index=False, if_exists="replace",con=conn)
    

# for tep

def get_hub_data():
    conn = connect(db='Tepro', server='USLSACASQL2')

    query = """SELECT *
    FROM [Tepro].[dbo].[Tep_patient]
    """
    tep_hub = pd.read_sql(query,conn)

    tep_hub['enrollment_date']=pd.to_datetime(tep_hub['enrollment_date'])
    tep_hub['infusion_1_date']=pd.to_datetime(tep_hub['infusion_1_date'])
    tep_hub['scheduled_start_date']=pd.to_datetime(tep_hub['scheduled_start_date'])

    # contact info at veeva
    conn = connect(db='Tepro_Veeva', server='USLSACASQL2')

    query = """ SELECT a.* , ci.Account_MVN__c, ci.State_MVN__c, ci.Zip_MVN__c, ci.Address_MVN__c
    FROM [Tepro_Veeva].[dbo].[veeva_ib_accounts] AS a 
    JOIN [Tepro_Veeva].[dbo].[veeva_ib_contact_info] AS ci ON a.Id = ci.Account_MVN__c
    JOIN [Tepro_Veeva].[dbo].[veeva_ib_record_type] AS rt ON a.RecordTypeId = rt.Id
    WHERE rt.Name = 'Patient'
    AND ci.Address_MVN__c IS NOT NULL
    """
    veeva_contact = pd.read_sql(query,conn)

    # get zip3
    veeva_contact['zip3']=veeva_contact['Zip_MVN__c'].str[0:3]


    # tep veeva_id
    conn = connect(db='Tepro_Veeva', server='USLSACASQL2')

    query = """SELECT *
    FROM [Tepro_Veeva].[dbo].[veeva_ib_program_member]
    """
    veeva_id = pd.read_sql(query,conn)

    # get zip3 with name
    veeva_info=veeva_id[['Name','Member_MVN__c']].merge(veeva_contact[['Id','State_MVN__c', 'zip3']], \
                                                        left_on='Member_MVN__c', right_on='Id', how='left').drop_duplicates()

    # hub data with zip3

    tep_hub_data=tep_hub[['patient_id','product_name','enrollment_date','aging','aging_name','hcp_id','treating_hcp_id','prescribing_hcp_id',\
                        'enrolling_hcp_id','dob','age','gender','state','primary_payer','primary_paytype','secondary_payer','secondary_paytype',\
                        'scheduled_start_date','infusion_1_date','infusion_2_date','infusion_3_date','infusion_4_date','infusion_5_date','infusion_6_date',\
                        'infusion_7_date','infusion_8_date']]\
                        .merge(veeva_info[['Name','State_MVN__c','zip3']], left_on=['patient_id'], right_on=['Name'], how='left')

    tep_hub_data['dob_year']=pd.to_datetime(tep_hub_data["dob"]).dt.year
    tep_hub_data['dob_year'] = tep_hub_data['dob_year'].fillna(0)
    tep_hub_data['dob_year'] = tep_hub_data['dob_year'].apply(lambda x: str(int(x)) if x != 0 else None)

    tep_hub_data.rename(columns={'hcp_id':'npi', 'treating_hcp_id':'treating_npi','prescribing_hcp_id':'prescribing_npi',\
                                'enrolling_hcp_id':'enrolling_npi'}, inplace=True)

    tep_hub_data.columns = ['hub_'+ str(col) for col in tep_hub_data.columns]

    return tep_hub_data

def getSteroidStrength(claims): 
    
    ## Get Unique NDCs: 
    steroidNames = claims.loc[claims['code_group2'].isin(['iv_steroid','oral_steroid'])].drop_duplicates(subset = ['code_desc','strength','ndc'])[['code','code_desc','strength','code_group2','ndc','brand','qty','days_supply','admin']].reset_index(drop = True)
    print('There are '+ str(len(steroidNames))+ ' steroid types (NDCs & code descriptions)')

    ### Break into pack, pills, solutions (to modify strings accordingly):
    pack = ['TABLET THERAPY PACK']
    pills = ['TABLET', 'TABLET DISPERSIBLE', 'SOLUTION RECONSTITUTED','TABLET ENTERICCOATED']
    solutions = ['SYRUP', 'SOLUTION','SUSPENSION','CONCENTRATE']
    
    ## Check that there are no other admin modes: 
    li1 = list(steroidNames['admin'].unique())
    li2 = ['TABLET THERAPY PACK','TABLET', 'TABLET DISPERSIBLE', 'SOLUTION RECONSTITUTED','TABLET ENTERICCOATED','SYRUP', 'SOLUTION','SUSPENSION','CONCENTRATE']
    print('Number of Missing steroids administration types (should be <=1): '+ str(len(list(set(li1) - set(li2)))))
    print('Missing steroids administration types should only contain "None": '+ str(list(set(li1) - set(li2))))
    
    ### Get Strength Num for packs
    steroid_pack = steroidNames.loc[(steroidNames['admin'].isin(pack))&(steroidNames['code_group2']=='oral_steroid')]
    ## Remove all characters after paranthesis (qty)
    steroid_pack['strength_num'] = steroid_pack['code_desc'].str.split('(').str[0]
    # Only keep numbers (strength):
    steroid_pack['strength_num'] = steroid_pack['strength_num'].map(lambda x: re.sub("[^0-9]", "", x))
    steroid_pack.drop_duplicates(subset = ['code_desc'])

    ### Get Strength Num for pills (MG)
    steroid_mg = steroidNames.loc[(steroidNames['admin'].isin(pills))&(steroidNames['code_group2']=='oral_steroid')]
    steroid_mg['strength_num'] = steroid_mg['strength'].str.split(' ').str[0]
    steroid_mg.drop_duplicates(subset = ['strength'])

    steroidNames2 = pd.concat([steroid_pack,steroid_mg],ignore_index = True) ## Merge packs & pills

    ### Get Strength Num for solutions: 
    steroid_sol = steroidNames.loc[(steroidNames['admin'].isin(solutions))&(steroidNames['code_group2']=='oral_steroid')]
    steroid_sol['mg'] = steroid_sol['strength'].str.split('/').str[0]
    steroid_sol['mg'] = steroid_sol['mg'].map(lambda x: re.sub("[^.0-9]", "", x))
    steroid_sol['mg'] = steroid_sol['mg'].astype(np.float64)
    steroid_sol['mL'] = steroid_sol['strength'].str.split('/').str[1] ## split MG/ML (obtain ML)
    steroid_sol['mL'] = steroid_sol['mL'].map(lambda x: re.sub("[^0-9]", "", x)) ## convert to number
    steroid_sol['mL'] = steroid_sol['mL'].replace(r'^\s*$', 1, regex=True) ## Replace empty string with 1 (per 1ML)
    steroid_sol['mL'] = steroid_sol['mL'].astype(np.float64)
    steroid_sol['strength_num'] = np.round(steroid_sol['mg']/steroid_sol['mL'],2)
    steroid_sol.drop_duplicates(subset = ['strength'])

    steroidNames2 = pd.concat([steroidNames2,steroid_sol.drop(columns=['mg','mL'])],ignore_index = True) ## Merge packs, pills & solution

    ### Get Strength Num for IV Steroids: 
    steroid_iv = steroidNames.loc[steroidNames['code_group2']=='iv_steroid']
    steroid_iv['strength_num'] = steroid_iv['code_desc'].map(lambda x: re.sub("[^.0-9]", "", x))
    steroid_iv['strength_num'] = steroid_iv['strength_num'].replace(r'^\s*$', np.nan, regex=True) ## Replace empty string with np.nan (strength)

    steroid_iv.drop_duplicates(subset = ['code_desc','code'])

    steroidNames2 = pd.concat([steroidNames2,steroid_iv], ignore_index = True) ## Merge packs, pills, solution, and IVs
    steroidNames2['strength_num'] = steroidNames2['strength_num'].astype(np.float64)

    return steroidNames2

def getSpecialties_fromNPI(df_npi):
    ## Given df_npi get npi codes 
    npiCodes = list(df_npi['npi'].unique())

    ## Get npis: 
    dfApollo = mg.pull_mongo_data(npiCodes)
    print('Found '+ str(dfApollo['npi'].nunique()) + ' NPIs in Apollo from '+ str(len(npiCodes))+ ' NPIs in claims')

    ## Get columns of interest: 
    cols = ['npi','specialty_group','sub_specialty','specialty']
    df_npi = pd.merge(df_npi, dfApollo[cols], how = 'left', on = 'npi')
    return df_npi

# def getTEDprofile(df): ## input patient table
#     listCols = ['graves_dx_date','e0500_ophtho_dx_date','ted_icd9_dx_date','proptosis_dx_date','vis_impair_dx_date','conf_graves_dx_date']
#     df[listCols] =  df[listCols].apply(pd.to_datetime)
#     df.loc[df["graves_dx_date"].notnull(),"profile"] = "graves only"
#     df.loc[df["e0500_ophtho_dx_date"].notnull(),"profile"] = "E05000 from optho"
#     df.loc[(df["vis_impair_dx_date"]-df["conf_graves_dx_date"]).dt.days>-1095,"profile"] = "profile 3"
#     df.loc[(df["proptosis_dx_date"]-df["graves_dx_date"]).dt.days>-1095,"profile"] = "profile 2"
#     df.loc[df["ted_icd9_dx_date"].notnull(),"profile"] = "profile 1"
#     return df

def getTimeFromTEP(pat_tz,claims_tz):
    ### Get time since 1st Claim --> Tepezza date: 
    pat_tz['first_TEP'] = pd.to_datetime(pat_tz['first_TEP']);                pat_tz['first_claim_date'] = pd.to_datetime(pat_tz['first_claim_date']);  
    pat_tz['hub_enroll_date'] = pd.to_datetime(pat_tz['hub_enroll_date']);    pat_tz['last_claim_date'] = pd.to_datetime(pat_tz['last_claim_date']);
    
    pat_tz['mo_FirstClaimToTep'] = np.round(((pat_tz['first_TEP'] - pat_tz['first_claim_date']).astype('timedelta64[D]'))/30,2)
    pat_tz['mo_LastClaimToTep'] = np.round(((pat_tz['first_TEP'] - pat_tz['last_claim_date']).astype('timedelta64[D]'))/30,2) 
    
    ### Get earliest date between Hub & Tepezza:
    pat_tz['earliest_hub_tep'] = pat_tz[['first_TEP','hub_enroll_date']].min(axis = 1); pat_tz['earliest_hub_tep'] = pd.to_datetime(pat_tz['earliest_hub_tep']);  
    
    ### Get time since from 1st TED dx --> Tepezza/Hub date: 
    pat_tz['ted_dx_date'] = pd.to_datetime(pat_tz['ted_dx_date']);     
    pat_tz['mo_FirstTEDToHub'] = np.round(((pat_tz['earliest_hub_tep'] - pat_tz['ted_dx_date']).astype('timedelta64[D]'))/30,2)

    ### Get time from entering hub/tep to 1st/last claim and 1st eye code:
    pat_tz['mo_FirstClaimToHub'] = np.round(((pat_tz['earliest_hub_tep'] - pat_tz['first_claim_date']).astype('timedelta64[D]'))/30,2) 
    pat_tz['mo_LastClaimToHub'] = np.round(((pat_tz['earliest_hub_tep'] - pat_tz['last_claim_date']).astype('timedelta64[D]'))/30,2) 
    

    ### Get time since 1st Eye Code --> Tepezza date: 
    pat_tz['proptosis_dx_date'] = pd.to_datetime(pat_tz['proptosis_dx_date']); pat_tz['eye_sympt_dx_date'] = pd.to_datetime(pat_tz['eye_sympt_dx_date']); pat_tz['vis_impair_dx_date'] = pd.to_datetime(pat_tz['vis_impair_dx_date'])
    pat_tz['earliestEyeCode'] = pat_tz[['proptosis_dx_date','eye_sympt_dx_date','vis_impair_dx_date']].min(axis = 1)
    pat_tz['mo_FirstEyeCodeToTep'] = np.round(((pat_tz['first_TEP'] - pat_tz['earliestEyeCode']).astype('timedelta64[D]'))/30,2)
    pat_tz['mo_FirstEyeCodeToHub'] = np.round(((pat_tz['earliest_hub_tep'] - pat_tz['earliestEyeCode']).astype('timedelta64[D]'))/30,2)
    
    ### Get time since 1st Graves Code --> Tepezza date: 
    pat_tz['graves_dx_date'] = pd.to_datetime(pat_tz['graves_dx_date']); 
    pat_tz['mo_FirstGravesToTep'] = np.round(((pat_tz['first_TEP'] - pat_tz['graves_dx_date']).astype('timedelta64[D]'))/30,2)

    ### Create categories for each 'time from xx'--> Tepezza:
    labelsTime = ['dx_after_first_tep','0-1 mo','1-3 mo','4-5 mo','6-12 mo','1-2 yrs','2-3 yrs','3-4 yrs','4-5 yrs','>5 yrs']; 
    #timeBucket = [-100,0,1,4,6,12,24,36,48,60,100]
    timeBucket = [-100,-0.02,1,4,6,12,24,36,48,60,100] ## 0 should not be 'dx_after_tep_start_date' 
    pat_tz['cat_FirstEyeCodeToTep'] = pd.cut(pat_tz['mo_FirstEyeCodeToTep'],timeBucket, labels=labelsTime)
    pat_tz['cat_FirstGravesToTep'] = pd.cut(pat_tz['mo_FirstGravesToTep'],timeBucket, labels= labelsTime)
    pat_tz['cat_FirstClaimToTep'] = pd.cut(pat_tz['mo_FirstClaimToTep'],timeBucket, labels= labelsTime)
    pat_tz['cat_FirstEyeCodeToHub'] = pd.cut(pat_tz['mo_FirstEyeCodeToHub'],timeBucket, labels= labelsTime)
    
    ### Create Acute vs. Chronic Label:
    pat_tz['acute_chronic_label'] = np.nan
    pat_tz.loc[pat_tz['mo_FirstTEDToHub']<0,'acute_chronic_label'] = 'dx_after_tep_hub'
    pat_tz.loc[(pat_tz['mo_FirstTEDToHub']>=0)&(pat_tz['mo_FirstTEDToHub']<=24),'acute_chronic_label'] = 'acute (0-2 years)'
    pat_tz.loc[(pat_tz['mo_FirstTEDToHub']>24)&(pat_tz['mo_FirstTEDToHub']<36),'acute_chronic_label'] = 'acute (2-3 years)'
    pat_tz.loc[pat_tz['mo_FirstTEDToHub']>=36,'acute_chronic_label'] = 'chronic (3+ years)'
    pat_tz.loc[pat_tz['ted_dx_date'].isna(),'acute_chronic_label'] = 'missing_ted_dx' 
    pat_tz.loc[pat_tz['earliest_hub_tep'].isna(),'acute_chronic_label'] = 'missing_tep_hub'
    
    ### Create Prevalent/Incident Label: 
    pat_tz['incident_prevalent_label'] = np.nan
    pat_tz.loc[pat_tz['mo_FirstTEDToHub']<0,'incident_prevalent_label'] = 'dx_after_tep_hub'
    pat_tz.loc[(pat_tz['mo_FirstTEDToHub']>=0)&(pat_tz['mo_FirstTEDToHub']<=12),'incident_prevalent_label'] = 'Incident'
    pat_tz.loc[pat_tz['mo_FirstTEDToHub']>12,'incident_prevalent_label'] = 'Prevalent'
    pat_tz.loc[pat_tz['ted_dx_date'].isna(),'incident_prevalent_label'] = 'missing_ted_dx'
    pat_tz.loc[pat_tz['earliest_hub_tep'].isna(),'incident_prevalent_label'] = 'missing_tep_hub'

    ##### Claims Table: Merge 1st TEP date & label acute/chronic
    claims_tz = pd.merge(claims_tz, pat_tz[['patient_id','first_TEP','earliest_hub_tep','acute_chronic_label','incident_prevalent_label']], on = 'patient_id', how = 'left')

    ## Compute time from each claim to Tep: 
    labelsTime = ['claim_after_first_tep','0-1 mo','1-3 mo','4-5 mo','6-12 mo','1-2 yrs','2-3 yrs','3-4 yrs','4-5 yrs','>5 yrs']; 
    claims_tz['first_TEP'] = pd.to_datetime(claims_tz['first_TEP']);claims_tz['svc_date'] = pd.to_datetime(claims_tz['svc_date']);claims_tz['earliest_hub_tep'] = pd.to_datetime(claims_tz['earliest_hub_tep'])
    
    claims_tz['mo_claimToTep'] = np.round(((claims_tz['earliest_hub_tep'] - claims_tz['svc_date']).astype('timedelta64[D]'))/30,2)
    claims_tz['cat_claimToTep'] = pd.cut(claims_tz['mo_claimToTep'],timeBucket, labels = labelsTime)
    
    ### NOTE 5/3/2020: No longer need to calculate Graves E0500, Graves E0500 from Optho, TED Dx, and Profile locally (calculated in the cloud)
#     ### Add in the patient table Graves E0500, and Graves E0500 from Optho: e0500_dx_date; e0500_ophtho_dx_date
#     gravesE0500 = claims_tz.loc[claims_tz['code']=='E0500'].groupby('patient_id',as_index = False).agg({"svc_date":"min"}).rename(columns = {"svc_date":"e0500_dx_date"})
#     pat_tz = pd.merge(pat_tz,gravesE0500, on = 'patient_id',how ='left')
    
#     gravesE0500_optho = claims_tz.loc[(claims_tz['code']=='E0500')&(claims_tz['specialty_group']=='OPHTHALMOLOGY')].groupby('patient_id',as_index = False).agg({"svc_date":"min"}).rename(columns = {"svc_date":"e0500_ophtho_dx_date"})
#     pat_tz = pd.merge(pat_tz,gravesE0500_optho, on = 'patient_id',how ='left')
    
    ## Get profiles: 
    # pat_tz = getTEDprofile(pat_tz)
    return [pat_tz,claims_tz]

def _tep_features():
    pathIn = 'Z:/temp_data/sql_push/current/'

    #### Read data: 
    ## patient&claims table for both Hub patients 
    pat_tz = pd.read_parquet(pathIn + 'tep_patients.parquet')
    claims_tz = pd.read_parquet(pathIn + 'tep_claims.parquet')

    ## hcp_tz = contains NPIs for HCPs prescribing TEP, matched_hub = contains flag for patients matched in the hub (from Lihong fuzzy match)
    hcp_tz = pd.read_parquet(pathIn + 'tep_detail_providers.parquet')

    ## surgeryTypes = spreadsheet where surgeries were categorized; steroidTypes = spreadsheet where steroids were categorized (IV/low/high)
#     surgeryTypes = pd.read_csv('Z:/Projects/Adhoc/Claudia/tep_patient_journey/input/augustData/surgeriesCodes_augustPat_v1.csv')
    surgeryTypes = pd.read_csv('Z:/Projects/Adhoc/Claudia/tep_patient_journey/input/ted_surgery_code_groups.csv')
    pat_tz['patient_id'] = (pat_tz['patient_id'].astype(np.int64)).astype(str)

    ## Get all unique NDCs for each type of steroid
    steroidNames = getSteroidStrength(claims_tz)

    ## Add steroid types & strength back to claims:
    claims_tz2 = pd.merge(claims_tz, steroidNames[['code_desc','strength','ndc','strength_num']],how = 'left',on = ['code_desc','strength','ndc'])
    print('There are '+ str(len(claims_tz2.loc[claims_tz2['code_group2'].isin(['iv_steroid', 'oral_steroid'])].drop_duplicates(subset =['code_desc','strength','ndc']))) + ' types of steroids (NDCs)')
    claims_tz2.sort_values(by = ['patient_id','svc_date'], ascending = True, inplace = True)

    ## Change qty & Days_supply for IV to 1: 
    claims_tz2.loc[(claims_tz2['code_group2']=='iv_steroid')&(claims_tz2['qty'].isna())&(claims_tz2['strength_num'].notna()),'qty'] =1
    claims_tz2.loc[(claims_tz2['code_group2']=='iv_steroid')&(claims_tz2['days_supply'].isna())&(claims_tz2['strength_num'].notna()),'days_supply'] =1

    ## Get overall dose per prescription & Dose per Day: 
    claims_tz2['qty'] = claims_tz2['qty'].astype(np.float64);    claims_tz2['days_supply'] = claims_tz2['days_supply'].astype(np.float64)
    claims_tz2['dose_prescription_overall'] = claims_tz2['strength_num']*claims_tz2['qty']
    claims_tz2['dose_prescription_day'] = claims_tz2['dose_prescription_overall']/claims_tz2['days_supply']

    ## Add Dose Type: 
    claims_tz2['dose_type'] = np.nan
    claims_tz2.loc[(claims_tz2['code_group2']=='oral_steroid')&(claims_tz2['strength_num']>=20),'dose_type'] = 'high oral' ## Keep high as >=20 mg
    claims_tz2.loc[(claims_tz2['code_group2']=='oral_steroid')&(claims_tz2['strength_num']<20),'dose_type'] = 'low oral'
    claims_tz2.loc[(claims_tz2['code_group2']=='iv_steroid'),'dose_type'] = 'IV'

    ## Add Surgery types to Claims: 
    li1 = list(claims_tz2.loc[claims_tz2['code_group2'] == 'surgery','code_desc'].unique()) ## list of all the surgery types in Claims
    li2 = list(surgeryTypes['code_desc'].unique()) ## list of all surgery types in spreadsheet
    print('Check that there are no steroid codes in claims not accounted for in the spreadsheet: '+ str(len(list(set(li1) - set(li2)))))
    claims_tz3 = pd.merge(claims_tz2,surgeryTypes[['code_desc','surgery_type','surgery_type2']],on = ['code_desc'],how = 'left')
    
    ## Add Specialties and sub-specialties from Apollo
    claims_tz4 = getSpecialties_fromNPI(claims_tz3)

    ## Define first_tep from IQVIA vs Hub:
    pat_tz['first_TEP_hub'] = pd.to_datetime(pat_tz['hub_therapy_start_date']);
    pat_tz['tep_proc_first_date'] = pd.to_datetime(pat_tz['tep_proc_first_date']) ## already had this calculated from cloud (it's first tepezza from iqvia claims)

    ## First TEPEZZA is the IQVIA one, unless that's missing (then replace with first_tep_Hub if present):
    pat_tz['first_TEP'] = pat_tz['tep_proc_first_date']
    pat_tz.loc[pat_tz['tep_proc_first_date'].isna(),'first_TEP'] = pat_tz.loc[pat_tz['tep_proc_first_date'].isna(),['tep_proc_first_date','first_TEP_hub']].min(axis = 1)

    ## Also get the month and quarter of the 1st TEP: 
    pat_tz['first_TEP_month'] = pd.to_datetime(pat_tz['first_TEP']).dt.to_period('M')
    pat_tz['first_TEP_quarter'] = pd.to_datetime(pat_tz['first_TEP']).dt.quarter

    ## Add time from dx --> TEP: 
    pat_tz2,claims_tz5 = getTimeFromTEP(pat_tz,claims_tz4)
    print('Patients with TEPEZZA: '+ str(pat_tz2.loc[pat_tz2['first_TEP'].notna(),'patient_id'].nunique())+ ' from total ' + str(pat_tz2['patient_id'].nunique()))

    ## Add 1st TEP hub Date (Month): 
    pat_tz2['first_TEPHub_month'] = pd.to_datetime(pat_tz2['earliest_hub_tep']).dt.to_period('M')

    ####### Qualifying patients ###################################################################### 
    ## Getting patients with TEP/Hub before last claim: 
    pat_tz2['first_tepHub_before_claims_end'] = np.zeros((len(pat_tz2),1))
    lastMonthClaims = pd.to_datetime(claims_tz5['svc_date']).dt.to_period('M').max()
    pat_tz2.loc[pat_tz2['first_TEPHub_month']<lastMonthClaims+1,'first_tepHub_before_claims_end']=1
    print('First Month TEP: '+ str(claims_tz.loc[claims_tz['code_group2']=='tep','svc_date'].min()))
    print('Last month claims: '+ str(pd.to_datetime(claims_tz['svc_date']).dt.to_period('M').max()))

    ## Getting patients with > 6 mo claims history: 
    pat_tz2['6mo_claims_history'] = np.zeros((len(pat_tz2),1))
    pat_tz2.loc[pat_tz2['mo_FirstClaimToHub']>6,'6mo_claims_history']=1

    ## Getting patients with last claim < 3 months before 1st TEP: 
    pat_tz2['active_3mo_before_tep'] = np.zeros((len(pat_tz2),1))
    pat_tz2.loc[pat_tz2['mo_LastClaimToHub'] < 3,'active_3mo_before_tep']=1

    print('=='*50); print();
    print('Patients TEP/hub before last claims month: '+ str(pat_tz2.loc[pat_tz2['first_tepHub_before_claims_end']==1,'patient_id'].nunique()))
    print('Patients with at least 6 mo claims: '+ str(pat_tz2.loc[pat_tz2['6mo_claims_history']==1,'patient_id'].nunique()))
    print('Patients active 3 mo before TEP: '+ str(pat_tz2.loc[pat_tz2['active_3mo_before_tep']==1,'patient_id'].nunique()))
    print();print('=='*50);

    print('Patients with TEPEZZA before last claim: '+ str(pat_tz2.loc[(pat_tz2['first_tepHub_before_claims_end']==1),'patient_id'].nunique())); 
    print('Patients with TEPEZZA before last claim & > 6mo claims: '+ str(pat_tz2.loc[(pat_tz2['first_tepHub_before_claims_end']==1) & (pat_tz2['6mo_claims_history']==1),'patient_id'].nunique()));
    print('Patients with TEPEZZA before last claim & > 6mo claims & last claim < 3 mo before TEP: '+ str(pat_tz2.loc[(pat_tz2['first_tepHub_before_claims_end']==1)&(pat_tz2['6mo_claims_history']==1)&(pat_tz2['active_3mo_before_tep']==1),'patient_id'].nunique()))
    print();print('=='*50);

    ## Qualifying patients = 1st TEP before last claim & >6 mo claims history & last claim <3 mo from 1st TEP - modified 4/27 (to include patients with 1st hub date/tepezza before end claims)
    pat_tz3 = pat_tz2.copy()
    pat_tz3.loc[(pat_tz3['first_tepHub_before_claims_end']==1)&(pat_tz3['6mo_claims_history']==1)&(pat_tz3['active_3mo_before_tep']==1),"qualifying_tep_patient"]=1
    pat_tz3.loc[pat_tz3["qualifying_tep_patient"].isna(),"qualifying_tep_patient"]=0
    print('There are '+ str(pat_tz3.loc[pat_tz3['qualifying_tep_patient']==1,'patient_id'].nunique())+
          ' qualifying patients from '+ str(pat_tz3.loc[pat_tz3['earliest_hub_tep'].notna(),'patient_id'].nunique())+' TEP/Hub patients')
    
    ## Adding Oculoplastic Label:'Oculoplastics' or 'Other Oculofacial and Plastics' 
    seen_oculo = claims_tz5[claims_tz5["sub_specialty"].isin(['Oculoplastics','Other Oculofacial and Plastics'])][["patient_id"]].drop_duplicates()
    seen_oculo["seen_oculo"] = 1
    pat_tz3 = pat_tz3.merge(seen_oculo,"left","patient_id")

    ## Only Claims pre-TEPEZZA: 
    claims_pre_tz = claims_tz5.loc[pd.to_datetime(claims_tz5['svc_date'])<=pd.to_datetime(claims_tz5['earliest_hub_tep'])]
    print('Patients with Claims Pre-Hub: '+ str(claims_pre_tz['patient_id'].nunique()))
    
    print(claims_pre_tz)
    ## Adding patients with low oral/high oral/IV steroids (pre-Hub)
    doses = claims_pre_tz[claims_pre_tz.dose_type.isin(['low oral','high oral','IV'])][["patient_id","dose_type"]].drop_duplicates() #.query("dose_type.isin(['low oral','high oral','IV'])")
    doses["dose"] = 1
    doses = doses.set_index(["patient_id","dose_type"]).unstack()
    doses.columns = doses.columns.to_flat_index()
    new_columns = []
    for x in doses.columns.to_flat_index():
        line = "".join(x)
        if "dose" in line:
            line = "steroid_" + line[4:].lower()
        line = line.replace(' ','_')
        new_columns.append(line)
    doses.columns = new_columns
    doses = doses.reset_index()
    pat_tz3 = pat_tz3.merge(doses,"left","patient_id")

    print()
    print('All patients with IV steroids *in Claims* (pre-TEP): '+ str(claims_pre_tz.loc[claims_pre_tz['dose_type']=='IV','patient_id'].nunique()))
    print('All patients with IV steroids *in Patient Table* (pre-TEP): '+ str(pat_tz3.loc[(pat_tz3['steroid_iv']==1),'patient_id'].nunique()))
    print('Qualifying patients with IV steroids (pre-TEP): '+ str(pat_tz3.loc[(pat_tz3['steroid_iv']==1)&(pat_tz3['qualifying_tep_patient']==1),'patient_id'].nunique())); print()
    print('All patients with IV steroids *in Patient Table* (pre-TEP): '+ str(pat_tz3.loc[(pat_tz3['steroid_high_oral']==1),'patient_id'].nunique()))
    print('All patients with high Oral steroids *in Claims* (pre-TEP): '+ str(claims_pre_tz.loc[claims_pre_tz['dose_type']=='high oral','patient_id'].nunique()))
    print('Qualifying patients with high Oral steroids (pre-TEP): '+ str(pat_tz3.loc[(pat_tz3['steroid_high_oral']== 1)&(pat_tz3['qualifying_tep_patient']==1),'patient_id'].nunique())); print()

    ## Adding patients with surgery claims:
    surgery = claims_pre_tz[(claims_pre_tz.surgery_type.notna())|(claims_pre_tz.surgery_type2.notna())][["patient_id"]].drop_duplicates() #.query("(surgery_type.notna())or(surgery_type2.notna())")
    surgery["had_surgery"] = 1
    pat_tz3 = pat_tz3.merge(surgery,"left","patient_id")
    print('All patients with Surgeries *in Claims* (pre-TEP): '+ str(claims_pre_tz.loc[claims_pre_tz['surgery_type'].notna(),'patient_id'].nunique()))
    print('All patients with Surgeries *in Patient Table* (pre-TEP): '+ str(pat_tz3.loc[(pat_tz3['had_surgery']==1),'patient_id'].nunique()))
    print('Qualifying patients with Surgeries (pre-TEP): '+ str(pat_tz3.loc[(pat_tz3['had_surgery']== 1)&(pat_tz3['qualifying_tep_patient']==1),'patient_id'].nunique()))
    print()

    ### Get columns necessary for tep_patient table 
    pat_tz3['tep_start_date'] = pat_tz3['first_TEP']
    colsKeep = ["patient_id","qualifying_tep_patient","tep_start_date","seen_oculo","steroid_iv","steroid_high_oral","steroid_low_oral","had_surgery",
               'earliest_hub_tep','mo_FirstTEDToHub','incident_prevalent_label','acute_chronic_label']

    return pat_tz3[colsKeep]


    # for kxx
def _hub_payer_info():
    ## Read in payer type from hub
    sys.path.append(('c:\\projects\\djosephs\\DS\\src\\processing\\'))
    conn = connect(db='Tepro', server='USLSACASQL2')
    query = """SELECT *  FROM [Tepro].[dbo].[Tep_patient]"""
    tep_hub = pd.read_sql(query,conn)
    tep_hub.rename(columns = {"patient_id":"hub_patient_id"}, inplace = True)

    ## Merge payer info onto matched_hub:
    sys.path.append(('c:\\projects\\djosephs\\DS\\src\\processing\\'))
    conn = connect(db='Tepro', server='USLSACASQL2')
    query = """SELECT *  FROM [Tepro].[dbo].[Tep_patient]"""
    tep_hub = pd.read_sql(query,conn)
    tep_hub.rename(columns = {"patient_id":"hub_patient_id"}, inplace = True)

    ## Merge payer info onto matched_hub:
    # matched_hub = pd.merge(matched_hub,tep_hub[['hub_patient_id','primary_payer','primary_paytype','secondary_payer','secondary_paytype']], on = 'hub_patient_id',how = 'left')

    match_result = pd.read_parquet('Z:/temp_data/sql_push/current/tep_detail_match_result.parquet')
    match_result = match_result.rename(columns={"claims_patient_id":"patient_id"})[["patient_id","hub_patient_id"]]
    match_result = match_result.merge(tep_hub,"inner",'hub_patient_id')[['patient_id','primary_payer','primary_paytype','secondary_payer','secondary_paytype']]
    match_result = match_result.rename(columns={"primary_payer":"hub_primary_payer",
                                                "primary_paytype":"hub_primary_paytype",
                                                "secondary_payer":"hub_secondary_payer",
                                                "secondary_paytype":"hub_secondary_paytype"})
    return match_result



def get_immuno_pats():
    time_cutoff = 60

    master_code=pd.read_excel('Z:/cluster/infused_product/config/master_codes_v4.xlsx')
    master_code.loc[master_code['code_desc']=='methotrexate sodium','code_desc'] = 'methotrexate'
    immuno_kxx_code=master_code[(master_code['code_group2']=='immunomodulator')&(master_code['prod']=='kxx')]
    immuno_kxx_code['code'] = immuno_kxx_code['code'].str.zfill(11)
    #read-in claim data

    immuno_claim=pd.read_parquet(data_dir+'kxx_detail_immuno_claims.parquet')
    immuno_claim['svc_date']=pd.to_datetime(immuno_claim['svc_date'])
    rx_immuno_claim=immuno_claim[immuno_claim['code_category']=='rx']
    rx_immuno_claim['ndc']=rx_immuno_claim['ndc'].astype(str).apply(lambda x: x.replace('.0',''))
    proc_immuno_claim=immuno_claim[immuno_claim['code_category'].isin(['prc_cd'])]

    kxx_claim=pd.read_parquet(data_dir+'kxx_claims.parquet')
    kxx_claim=kxx_claim.query("code_group2=='kxx'")
    kxx_claim['svc_date']=pd.to_datetime(kxx_claim['svc_date'])

    hcp_df = pd.read_parquet(data_dir+'kxx_detail_providers.parquet')

    kxx_claim_deduped=kxx_claim[['patient_id','code','svc_date','hcp_id']].drop_duplicates()
    kxx_claim_deduped=kxx_claim_deduped.groupby(['patient_id','code','svc_date']).first().reset_index()

    kxx_agg=kxx_claim_deduped.groupby(['patient_id']).agg({'svc_date' : ['count', 'min']})
    kxx_agg.columns=['kxx_vials','first_kxx_date']
    kxx_agg=kxx_agg.reset_index()

    kxx_first_hcp=kxx_claim_deduped.loc[kxx_claim_deduped.groupby(['patient_id'])['svc_date'].idxmin()][['patient_id','hcp_id']]
    kxx_first_hcp.columns=['patient_id', 'first_kxx_hcp_id']
    kxx_agg=pd.merge(kxx_agg, kxx_first_hcp, on=['patient_id'], how='left')
    kxx_agg['kxx_cutoff_date']=kxx_agg['first_kxx_date'].apply(lambda x: x+ relativedelta(years=-1))

    immuno_proc_deduped=proc_immuno_claim[['patient_id','code','svc_date','hcp_id']].drop_duplicates()
    immuno_proc_deduped=immuno_proc_deduped.groupby(['patient_id','code','svc_date']).first().reset_index()


    # check whether ANY immuno claim is within time limit 

    all_immuno = pd.concat([rx_immuno_claim[['patient_id','svc_date']].drop_duplicates(),immuno_proc_deduped[['patient_id','svc_date']].drop_duplicates()])
    all_immuno = pd.merge(kxx_agg,all_immuno, how = 'outer',on = ['patient_id'])

    #filter out recs where patient has no immuno
    all_immuno = all_immuno[all_immuno['svc_date'].notnull()]
    all_immuno['time_diff'] = (abs(all_immuno['svc_date'] - all_immuno['first_kxx_date'])).apply(lambda x: x.days)
    #all_immuno['time_between'] = (all_immuno['svc_date'] - all_immuno['first_kxx_date']).apply(lambda x: x.days)

    first_immuno_rx=rx_immuno_claim.loc[rx_immuno_claim.groupby(['patient_id'])['svc_date'].idxmin()][['patient_id','svc_date','hcp_id','ndc']].rename(columns={'ndc':'code'})
    first_immuno_rx['code']=first_immuno_rx['code'].str.zfill(11)
    #exclude immuno from iqvia(code_desc=nan)
    first_immuno_rx=first_immuno_rx[first_immuno_rx['code'].isin(immuno_kxx_code['code'])]

    immuno_proc_deduped=proc_immuno_claim[['patient_id','code','svc_date','hcp_id']].drop_duplicates()
    immuno_proc_deduped=immuno_proc_deduped.groupby(['patient_id','code','svc_date']).first().reset_index()
    first_immuno_proc=immuno_proc_deduped.loc[immuno_proc_deduped.groupby(['patient_id'])['svc_date'].idxmin()][['patient_id','svc_date','hcp_id','code']]#.rename(columns={'prc_cd':'code'})

    first_immuno=pd.concat([first_immuno_rx, first_immuno_proc])
    first_immuno.index=range(len(first_immuno))
    first_immuno=first_immuno.loc[first_immuno.groupby(['patient_id'])['svc_date'].idxmin()].rename(columns={'svc_date':'first_immuno_date','hcp_id':'first_immuno_hcp_id'})
    
    kxx_pats=pd.merge(kxx_agg,first_immuno,on=['patient_id'], how='outer')
    kxx_pats['code'] = kxx_pats['code'].astype(str).apply(lambda x: x.replace('.0',''))
    kxx_pats = pd.merge(kxx_pats, immuno_kxx_code[['code','code_desc']],how = 'left',on = ['code']).rename(columns = {'code_desc':'prod'})
    #new defined immuno pats:apply the one year bound to the time frame
    modified_immuno=all_immuno[(all_immuno['svc_date']>all_immuno['kxx_cutoff_date'])]#&(all_immuno['first_kxx_date']<=cutoff_date)]
    modified_first_immuno=modified_immuno.loc[modified_immuno.groupby(['patient_id'])['svc_date'].idxmin()][['patient_id','svc_date','first_kxx_date','first_kxx_hcp_id','kxx_cutoff_date']].rename(columns={'ndc':'code'})
    modified_first_immuno['time_diff']=(abs(modified_first_immuno['svc_date'] - modified_first_immuno['first_kxx_date'])).apply(lambda x: x.days)
    modified_first_immuno=modified_first_immuno[abs(modified_first_immuno['time_diff'])<=time_cutoff]
    #add hcp npis

    modified_first_immuno=pd.merge(modified_first_immuno, hcp_df[['hcp_id','npi']].astype(str), right_on=['hcp_id'], left_on=['first_kxx_hcp_id'], how='left').drop(['hcp_id'], axis=1).rename(columns={'npi':'first_kxx_hcp_npi'})
    modified_first_immuno['first_kxx_hcp_npi'] = modified_first_immuno['first_kxx_hcp_npi'].astype(str).apply(lambda x: x.replace('.0',''))
    modified_immuno=kxx_pats[kxx_pats['patient_id'].isin(modified_first_immuno['patient_id'])]
    modified_immuno=modified_immuno[modified_immuno['prod'].isin(['methotrexate','azathioprine'])]

    immuno_patients = modified_immuno[["patient_id"]].drop_duplicates()
    immuno_patients["is_immuno_patient"] = 1
    
    return immuno_patients

def TreaterReferrer():
    inputDir='Z:/temp_data/sql_push/current/'
    claim_seq_df=pd.read_parquet(inputDir+'seq_df.parquet')#referral sequence from cloud
    claim_seq_df[['prev_npi']]=claim_seq_df[['prev_npi']].astype(str)
    claim_seq_df[['npi']]=claim_seq_df[['npi']].astype(str)
    target_hcp=pd.read_excel('Z:/Projects/Adhoc/Blair/ted_referral_prediction/inputs/20210226/TEPEZZA Target Score_update 20210216.xlsx')#will be updated soon. The target file
    target_hcp[['NPI']]=target_hcp[['NPI']].astype(str)
    target_hcp_treater=target_hcp[target_hcp['HCPs Type']=='Treater']
    target_hcp_referrer=target_hcp[target_hcp['HCPs Type']=='Referrer']
    #hcp_patient is unique at hcp prev_hcp level and designed to calculate the patients referred from treaters to referrers
    hcp_patient=claim_seq_df.groupby(['npi','specialty','prev_npi','prev_specialty'])[['patient_id']].nunique().reset_index()
    #for treaters, find the top referrers in the same territory
    hcp_patient_treater=hcp_patient[hcp_patient['npi'].isin(target_hcp_treater['NPI'].unique())]
    #add territory first
    hcp_patient_treater=pd.merge(hcp_patient_treater, target_hcp[['NPI','Territory']].rename(columns={'NPI':'npi','Territory':'territory'}), on=['npi'], how='left')
    hcp_patient_treater=pd.merge(hcp_patient_treater, target_hcp[['NPI','Territory']].rename(columns={'NPI':'prev_npi','Territory':'prev_territory'}), on=['prev_npi'], how='left')
    #only 2295 of them have referrers in the same territory and no one has no territory 
    #unique number of treaters is 2450
    hcp_patient_treater=hcp_patient_treater[hcp_patient_treater['territory']==hcp_patient_treater['prev_territory']]
    hcp_patient_treater=hcp_patient_treater.sort_values(['npi','patient_id'], ascending=False)

    hcp_patient_treater_top=hcp_patient_treater.sort_values(['npi','patient_id'], ascending=False).groupby(['npi']).nth(0).reset_index()[['npi','prev_npi']]
    hcp_patient_treater_top.columns=['treater_referrer_npi','referrer1_npi']
    hcp_patient_treater_top2=hcp_patient_treater.sort_values(['npi','patient_id'], ascending=False).groupby(['npi']).nth(1).reset_index()[['npi','prev_npi']]
    hcp_patient_treater_top2.columns=['treater_referrer_npi','referrer2_npi']
    hcp_patient_treater_top3=hcp_patient_treater.sort_values(['npi','patient_id'], ascending=False).groupby(['npi']).nth(2).reset_index()[['npi','prev_npi']]
    hcp_patient_treater_top3.columns=['treater_referrer_npi','referrer3_npi']
    hcp_patient_treater_top123=pd.merge(hcp_patient_treater_top,hcp_patient_treater_top2, on=['treater_referrer_npi'], how='left' )
    hcp_patient_treater_top123=pd.merge(hcp_patient_treater_top123,hcp_patient_treater_top3, on=['treater_referrer_npi'], how='left' )

    hcp_patient_treater_top123['treater_referrer_index']='Treater'
    hcp_patient_treater_top123['treater1_npi']=np.nan
    hcp_patient_treater_top123['treater2_npi']=np.nan
    hcp_patient_treater_top123['treater3_npi']=np.nan
    #do the same for referrers
    hcp_patient_referrer=hcp_patient[hcp_patient['prev_npi'].isin(target_hcp_referrer['NPI'].unique())]
    #add territory first
    hcp_patient_referrer=pd.merge(hcp_patient_referrer, target_hcp[['NPI','Territory']].rename(columns={'NPI':'npi','Territory':'territory'}), on=['npi'], how='left')
    hcp_patient_referrer=pd.merge(hcp_patient_referrer, target_hcp[['NPI','Territory']].rename(columns={'NPI':'prev_npi','Territory':'prev_territory'}), on=['prev_npi'], how='left')
    #unique number of treaters is 15,106
    hcp_patient_referrer=hcp_patient_referrer[hcp_patient_referrer['territory']==hcp_patient_referrer['prev_territory']]
    #only 13,494 of them have referrers in the same territory and no one has no territory 
    hcp_patient_referrer=hcp_patient_referrer.sort_values(['prev_npi','patient_id'], ascending=False)

    hcp_patient_referrer_top=hcp_patient_referrer.sort_values(['prev_npi','patient_id'], ascending=False).groupby(['prev_npi']).nth(0).reset_index()[['prev_npi','npi']]
    hcp_patient_referrer_top.columns=['treater_referrer_npi','treater1_npi']
    hcp_patient_referrer_top2=hcp_patient_referrer.sort_values(['prev_npi','patient_id'], ascending=False).groupby(['prev_npi']).nth(1).reset_index()[['prev_npi','npi']]
    hcp_patient_referrer_top2.columns=['treater_referrer_npi','treater2_npi']
    hcp_patient_referrer_top3=hcp_patient_referrer.sort_values(['prev_npi','patient_id'], ascending=False).groupby(['prev_npi']).nth(2).reset_index()[['prev_npi','npi']]
    hcp_patient_referrer_top3.columns=['treater_referrer_npi','treater3_npi']
    hcp_patient_referrer_top123=pd.merge(hcp_patient_referrer_top,hcp_patient_referrer_top2, on=['treater_referrer_npi'], how='left' )
    hcp_patient_referrer_top123=pd.merge(hcp_patient_referrer_top123,hcp_patient_referrer_top3, on=['treater_referrer_npi'], how='left' )

    hcp_patient_referrer_top123['treater_referrer_index']='Referrer'
    hcp_patient_referrer_top123['referrer1_npi']=np.nan
    hcp_patient_referrer_top123['referrer2_npi']=np.nan
    hcp_patient_referrer_top123['referrer3_npi']=np.nan
    treater_referrer_index=pd.concat([hcp_patient_referrer_top123, hcp_patient_treater_top123])
    return treater_referrer_index


def get_kxx_hub_data():
    conn = connect(server = 'USLSACASQL1',db = 'Rheum_Veeva')
    kxx_hub = """SELECT p.Name, p.Enrollment_Date_MVN__c, p.Status_MVN__c, p.Status_Reason_MVN__c, 
        p.HZN_Patient_Age__c, p.DOB_MVN__c, p.Patient_Birthdate_MVN__c, p.Gender_MVN__c, 
        aa.city Patient_city, aa.state Patient_state, aa.zip Patient_zip,
        a.NPI_vod__c,a.FirstName Enrolling_Physician_FirstName, a.LastName Enrolling_Physician_LastName,a.Primary_Address_Line_1_MVN__c, a.Primary_City_MVN__c, a.Primary_State_MVN__c,
        a.Primary_Zip_Code_MVN__c, a.Horizon_Specialty_MVN__c, a.Specialty_1_vod__c
        FROM Rheum_Veeva.dbo.veeva_ib_program_member p
        LEFT JOIN Rheum_Veeva.dbo.veeva_ib_accounts a ON a.Id = p.Enrolling_Physician_MVN__c
        LEFT JOIN Rheum_Veeva.dbo.vw_PM_contact_info_address aa ON aa.pm_id = p.Name
        WHERE p.Program_Name_MVN__c = 'KRYSTEXXA'
        AND p.Enrollment_Date_MVN__c >= '2020-01-01'
        ORDER BY p.Enrollment_Date_MVN__c desc"""
    kxx_hub = pd.read_sql(kxx_hub, conn)
    return kxx_hub

if __name__=="__main__":
    print("getting tep patients hub data")
    get_hub_data().to_parquet("Z:/temp_data/up/hub/tep_pats_hub_data.parquet")
    print("getting kxx patients hub data")
    get_kxx_hub_data().to_parquet("Z:/temp_data/up/hub/kxx_pats_hub_data.parquet")
    print("hub data saved")