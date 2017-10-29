import time
import pandas as pd
import numpy as np
from os.path import dirname
from . import converters;

from sklearn.datasets.base import Bunch

#Load dataframe
def load_df_dogs_2016(NApolicy = 'none', dropColumns = [], fixErrors = True, censoringPolicy = 'none', newFeats = True):
    module_path = dirname(__file__)
    data = pd.read_excel(module_path + "/data/dogs.xlsx",
                         spreadsheet="2006-2016",
                         converters={"IP": converters.sieno_converter,
                                     "Furosemide": converters.sieno_converter,
                                     "Ache-i": converters.sieno_converter,
                                     "Pimobendan": converters.sieno_converter,
                                     "Spironolattone": converters.sieno_converter,
                                     "Antiaritmico": converters.sieno_converter,
                                    },
                         dtype={"Cartella": np.str,
                                "Gravità IP": np.int,
                                "Vrig Tric": np.float,
                                "Età": np.float,
                                "MORTE": np.float,
                                "MC": np.float,
                                "Data di nascita": np.str,
                                "Data 1° visita": np.str,
                                "Inizio Terapia": np.str,
                                "Data morte ": np.str,
                                "SURVIVAL TIME": np.int,
                                "Terapia": np.int,
                                "isachc": np.str,
                                "CLASSE": np.str,
                                "Peso (Kg)": np.float,
                                "Asx/Ao": np.float,
                                "E": np.float,
                                "E/A": np.float,
                                "FE %": np.float,
                                "FS%": np.float,
                                "EDVI": np.float,
                                "ESVI": np.float,
                                "Allo diast": np.float,
                                "Allo sist": np.float
                               }
                        )
    data.rename(columns={"Cartella": "Folder",
                         "Gravità IP": "IP Gravity",
                         "Data di nascita": "Birth date",
                         "Data 1° visita": "First visit",
                         "Età": "Age",
                         "Inizio Terapia": "Therapy started",
                         "MORTE": "Dead",
                         "Data morte ": "Date of death",
                         "SURVIVAL TIME": "Survival time",
                         "Terapia": "Therapy Category",
                         "CLASSE": "Class",
                         "Peso (Kg)": "Weight (Kg)",
                         "FS%": "FS %"
                        }, inplace=True)

    timeCols = ["Birth date", "First visit", "Therapy started", "Date of death"]

    #Setting up new column "Therapy to visit", a time delta in days
    thertovisit = pd.Series()

    for i, row in data.iterrows():
        #Use the same date format
        for attr in timeCols:
            data.set_value(i, attr, converters.date_converter(row[attr]))

        row = data.iloc[i, :]

        if fixErrors:
            #Fix incorrect survival time
            if (row["Date of death"] - row["First visit"]).days != row["Survival time"]:
                data.set_value(i, "Survival time", (row["Date of death"] - row["First visit"]).days)

        row = data.iloc[i, :]
        #Compute "Therapy to visit"
        thertovisit.set_value(i, (row["First visit"] - row["Therapy started"]).days)

        for attr in timeCols:
            data.set_value(i, attr, time.mktime(row[attr].timetuple()))

    if newFeats:
        #add "Therapy to visit" feature to the dataset
        data["Therapy to visit"] = thertovisit

    #Censoring policies
    #drop censored rows
    if censoringPolicy=='drop':
        data.drop(data[data["Dead"]==0].index, inplace=True)
    #substitute survtime with max survtime of dead subjects
    elif censoringPolicy=='max':
        survmax = data["Survival time"][data["Dead"]==1].max()
        for i, row in data.iterrows():
            if row["Dead"]==0:
                data.set_value(i, "Survival time", survmax)

    #Delete useless columns
    data.drop(dropColumns, axis="columns", inplace=True)

    #NA policies
    #drop
    if NApolicy=='drop':
        data.dropna(axis=0, how='any', inplace=True)
    #Fill NA with mean value of feature
    elif NApolicy=='mean':
        means = {nacol:data[nacol].mean() for nacol in data.columns[data.isnull().any()].tolist()}
        data.fillna(value=means, inplace=True)
    #Fill NA with generated normal values
    elif NApolicy=='normal':
        params = {nacol:(data[nacol].mean(),data[nacol].std()) for nacol in data.columns[data.isnull().any()].tolist()}
        for i, row in data.iterrows():
            for nacol in params.keys():
                if pd.isnull(row[nacol]):
                    data.set_value(i, nacol, np.random.normal(loc=params[nacol][0], scale=params[nacol][1]))

    return data

#default drop columns
dropNonNumeric = ["Folder", "isachc", "Class"]
dropIrrelevant = ["IP", "Furosemide", "Ache-i", "Pimobendan", "Spironolattone"]
dropDead = ["Dead", "MC"]
dropDates = ["Birth date", "First visit", "Therapy started", "Date of death"]

#load sklearn Bunch object with Survival time as target
def load_skl_dogs_2016(NApolicy='drop', dropColumns=dropNonNumeric+dropIrrelevant+dropDead+dropDates, censoringPolicy='none', newFeats=True, scaler=None):

    data = load_df_dogs_2016(NApolicy = NApolicy, dropColumns = dropColumns, censoringPolicy=censoringPolicy, newFeats=newFeats)

    #Target column
    targetArray = data.loc[:, "Survival time"].as_matrix()
    data.drop("Survival time", axis="columns", inplace=True)

    featureNames = list(data.columns)
    dataMatrix = data.as_matrix()
    if scaler is not None:
        dataMatrix = scaler.fit_transform(dataMatrix)

    return Bunch(feature_names = featureNames, data = dataMatrix, target = targetArray)
