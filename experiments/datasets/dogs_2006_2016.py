import time
import pandas as pd
import numpy as np
from os.path import dirname
from . import converters;

from sklearn.datasets.base import Bunch

#Load dataframe
def load_df_dogs_2016(dropNA = False, dropColumns = [], fixErrors = True):
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
    deleteList = []

    for i, row in data.iterrows():
        #Use the same date format
        for attr in timeCols:
            data.set_value(i, attr, converters.date_converter(row[attr]))

        row = data.iloc[i, :]

        if fixErrors:
            #Fix incorrect survival time
            if (row["Date of death"] - row["First visit"]).days != row["Survival time"]:
                data.set_value(i, "Survival time", (row["Date of death"] - row["First visit"]).days)

        for attr in timeCols:
            data.set_value(i, attr, time.mktime(row[attr].timetuple()))

    #Delete rows set for deletion
    data.drop(deleteList, inplace=True)

    #Delete useless columns
    data.drop(dropColumns, axis="columns", inplace=True)

    #Delete NA
    if dropNA:
        data.dropna(axis=0, how='any', inplace=True)

    return data

#default drop columns
dropNonNumeric = ["Folder", "isachc", "Class"]
dropIrrelevant = ["IP", "Dead", "MC", "Furosemide", "Ache-i", "Pimobendan", "Spironolattone"]
dropDates = ["Birth date", "First visit", "Therapy started", "Date of death"]

#load sklearn Bunch object with Survival time as target
def load_skl_dogs_2016(NApolicy='drop', dropColumns=dropNonNumeric+dropIrrelevant+dropDates, scaler=None):

    napolicies = {'drop':True}
    data = load_df_dogs_2016(dropNA = napolicies[NApolicy], dropColumns = dropColumns)

    #Target column
    targetArray = data.loc[:, "Survival time"].as_matrix()
    data.drop("Survival time", axis="columns", inplace=True)

    featureNames = list(data.columns)
    dataMatrix = data.as_matrix()
    if scaler is not None:
        dataMatrix = scaler.fit_transform(dataMatrix)

    return Bunch(feature_names = featureNames, data = dataMatrix, target = targetArray)
