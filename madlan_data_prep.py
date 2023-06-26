# -*- coding: utf-8 -*-
"""
Created on Mon Jun 26 13:07:31 2023

@author: shirt
"""

import pandas as pd
import numpy as np 
import re
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RepeatedKFold 


def prepare_data(file):
    
    
    data = pd.read_csv(file)
    data = data.drop_duplicates()
    
    data = data.reset_index(drop=True)  
    data['price'] = data['price'].str.replace('[^\d.]', '', regex=True)
    data['price'] = pd.to_numeric(data['price'], errors='coerce', downcast='integer')
    data.dropna(subset=['price'], inplace=True)
    data['price'] = data['price'].astype('int64')
     
    data['Area'] = data['Area'].str.replace('[^\d.]', '', regex=True)
    data['Area'] = pd.to_numeric(data['Area'], errors='coerce', downcast='integer')
    # data['Area'] = np.where(data['Area'].isnull(), 0, data['Area'])
    data.dropna(subset=['Area'], inplace=True) 
    data['Area'] = data['Area'].astype('int64')
    
    data['room_number'] = data['room_number'].apply(lambda x: float(re.search(r'\d+\.?\d*', str(x)).group()) if re.search(r'\d+\.?\d*', str(x)) else None)
    data['num_of_images'] = data['num_of_images'].astype(float).fillna(0)
    
    data['condition '] = data['condition '].fillna('לא צויין')
    data['condition '] = data['condition '].apply(lambda x: 'לא צויין' if re.search(r'(FALSE|None)', str(x)) else x)
    
    data['description '] = data['description '].str.replace(',', ' ')
    data['description '] = data['description '].str.replace('!', ' ')
    data['description '] = data['description '].str.replace('-', '')
    data['description '] = data['description '].str.replace('\.', ' ')
    
    data['Street'] = data['Street'].str.replace(r'(?<=[א-ת])(?=[א-ת])', ' ',regex=False)
    data['city_area'] = data['city_area'].str.replace(r'(?<=[א-ת])(?=[א-ת])', '',regex=False)
    data['floor'] = data['floor_out_of'].apply(lambda x: re.search(r'קומה (\d+) מתוך', str(x)).group(1) if (re.search(r'מתוך', str(x)) and re.search(r'קומה (\d+) מתוך', str(x))) else 0)
    data['total_floor'] = data['floor_out_of'].str.extract(r'מתוך (\d+)')
    data['total_floor'] = data['total_floor'].astype(float).fillna(0)
    data.loc[data['type'].str.contains(r'(דו|פרטי)'), 'total_floor'] = 0
    data['total_floor'] = data['total_floor'].astype('int64') ##### אם יש ערך מספרי בפלור ו0 בטוטאל פלור זה טעות בדאטה
    data['floor'] = data['floor'].astype('int64')
    
    data['hasElevator '] = data['hasElevator '].apply(lambda x: 0 if re.search(r'(אין|FALSE|לא)', str(x)) else 1)
    data['hasBalcony '] = data['hasBalcony '].apply(lambda x: 0 if re.search(r'(אין|FALSE|לא)', str(x)) else 1)
    data['hasParking '] = data['hasParking '].apply(lambda x: 0 if re.search(r'(אין|FALSE|לא)', str(x)) else 1)
    data['hasBars '] = data['hasBars '].apply(lambda x: 0 if re.search(r'(אין|FALSE|לא)', str(x)) else 1)
    data['hasMamad '] = data['hasMamad '].apply(lambda x: 0 if re.search(r'(אין|FALSE|לא)', str(x)) else 1)
    data['hasAirCondition '] = data['hasAirCondition '].apply(lambda x: 0 if re.search(r'(אין|FALSE|לא)', str(x)) else 1)
    data['hasStorage '] = data['hasStorage '].apply(lambda x: 0 if re.search(r'(אין|FALSE|לא)', str(x)) else 1)
    data['handicapFriendly '] = data['handicapFriendly '].apply(lambda x: 0 if re.search(r'(nan|אין|FALSE|לא)', str(x)) else 1)
    
    def categorize_entrance_date(date):
        if date in ['גמיש', 'מיידי', 'גמיש ']:
               return 'flexible'
        if date == 'לא צויין':
               return 'not_defined'
        else:
            date = pd.to_datetime(date, format='%d/%m/%Y')
            time_difference = (date - pd.Timestamp.now()).days
            if time_difference < 0:
                 return 'flexible'
            if time_difference < 180:
                 return 'less_than_6 months'
            elif time_difference >= 180 and time_difference <= 365:
                 return 'months_6_12'
            else:
                 return 'above_year'

    data['entrance_date '] = data['entranceDate '].apply(categorize_entrance_date)
    
    
    data['City'].unique()
    data['City'] = data['City'].str.replace('נהריה',' נהרייה')
    data['City'] = data['City'].str.replace('נהריה',' נהריה')
    data['City'] = data['City'].str.replace('נהריה','נהרייה')
    data['City'] = data['City'].str.replace(' שוהם', 'שוהם')
    for col in ['City', 'type','Street' ,'condition ']:
           if data[col].dtype == 'object':
                data[col] = data[col].str.strip()
                
    dataset = data.copy()
    dataset['city_area'] = dataset['city_area'].fillna('מרכז')
    dataset['city_area'] = dataset.apply(lambda x: str(x['city_area']) + ' ' + str(x['City']), axis=1)
    dataset['city_area'] = dataset['city_area'].astype('str')

    dataset.dropna(subset=['room_number'], inplace=True)
    dataset = dataset.drop(columns=['floor_out_of','number_in_street','Street','num_of_images','description ','city_area'])
    dataset = dataset.reindex(columns=['City','type','room_number','Area', 'hasElevator ',
       'hasParking ', 'hasBars ','entrance_date ', 'hasStorage ', 'condition ',
       'hasAirCondition ', 'hasBalcony ', 'hasMamad ', 'handicapFriendly ',
       'furniture ', 'floor', 'total_floor','publishedDays ',
       'price'])

    dataset['publishedDays '] = dataset['publishedDays '].fillna(0)
    dataset['publishedDays '] = dataset['publishedDays '].apply(lambda x: 60 if re.search(r'(60+)', str(x)) else x)
    dataset['publishedDays '] = dataset['publishedDays '].apply(lambda x: 0 if re.search(r'(None |Nan|חדש|-|None)', str(x)) else int(x))
    dataset['publishedDays '] = dataset['publishedDays '].apply(lambda x: 1 if x > 30 else 0)
    
    return dataset
dataset = prepare_data("C:\\Users\\shirt\Desktop\\מטלה מסכמת\\output_all_students_Train_v10.csv")
print(dataset)