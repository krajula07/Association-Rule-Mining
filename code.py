# -*- coding: utf-8 -*-
"""
Created on Wed Dec  2 21:54:35 2020

@author: krajula
"""


import pandas as pd
import numpy as np
import time
from datetime import timedelta
from datetime import datetime

from numpy import savetxt
from matplotlib import pyplot as plt

from collections import Counter

CGM_Patient1_File='CGMData.csv'
CGM_data_P1 = pd.read_csv(CGM_Patient1_File,low_memory=False,parse_dates=[['Date','Time']])
CGM_data_P1['Date']=CGM_data_P1.Date_Time.dt.date
CGM_data_P1['Time']=CGM_data_P1.Date_Time.dt.time

Insulin_Patient1_File='InsulinData.csv'
Insulin_data_P1 = pd.read_csv(Insulin_Patient1_File,low_memory=False,parse_dates=[['Date','Time']])
Insulin_data_P1['Date']=Insulin_data_P1.Date_Time.dt.date
Insulin_data_P1['Time']=Insulin_data_P1.Date_Time.dt.time

MealTime_P1 = Insulin_data_P1[Insulin_data_P1['BWZ Estimate (U)'].notnull()]
#print(MealTime_P1)
MealmodeOn=MealTime_P1.tail(1)
#print(MealmodeOn)
min=CGM_data_P1.head(1)
max=CGM_data_P1.tail(1)
min_value=min['Date_Time']
max_value=max['Date_Time']
datetime_t1=pd.to_datetime(min['Date_Time'])
#print(datetime_t1)


Previous_time_stamp=max_value
mealdata=[]
df = pd.DataFrame() 
dfs={}
d=[]
l_Itime=[]
l_CTime=[]
check=[]
nomeal_d=[]
d_p1_nm=[]
Column_y_values=[]
CGM_B_meal_values=[]
for i, row in enumerate(MealTime_P1[::-1].iterrows()):
    #print(row[1][0])
    temp_I=[row[1][0]]
    l_Itime.append(temp_I)
    temp_C=[CGM_data_P1['Date_Time']]
    l_CTime.append(temp_C)
    Compared_rows=CGM_data_P1[((CGM_data_P1['Date_Time'])>=(row[1][0]))]
    Compared_row=Compared_rows.tail(1)
    
    #print(datetime_t2)
    if i==0:
        t1=(pd.to_datetime(Compared_row['Date_Time']).to_frame())
        
        start=(pd.to_datetime(Compared_row['Date_Time'])-timedelta(minutes=30)).to_frame()
        end=(pd.to_datetime(Compared_row['Date_Time'])+timedelta(hours=2)).to_frame()
        temp=[start.values[0],t1.values[0],end.values[0]]
        check.append(temp)
    
        nm=(pd.to_datetime(Compared_row['Date_Time'])+timedelta(hours=2)).to_frame()
        S_time = start['Date_Time'].apply(lambda x: x.strftime('%Y-%m-%d %H:%M:%S'))
        E_time= end['Date_Time'].apply(lambda x: x.strftime('%Y-%m-%d %H:%M:%S'))
        Exact_time=t1['Date_Time'].apply(lambda x: x.strftime('%Y-%m-%d %H:%M:%S'))
        #print(S_time,E_time,Exact_time)
        mask = (CGM_data_P1['Date_Time'] >= S_time.values[0]) & (CGM_data_P1['Date_Time'] < E_time.values[0])
        rows=CGM_data_P1.loc[mask]
        temp=rows['Sensor Glucose (mg/dL)'].to_frame()
        t=list(temp.iloc[: ,0].values)
        dfs[i]=t
        d.append(t)
        
        l=CGM_data_P1.loc[CGM_data_P1['Date_Time'] == Exact_time.values[0]]['Sensor Glucose (mg/dL)'].to_frame()
        t_value=list(l.values)
        CGM_B_meal_values.append(t_value)
        #print(CGM_B_meal_values,type(CGM_B_meal_values[0]),type(CGM_B_meal_values),'kkkkkkkkk')
        
        Column_y_values.append(row[1][0])
        
    else:
        t2=(pd.to_datetime(Compared_row['Date_Time']).to_frame())
        #print(t1.values[0],t2.values[0],end.values[0])
        right=t1.values[0]<=t2.values[0]
        #print(right)
        left=t2.values[0]<end.values[0]
        #print(left)
        eq=(t2.values[0]==end.values[0])
        
        St_time = nm['Date_Time'].apply(lambda x: x.strftime('%Y-%m-%d %H:%M:%S'))
        #print(St_time)
        Et_time= t2['Date_Time'].apply(lambda x: x.strftime('%Y-%m-%d %H:%M:%S'))
        #print(St_time-Et_time)
        Exact_time=t1['Date_Time'].apply(lambda x: x.strftime('%Y-%m-%d %H:%M:%S'))
        mask = (CGM_data_P1['Date_Time'] >= St_time.values[0]) & (CGM_data_P1['Date_Time'] < Et_time.values[0])
        rows_t=CGM_data_P1.loc[mask]
        temp=rows_t['Sensor Glucose (mg/dL)'].to_frame()
        t_p1=list(temp.iloc[: ,0].values)
        
        d_p1_nm.append(t_p1)
        #Column_y_values.append(row[1][0])
        #print(eq)
        if (right and left):
            #print('kavya in between ')
            d.pop()
            check.pop()
            d_p1_nm.pop()
            Column_y_values.pop()
            CGM_B_meal_values.pop()
            
            t1=(pd.to_datetime(Compared_row['Date_Time']).to_frame())
            nm=(pd.to_datetime(Compared_row['Date_Time'])+timedelta(hours=2)).to_frame()
            start=(pd.to_datetime(Compared_row['Date_Time'])-timedelta(minutes=30)).to_frame()
            end=(pd.to_datetime(Compared_row['Date_Time'])+timedelta(hours=2)).to_frame()
            temp=[start.values[0],t1.values[0],end.values[0]]
            check.append(temp)
            S_time = start['Date_Time'].apply(lambda x: x.strftime('%Y-%m-%d %H:%M:%S'))
            E_time= end['Date_Time'].apply(lambda x: x.strftime('%Y-%m-%d %H:%M:%S'))
            Exact_time=t1['Date_Time'].apply(lambda x: x.strftime('%Y-%m-%d %H:%M:%S'))
            mask = (CGM_data_P1['Date_Time'] >= S_time.values[0]) & (CGM_data_P1['Date_Time'] < E_time.values[0])
            rows=CGM_data_P1.loc[mask]
            temp=rows['Sensor Glucose (mg/dL)'].to_frame()
            t=list(temp.iloc[: ,0].values)
            dfs[i]=t
            d.append(t)
            l=CGM_data_P1.loc[CGM_data_P1['Date_Time'] == Exact_time.values[0]]['Sensor Glucose (mg/dL)'].to_frame()
            t_value=list(l.values)
            CGM_B_meal_values.append(t_value)
            Column_y_values.append(row[1][0])
            
            
            
        elif (eq):
            #print('kavya in equal')
            d.pop()
            CGM_B_meal_values.pop()
            check.pop()
            d_p1_nm.pop()
            Column_y_values.pop()
            t1=(pd.to_datetime(Compared_row['Date_Time']).to_frame())
            nm=(pd.to_datetime(Compared_row['Date_Time'])+timedelta(hours=2)).to_frame()
            start=(pd.to_datetime(Compared_row['Date_Time'])-timedelta(minutes=30)).to_frame()
            end=(pd.to_datetime(Compared_row['Date_Time'])+timedelta(hours=2)).to_frame()
            temp=[start.values[0],t1.values[0],end.values[0]]
            check.append(temp)
            S_time = start['Date_Time'].apply(lambda x: x.strftime('%Y-%m-%d %H:%M:%S'))
            E_time= end['Date_Time'].apply(lambda x: x.strftime('%Y-%m-%d %H:%M:%S'))
            Exact_time=t1['Date_Time'].apply(lambda x: x.strftime('%Y-%m-%d %H:%M:%S'))
            mask = (CGM_data_P1['Date_Time'] >= S_time.values[0]) & (CGM_data_P1['Date_Time'] < E_time.values[0])
            rows=CGM_data_P1.loc[mask]
            temp=rows['Sensor Glucose (mg/dL)'].to_frame()
            t=list(temp.iloc[: ,0].values)
            dfs[i]=t
            d.append(t)
            l=CGM_data_P1.loc[CGM_data_P1['Date_Time'] == Exact_time.values[0]]['Sensor Glucose (mg/dL)'].to_frame()
            t_value=list(l.values)
            CGM_B_meal_values.append(t_value)
            Column_y_values.append(row[1][0])
        else:
            t1=(pd.to_datetime(Compared_row['Date_Time']).to_frame())
            nm=(pd.to_datetime(Compared_row['Date_Time'])+timedelta(hours=2)).to_frame()
            start=(pd.to_datetime(Compared_row['Date_Time'])-timedelta(minutes=30)).to_frame()
            end=(pd.to_datetime(Compared_row['Date_Time'])+timedelta(hours=2)).to_frame()
            temp=[start.values[0],t1.values[0],end.values[0]]
            check.append(temp)
            S_time = start['Date_Time'].apply(lambda x: x.strftime('%Y-%m-%d %H:%M:%S'))
            E_time= end['Date_Time'].apply(lambda x: x.strftime('%Y-%m-%d %H:%M:%S'))
            Exact_time=t1['Date_Time'].apply(lambda x: x.strftime('%Y-%m-%d %H:%M:%S'))
            mask = (CGM_data_P1['Date_Time'] >= S_time.values[0]) & (CGM_data_P1['Date_Time'] < E_time.values[0])
            rows=CGM_data_P1.loc[mask]
            temp=rows['Sensor Glucose (mg/dL)'].to_frame()
            t=list(temp.iloc[: ,0].values)
            dfs[i]=t
            d.append(t)
            l=CGM_data_P1.loc[CGM_data_P1['Date_Time'] == Exact_time.values[0]]['Sensor Glucose (mg/dL)'].to_frame()
            t_value=list(l.values)
            CGM_B_meal_values.append(t_value)
            Column_y_values.append(row[1][0])
            
            
#print(len(d))
meal_P1=pd.DataFrame(d)
print(len(l_Itime))
print(len(l_CTime)) 
print(len(check)) 

print(len(Column_y_values))
time_column_Y = pd.DataFrame (Column_y_values,columns=['Date_Time'])
print (time_column_Y.info())
time_column_Y.set_index('Date_Time')
Insulin_data_P1.set_index('Date_Time')
merged=time_column_Y.merge(Insulin_data_P1, how='inner')
MealAmountData = merged[merged['BWZ Estimate (U)'].notnull()]
#MealAmountData['BWZ Estimate (U)'].to_csv('mealAmountData1.csv',header=False,index=False)
#merged.to_csv('check_d.csv')
#step2:
rounded_tValues=MealAmountData['BWZ Estimate (U)'].tolist()
rounded_tValues=[round(intValue) for intValue in rounded_tValues]
nomeal_ref_p1=pd.DataFrame(check)

meal_P1 = meal_P1.drop(meal_P1.columns[-1],axis=1) 
#meal_P1.to_csv('mealData1.csv',header=False,index=False) 

#maximum CgM values 
CGM_max_values=meal_P1.max(axis=1)
CGM_max_values=CGM_max_values.to_frame()    
CGM_B=CGM_max_values.values.tolist()
CGM_B_max = [ item for elem in CGM_B for item in elem] 
#print(CGM_B_max,len(CGM_B_max),'hhhhhhhhhhhhhhhhhhhhhhhhhhhh')

#minimum CgM values 
CGM_min_values=meal_P1.min(axis=1)
CGM_min_values=CGM_min_values.to_frame()    
CGM_B=CGM_min_values.values.tolist()
CGM_B_min = [ item for elem in CGM_B for item in elem] 
#print(CGM_B_min,len(CGM_B_min),'hhhhhhhhhhhhhhhhhhhhhhhhhhhh')

#CGM meal values
#print(len(CGM_B_meal_values))
meal_Cgm_values=[]
for point in CGM_B_meal_values:
    x=point[0][0]
    meal_Cgm_values.append(x)
meal_time=meal_P1[23]
#print(meal_time)
meal_P1['min_value']=CGM_min_values
meal_P1['meal_check']=meal_Cgm_values
meal_P1['meal_value']=meal_time
meal_P1['max_value']=CGM_max_values


meal_P1['bolus_values']=rounded_tValues

meal_P1=meal_P1.dropna()

CGM_minvalue=meal_P1['min_value'].min()
CGM_maxvalue=meal_P1['max_value'].max()
#print(CGM_minvalue,CGM_maxvalue)
q=(CGM_maxvalue-CGM_minvalue)//20

def create_bins(lower_bound, width, quantity):
    bins = []
    for low in range(lower_bound, 
                     lower_bound + quantity*width + 1, width):
        bins.append((low, low+width))
    return bins
bins = create_bins(lower_bound=int(CGM_minvalue),
                   width=20,
                   quantity=int(q))
def find_bin(value, bins):
    for i in range(0, len(bins)):
        if bins[i][0] <= value < bins[i][1]:
            return i
    return -1


binned_weights_CGM_max = []
CGM_max_bins=[]

for value in meal_P1['max_value']:
    bin_index = find_bin(value, bins)
    CGM_max_bins.append(bin_index)
    #binned_weights_CGM_max.append(bin_index)
    
frequencies = Counter(binned_weights_CGM_max)

binned_meals_CGM_max = []
Meal_bins=[]
for value in meal_P1['meal_value']:
    bin_index = find_bin(value, bins)
    #print(value, bin_index, bins[bin_index])
    Meal_bins.append(bin_index)
    binned_meals_CGM_max.append(bin_index)


meal_P1['max_bin_values']=CGM_max_bins
meal_P1['Meal_bin_value']=Meal_bins




def item_sets(data):
    
    itemset = pd.DataFrame()
    itemset['CGM_max']=data['max_bin_values']
    itemset['CGM_Meal']=data['Meal_bin_value']
    itemset['Bolus_value']=data['bolus_values']
    return itemset
itemset=item_sets(meal_P1)
#print(itemset)

def frequent_item_sets(data):
    data_group = data.groupby(['CGM_max','CGM_Meal','Bolus_value'],sort=True).size().reset_index(name="Count")
    return data_group
d_group=frequent_item_sets(itemset)
d_max_freq = d_group["Count"].max()
d_group.to_csv("Frequentitemsets.csv")

def largestcountfrequentsets(list_frequent_sets,data,x):
    l = len(data)
    for i in range(0,l):
        if data.iloc[i,3] == x:
            list_frequent_sets.append("{"+str(data.iloc[i,0])+","+str(data.iloc[i,1])+","+str(data.iloc[i,2])+"}")
            
list_frequent_itemsets = [] 
largestcountfrequentsets(list_frequent_itemsets,d_group,d_max_freq)
lfsets = pd.DataFrame(list_frequent_itemsets)
#lfsets.to_csv('Most_Frequent_ItemSets.csv',index=False, header=False)

def countandconfidence(data):
    U_c = data.drop_duplicates()
    rep = []
    confidence = []
    for idx, row in U_c.iterrows():
            cmp = data[data['CGM_max'] == row['CGM_max']]
            cmp = cmp[cmp['CGM_Meal'] == row['CGM_Meal']]
            shape1 = cmp.shape[0]
            cmp = cmp[cmp['Bolus_value'] == row['Bolus_value']]
            shape2 = cmp.shape[0]
            rep.append(shape2)
            confidence.append(float(shape2/shape1))
    U_c['repititions'] = rep
    U_c['confidence'] = confidence
    return U_c
y=countandconfidence(itemset)
#y.to_csv('confidence_set.csv')

d_conf_max = y["confidence"].max()


def find__rules(list_rules, data,x):
    l = len(data)
    for i in range(0,l):
        number = data.iloc[i,4]
        if number == x:
            list_rules.append("{"+str(data.iloc[i,0])+","+str(data.iloc[i,1])+"->"+str(data.iloc[i,2])+"}")


list_rules = []
find__rules(list_rules, y, d_conf_max) 
lrules = pd.DataFrame(list_rules)
lrules.to_csv('LargestConfidenceRules.csv',index=False,header=False) 
          
def anomalous_values(list_anomalous_rules, data,x):
    l = len(data)
    for i in range(0,l):
        number = data.iloc[i,4]
        #print(number)
        if number< x:
            list_anomalous_rules.append("{"+str(data.iloc[i,0])+","+str(data.iloc[i,1])+"->"+str(data.iloc[i,2])+"}") 
            
list_anomalous_rules = []
x=0.15 #15 percent is 0.15
anomalous_values(list_anomalous_rules, y, x)
arules = pd.DataFrame(list_anomalous_rules)
arules.to_csv('AnomalousRules.csv',index=False,header=False)
            


