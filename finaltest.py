import disaggregate
from __future__ import print_function, division
import itertools
from copy import deepcopy
from collections import OrderedDict
from warnings import warn
from datetime import datetime
import matplotlib.pyplot as plt 
import pandas as pd
import numpy as np
from six import iteritems
from builtins import range
from disaggregate import fhmm_exact

def convert_to_datetime(x):
    return datetime.fromtimestamp(x)

def resample_data(df,resample_freq):
    avg_df = df.resample(resample_freq).mean()
    return avg_df

app_list=['refrigerator', 'guest_ac', 'guest_lightfans', 'living_ac', 'conf_lights', 'conferenceroom_AC']

total_power=OrderedDict()
for name in app_list:
    filename=name + '.csv'
    trainset = pd.read_csv(filename,names = ['unix_date','meter_reading'])
    trainset['date'] = trainset['unix_date'].apply(convert_to_datetime)
    trainset = trainset.set_index('date').drop('unix_date', axis = 1)
    trainset['time'] = trainset.index.time
    total_power[name]=trainset['meter_reading']*1000
    
fhmm = fhmm_exact.FHMM()
fhmm.train(total_power)

dataset = pd.read_csv('test_file.csv')
dataset.columns=['unix_date', 'refrigerator', 'guest_ac', 'guest_lightfans', 'microwave', 'living_ac', 'conf_lights', 'conferenceroom_AC', 'control_room']
#dataset = pd.read_csv(filename,names = ['unix_date', 'refrigerator', 'guest_ac', 'guest_lightfans', 'microwave', 'living_ac', 'conf_lights', 'conferenceroom_AC', 'control_room'])
dataset['date'] = dataset['unix_date'].apply(convert_to_datetime)
dataset = dataset.set_index('date').drop('unix_date', axis = 1)
dataset['time'] = dataset.index.time
dataset=dataset.set_index('time')
#dataset.columns=['index', 'refrigerator', 'guest_ac', 'guest_lightfans', 'microwave', 'living_ac', 'conf_lights', 'conferenceroom_AC', 'control_room']
#dataset['Date_time']=dataset['time'].apply(convert_to_datetime)
power_total=np.zeros((1439,1))
for name in app_list:
    print(name)
    dataset[name]=dataset[name]*1000
    power_total=power_total+dataset[name].reshape(-1,1)

df=pd.DataFrame(power_total)
result=fhmm.disaggregate_chunk(df)
result['time']=dataset.index.values
result=result.set_index('time')

'''fig = plt.figure()
#plt.figure(figsize=(6,1))
ax = fig.add_subplot(111)
ax.plot(dataset['time'], dataset['guest_lightfans'], 'r')
# title and labels, setting initial sizes
fig.suptitle('Actual Power Consumption of Refrigerator', fontsize=15)
ax.set_xlabel('Time in HH:MM:SS', fontsize=13)
ax.set_ylabel('Power in Watt', fontsize=13)
plt.show()

fig = plt.figure()
#plt.figure(figsize=(6,1))
ax = fig.add_subplot(111)
ax.plot(dataset['time'][:1439], result['refrigerator'], 'g')
# title and labels, setting initial sizes
fig.suptitle('Predicted Power Consumption of Refrigerator', fontsize=15)
ax.set_xlabel('Time in HH:MM:SS', fontsize=13)
ax.set_ylabel('Power in Watt', fontsize=13)
plt.show()

fig = plt.figure()
#plt.figure(figsize=(6,1))
ax = fig.add_subplot(111)
ax.plot(dataset['time'], result['guest_ac'], 'r')
# title and labels, setting initial sizes
fig.suptitle('Actual Power Consumption of Guest Room AC', fontsize=15)
ax.set_xlabel('Time in HH:MM:SS', fontsize=13)
ax.set_ylabel('Power in Watt', fontsize=13)
plt.show()

fig = plt.figure()
#plt.figure(figsize=(6,1))
ax = fig.add_subplot(111)
ax.plot(dataset['time'][:1439], result['refrigerator'], 'g')
# title and labels, setting initial sizes
fig.suptitle('Predicted Power Consumption of Refrigerator', fontsize=15)
ax.set_xlabel('Time in HH:MM:SS', fontsize=13)
ax.set_ylabel('Power in Watt', fontsize=13)
plt.show()'''



plt.plot(dataset['refrigerator'], 'r', label="Predicted")
plt.ylabel('Power in Watts', fontsize=15)
plt.xlabel('Time in HH:MM:SS', fontsize=15)
plt.title('Actual Power Consumption of Refrigerator', fontsize=15)

plt.plot(result['refrigerator'], 'g', label="Predicted")
plt.ylabel('Power in Watts', fontsize=15)
plt.xlabel('Time in HH:MM:SS', fontsize=15)
plt.title('Predicted Power Consumption of Refrigerator', fontsize=15)


plt.plot(dataset['guest_ac'], 'r', label="Predicted")
plt.ylabel('Power in Watts', fontsize=15)
plt.xlabel('Number of Samples with 2Min Sampling rate', fontsize=15)
plt.title('Actual Power Consumption of Guest Room AC', fontsize=15)

plt.plot(result['guest_ac'], 'g', label="Predicted")
plt.ylabel('Power in Watts', fontsize=15)
plt.xlabel('Number of Samples with 2Min Sampling rate', fontsize=15)
plt.title('Predicted Power Consumption of Guest Room AC', fontsize=15)

plt.plot(dataset['guest_lightfans'], 'r', label="Predicted")
plt.ylabel('Power in Watts', fontsize=15)
plt.xlabel('Number of Samples with 2Min Sampling rate', fontsize=15)
plt.title('Predicted Power Consumption of Guest Room Light & Fans', fontsize=15)

plt.plot(result['guest_lightfans'], 'g', label="Predicted")
plt.ylabel('Power in Watts', fontsize=15)
plt.xlabel('Number of Samples with 2Min Sampling rate', fontsize=15)
plt.title('Predicted Power Consumption of Guest Room Light & Fans', fontsize=15)

plt.plot(dataset['microwave'], 'r', label="Predicted")
plt.ylabel('Power in Watts')
plt.xlabel('Number of Samples')
plt.title('Actual Power of Microwave') 

plt.plot(dataset['living_ac'], 'r', label="Predicted")
plt.ylabel('Power in Watts', fontsize=15)
plt.xlabel('Number of Samples with 2Min Sampling rate', fontsize=15)
plt.title('Actual Power Consumption of Hall AC', fontsize=15)


plt.plot(result['living_ac'], 'g', label="Predicted")
plt.ylabel('Power in Watts', fontsize=15)
plt.xlabel('Number of Samples with 2Min Sampling rate', fontsize=15)
plt.title('Predicted Power Consumption of Hall AC', fontsize=15)

plt.plot(dataset['conf_lights'], 'r', label="Predicted")
plt.ylabel('Power in Watts')
plt.xlabel('Number of Samples')
plt.title('Actual Power of Conference Room Lights') 

plt.plot(result['conf_lights'], 'g', label="Predicted")
plt.ylabel('Power in Watts')
plt.xlabel('Number of Samples')
plt.title('Predicted Power of Conference room Lights')

plt.plot(dataset['conferenceroom_AC'], 'r', label="Predicted")
plt.ylabel('Power in Watts')
plt.xlabel('Number of Samples')
plt.title('Actual Power of Conference Room AC') 


plt.plot(result['conferenceroom_AC'], 'g', label="Predicted")
plt.ylabel('Power in Watts')
plt.xlabel('Number of Samples')
plt.title('Predicted Power of Conference room AC')'''