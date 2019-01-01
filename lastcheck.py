from __future__ import print_function, division
import disaggregate

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
    dataset = pd.read_csv(filename,names = ['unix_date','meter_reading'])
    dataset['date'] = dataset['unix_date'].apply(convert_to_datetime)
    dataset = dataset.set_index('date').drop('unix_date', axis = 1)
    dataset['time'] = dataset.index.time
    total_power[name]=dataset['meter_reading']*1000
    
fhmm = fhmm_exact.FHMM()
fhmm.train(total_power)

testset = pd.read_csv('testingfile.csv')
testset.columns=['unix_date', 'refrigerator', 'guest_ac', 'guest_lightfans', 'microwave', 'living_ac', 'conf_lights', 'conferenceroom_AC', 'control_room']
testset*1000
power_total=np.zeros((1439,1))
for name in app_list:
    print(name)
    testset[name]=testset[name]*1000
    power_total=power_total+testset[name].reshape(-1,1)
    
df=pd.DataFrame(power_total)
result=fhmm.disaggregate_chunk(df)

plt.plot(testset['refrigerator'], 'r', label="Predicted")
plt.ylabel('Power in Watts', fontsize=15)
plt.xlabel('Time in HH:MM:SS', fontsize=15)
plt.title('Actual Power Consumption of Refrigerator', fontsize=15)

plt.plot(result['refrigerator'], 'g', label="Predicted")
plt.ylabel('Power in Watts', fontsize=15)
plt.xlabel('Time in HH:MM:SS', fontsize=15)
plt.title('Predicted Power Consumption of Refrigerator', fontsize=15)


plt.plot(testset['guest_ac'], 'r', label="Predicted")
plt.ylabel('Power in Watts', fontsize=15)
plt.xlabel('Number of Samples with 2Min Sampling rate', fontsize=15)
plt.title('Actual Power Consumption of Guest Room AC', fontsize=15)

plt.plot(result['guest_ac'], 'g', label="Predicted")
plt.ylabel('Power in Watts', fontsize=15)
plt.xlabel('Number of Samples with 2Min Sampling rate', fontsize=15)
plt.title('Predicted Power Consumption of Guest Room AC', fontsize=15)

plt.plot(testset['guest_lightfans'], 'r', label="Predicted")
plt.ylabel('Power in Watts', fontsize=15)
plt.xlabel('Number of Samples with 2Min Sampling rate', fontsize=15)
plt.title('Actual Power Consumption of Guest Room Light & Fans', fontsize=15)

plt.plot(result['guest_lightfans'], 'g', label="Predicted")
plt.ylabel('Power in Watts', fontsize=15)
plt.xlabel('Number of Samples with 2Min Sampling rate', fontsize=15)
plt.title('Predicted Power Consumption of Guest Room Light & Fans', fontsize=15)

plt.plot(testset['living_ac'], 'r', label="Predicted")
plt.ylabel('Power in Watts', fontsize=15)
plt.xlabel('Number of Samples with 2Min Sampling rate', fontsize=15)
plt.title('Actual Power Consumption of Hall AC', fontsize=15)


plt.plot(result['living_ac'], 'g', label="Predicted")
plt.ylabel('Power in Watts', fontsize=15)
plt.xlabel('Number of Samples with 2Min Sampling rate', fontsize=15)
plt.title('Predicted Power Consumption of Hall AC', fontsize=15)

plt.plot(testset['conf_lights'], 'r', label="Predicted")
plt.ylabel('Power in Watts')
plt.xlabel('Number of Samples')
plt.title('Actual Power of Conference Room Lights') 

plt.plot(result['conf_lights'], 'g', label="Predicted")
plt.ylabel('Power in Watts')
plt.xlabel('Number of Samples')
plt.title('Predicted Power of Conference room Lights')

plt.plot(testset['conferenceroom_AC'], 'r', label="Predicted")
plt.ylabel('Power in Watts')
plt.xlabel('Number of Samples')
plt.title('Actual Power of Conference Room AC') 


plt.plot(result['conferenceroom_AC'], 'g', label="Predicted")
plt.ylabel('Power in Watts')
plt.xlabel('Number of Samples')
plt.title('Predicted Power of Conference room AC')