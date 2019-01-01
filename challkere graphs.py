import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data_ref=pd.read_csv('chalkere.csv')

from datetime import datetime
def convert_to_datetime(x):
    return datetime.fromtimestamp(x)

data_ref['date'] = data_ref['TIME in UTC seconds'].apply(convert_to_datetime)
data=data_ref[:82]

y_data=data['Ract_pwr(kW)']
x_data=data['date']

fig = plt.figure()
#plt.figure(figsize=(6,1))
ax = fig.add_subplot(111)
ax.plot(x_data, y_data)
ax.xlabel('Time')
ax.ylabel('')
plt.show()