from preprocessing import preprocessing
from darts.models import NBEATSModel
from darts.dataprocessing.transformers import Scaler
from darts.utils.missing_values import fill_missing_values
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from darts import TimeSeries
from darts.metrics import mape, mase, smape
import random
#import warnings

#warnings.filterwarnings('ignore')

# Get data.
data = pd.read_csv('bq-results-20231122-143519-1700663741796.csv')
members = random.choices(data.member_token, k=2)
member_to_predict = random.choices(data.member_token, k=1)
print(member_to_predict)

print(data.member_token)
print(member_to_predict)

# Prep data.
train_list = []
val_list = []

for member in members:
    # Get member data.
    data_of_member = data[data.member_token == member]
    # Monthly binning.
    data_of_member = preprocessing.monthly_binning(data_of_member)
    # Get historical sum and term.
    historical_sum, term = preprocessing.set_term(data_of_member)
    # Convert to timeseries.
    data_of_member.reset_index(inplace=True)
    series = TimeSeries.from_dataframe(data_of_member, time_col='posted_date', value_cols=['total_earned'], fill_missing_dates=True)
    # Split the member data according to term.
    series = series.astype(np.float32)
    train, val = series[:(term)], series[(term):]
    # Manage term as validation.
    if len(train) == 0 or len(val) == 0 or len(val) == 1 or term == 0:
        print('Not enough data.')
        continue
    else:
        if (len(val) < term):
            term = len(val)
        if len(val) > term:
            train, val = series[:(len(data_of_member) - term)], series[(len(data_of_member) - term):]



    #series = TimeSeries.from_dataframe(member_data_prep, time_col='posted_period', fill_missing_dates=True, freq='M')
    #series_val = TimeSeries.from_dataframe(member_data_prep_val, time_col='posted_period', fill_missing_dates=True,
    #                                       freq='M')

    #train.plot()
    #val.plot()
    #plt.show()

    train_list.append(train)
    val_list.append(val)

# Scale data.
train_scaler = Scaler()
scaled_train = train_scaler.fit_transform(train_list)

#Get rid of NaN values.
scaled_train_nonan = []
for sctr in scaled_train:
    scaled_train_nonan.append(fill_missing_values(sctr))


# Train the model.
nbeats = NBEATSModel(
    input_chunk_length=6,
    output_chunk_length=6,
    generic_architecture=True,
    random_state=42)

nbeats.fit(
    scaled_train_nonan,
    epochs=50)

for member in member_to_predict:
    # Get member data.
    data_of_member = data[data.member_token == member_to_predict]

print(data_of_member)

historical_sum, term = preprocessing.set_term(data_of_member)
data_of_member.reset_index(inplace=True)
srs = TimeSeries.from_dataframe(data_of_member, time_col='posted_date', value_cols=['total_earned'],
                                   fill_missing_dates=True)
srs_scaled= train_scaler.fit_transform(srs)

# Predict.
scaled_pred_nbeats = nbeats.predict(n=term, series= srs_scaled)

# Inverse transform.
pred_nbeats = train_scaler.inverse_transform(scaled_pred_nbeats)



for ind in np.arange(0, len(pred_nbeats)):

    pred_nbeats[ind].plot()
    train_list[ind].plot()
    val_list[ind].plot()


plt.show()

#for ts in pred_nbeats:
#    ts.plot()
#    plt.show()


#plt.plot(train_list)
#plt.plot(val_list)
#plt.plot(pred_nbeats)

plt.show()





