# Generated from: DatetimeFeatures.ipynb
# Warning: This is an auto-generated file. Changes may be overwritten.

# # Datetime variable transformation
#
# The **DatetimeFeatures()** transformer is able to extract many different datetime features from existing datetime variables present in a dataframe. Some of these features are numerical, such as month, year, day of the week, week of the year, etc. and some are binary, such as whether that day was a weekend day or was the last day of its correspondent month. All features are cast to integer before adding them to the dataframe. <br>
# DatetimeFeatures() converts datetime variables whose dtype is originally object or categorical to a datetime format, but it does not work with variables whose original dtype is numerical. <br>
#
# For this demonstration, we use the Metro Interstate Traffic Volume Data Set, which is publicly available at https://archive.ics.uci.edu/ml/datasets/Metro+Interstate+Traffic+Volume


#for starters, we import the relevant modules and the DatetimeFeatures class
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from feature_engine.datetime import DatetimeFeatures


#load and inspect the dataset
data = pd.read_csv('Metro_Interstate_Traffic_Volume.csv')

data.head()


data.shape


# Inspect the columns typing and check for potentially missing values
pd.DataFrame({"type":data.dtypes, "nan count":data.isna().sum()})


# As it seems, this dataset only contains one datetime variable (named, indeed, _date\_time_). <br>
# Let's say we wanted to extract the _day of the month_ and the _hour_ features from it.
# Since _date\_time_ happens to be the only datetime variable in this dataset, we can do either of the following
# - let the transformer search for all datetime variables by initializing it with variables=None (which is the default option anyway)
# - specify which variables are going to be processed, which in this case would be setting variables="date_time"


dtfs = DatetimeFeatures(
    variables=None,
    features_to_extract=["day_of_month", "hour"]
)

# as per scikit-learn and feature-engine convention, we call the fit and transform method
# to process the data (even though this particular transformer does not learn any parameters)
dtfs.fit(data)


# check which variables have been picked up as datetime during fit
dtfs.variables_


data_transf = dtfs.transform(data)
data_transf.head()


# Notably, the dataframe identified that the object-like _date\_time_ variable could be cast to datetime and acquired the two columns _date\_time\_dotm_ and _date\_time\_hour_ corresponding to the features we required through the _features\_to\_extract_ argument. <br>
# **Note**: the original _date\_time_ column was removed from the dataframe in the process, as per default behaviour. If we want to keep it, we need to initialize the transformer passing drop_original=False.


# this time we specify what variable(s) we want the features extracted from
# we also want to keep the original datetime variable(s).
dtfs = DatetimeFeatures(
    variables="date_time",
    features_to_extract=["day_of_month", "hour"],
    drop_original=False
)

data_transf = dtfs.fit_transform(data)


data_transf.head()


# There are many more datetime features that DatetimeFeatures() can extract; see the docs for a full list. <br>
# The argument _features\_to\_extract_ has a default option aswell. Let's quickly see what it does.


dtfs = DatetimeFeatures(features_to_extract=None)

data_transf = dtfs.fit_transform(data)


# only show columns that were extracted from date_time
data_transf.filter(regex="date_time*").head()


# As shown above, DatetimeFeatures() extracts _month_, _year_, _day of the week_, _day of the month_, _hour_, _minute_ and _second_ by default. <br>
# **Note**: when a variable only contains date information all the time features default to 00:00:00 time; conversely, when a variable only contains time information, date features default to today's date at the time of calling the transform method.
#
# If we really want to extract _all_ of the available features we can set _features\_to\_extract_ to the special value "all". Beware, though, as your feature space might grow significantly and most of the extracted features are most likely not going to be too relevant.


dtfs = DatetimeFeatures(features_to_extract="all")

data_transf = dtfs.fit_transform(data)


data_transf.filter(regex="date_time*").head()


# Another thing to keep in mind is that oftentimes most of these features are going to be quasi-constant if not constant altogether. This can be for several reason, most likely due to the particular time window in which the data was collected. <br>
# We can thus combine the DatetimeFeatures() and DropConstantFeatures() transformers from feature_engine in a scikit-learn pipeline to automatically get rid of features we deem irrelevant to our analysis.


# data.drop('holiday', axis=1, inplace = True)
data['holiday'] = data['holiday'].replace({pd.NA: None, pd.NaT: None, np.nan: None})
data_for_pipe = data.drop('holiday', axis=1)


from sklearn.pipeline import Pipeline
from feature_engine.selection import DropConstantFeatures

pipe = Pipeline([
    ('datetime_extraction', DatetimeFeatures(features_to_extract=["year", "day_of_month", "minute", "second"])),
    ('drop_constants', DropConstantFeatures())
])



# print(data.isnull().sum()[data.isnull().sum() > 0])


data_transf = pipe.fit_transform(data_for_pipe)


data_transf.head()


# Since all data was gathered with only hour-precision, the _minute_ and _second_ features we had requested were extracted by DatetimeFeatures() but subsequently dropped by DropConstantFeatures(). This way we can avoid our feature space to become overly cluttered with useless information even when we are not being particularly diligent with the features we request to extract.

