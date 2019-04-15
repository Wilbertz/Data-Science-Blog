import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import collections
from itertools import chain
import sklearn
from time import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVR
import seaborn as sns

from mpl_toolkits.basemap import Basemap


def prepare_listings():
    """
    Cleanup of Airbnb data.
    :return: None
    """
    df= pd.read_csv("data/listings.csv")

    df.drop(
        columns=[
            'neighborhood_overview',
            'notes',
            'square_feet',
            'weekly_price',
            'monthly_price',
            'license'],
        inplace=True)

    df['security_deposit'] = df['security_deposit'].fillna(0)
    df['cleaning_fee'] = df['cleaning_fee'].fillna(0)

    df.drop(columns=['id'], inplace=True)

    unique_cols = [col for col in df.columns.values if df[col].nunique() == 1]
    df.drop(columns=unique_cols, inplace=True)

    url_cols = [col for col in df.columns.values if 'url' in col]
    df.drop(columns=url_cols,inplace=True)

    host_cols = ['host_id', 'host_name', 'host_since', 'host_location', 'host_about', 'host_verifications', 'host_listings_count']
    df.drop(columns=host_cols, inplace=True)

    location_cols = ['city', 'street', 'state', 'smart_location', 'latitude', 'longitude']
    df.drop(columns=location_cols, inplace=True)

    nb_cols = ['neighbourhood', 'host_neighbourhood', 'neighbourhood_cleansed']
    df.drop(columns=nb_cols, inplace=True)

    date_cols = ['first_review', 'last_review', 'calendar_updated']
    df.drop(columns=date_cols, inplace=True)

    text_cols = ['name', 'summary', 'space', 'description', 'transit']
    df.drop(columns=text_cols, inplace=True)

    df_amenities = df['amenities'].str.split(',', expand=True)
    df_amenities = df_amenities.replace('[^\w\s]', '', regex=True)
    amenities_unique = [df_amenities[x].unique().tolist() for x in df_amenities.columns.values]
    amenities_unique = set(list(chain.from_iterable(amenities_unique)))

    amenities_unique.remove('')
    amenities_unique.remove(None)

    for index in range(30):
        for a in amenities_unique:
            df_amenities[a] = 0

    for index in range(30):
        for a in amenities_unique:
            df_amenities[a] += np.where(df_amenities[index] == a, 1, 0)

    # Drop redundant column
    df_amenities.drop(columns=[x for x in range(30)], inplace=True)

    df = df.join(df_amenities, how='left')

    # Drop amenities column
    df.drop(columns='amenities', inplace=True)

    cols = ['host_response_rate', 'host_acceptance_rate']
    df[cols] = df[cols].replace('%', '', regex=True).astype(float)

    cols = ['price', 'security_deposit', 'cleaning_fee', 'extra_people']
    df[cols] = df[cols].replace('[\$,]', '', regex=True).astype(float)

    binary_cols = ['host_is_superhost', 'host_has_profile_pic', 'host_identity_verified', 'is_location_exact'
        , 'instant_bookable', 'require_guest_profile_picture', 'require_guest_phone_verification']
    df[binary_cols] = np.where(df[binary_cols] == 't', 1, 0)

    categorical_cols = ['host_response_time', 'neighbourhood_group_cleansed', 'zipcode'
        , 'property_type', 'room_type', 'bed_type', 'cancellation_policy']

    df = pd.get_dummies(data=df, columns=categorical_cols, drop_first=True)

    df.drop(columns='host_acceptance_rate', inplace=True)

    return df, df_amenities


def train_and_predict(learner, sample_size, X_train, y_train, X_test, y_test):
    '''
    inputs:
       - learn: the learning algorithm used for training and prediction
       - size: the size of samples (number) to be drawn from training set
       - X_train: features training set
       - y_train: review_scores_rating training set
       - X_test: features testing set
       - y_test: review_scores_rating testing set
    '''
    results = {}

    # Fit the learner to the training data and get training time
    start = time()
    learner = learner.fit(X_train[:sample_size], y_train[:sample_size])
    end = time()
    results['train_time'] = end - start

    # Get predictions on the test set(X_test), then get predictions on first 300 training samples
    start = time()
    predictions_test = learner.predict(X_test)
    predictions_train = learner.predict(X_train[:300])
    end = time()

    # Calculate the total prediction time
    results['pred_time'] = end - start

    # Compute accuracy on the first 300 training samples
    results['mse_train'] = mean_squared_error(y_train[:300], predictions_train)

    # Compute accuracy on test set
    results['mse_test'] = mean_squared_error(y_test, predictions_test)

    print("{} trained on {} samples.".format(learner.__class__.__name__, sample_size))
    print("MSE_train: %.4f" % results['mse_train'])
    print("MSE_test: %.4f" % results['mse_test'])
    print("Training score:%.4f" % learner.score(X_train, y_train))
    print("Test score:%.4f" % learner.score(X_test, y_test))
    return results


def plot():
    """
    This methods reads in neighbourhood data from
    the file data/neighbourhood.csv and create a file seattle_prices.png, that
    contains a Geoplot of Seattle Airbnb prices.
    :return: None
    """
    df = pd.read_csv("data/neighbourhood.csv", sep=";")
    fig, ax = plt.subplots(figsize=(20, 20))
    m = Basemap(projection='lcc', resolution='h', lat_0=47.63, lon_0=-122.3, width=2E4, height=3E4)

    m.drawcoastlines()
    m.drawrivers()
    m.drawstates()

    lon = df['longitude'].values
    lat = df['latitude'].values
    price = df['price'].values
    count = df['count'].values
    neighbourhood = df['neighbourhood'].values

    m.scatter(lon, lat, latlon=True, c=price, s=count/15, cmap='Reds')
    cbar = plt.colorbar(label='Price')
    cbar.ax.set_ylabel('Price', fontsize=20)

    for i, txt in enumerate(neighbourhood):
        x, y = m(lon[i], lat[i])
        plt.text(x, y, txt)

    ax.set_xlabel('Longitude', fontsize=20)
    ax.set_ylabel('Latitude', fontsize=20)
    plt.title('Seattle Airbnb Prices', fontsize=40)

    ax.xaxis.label.set_size(20)
    ax.yaxis.label.set_size(20)

    for a in [100, 1000, 5000]:
        plt.scatter([], [], c='red', alpha=0.5, s=a/15, label=str(a))
    plt.scatter([], [], c='red', alpha=0.5, s=0, label='Bookings')
    plt.legend(scatterpoints=1, frameon=False, labelspacing=1, loc='lower right')

    plt.savefig('seattle_prices.png')


if __name__ == "__main__":
    plot()
