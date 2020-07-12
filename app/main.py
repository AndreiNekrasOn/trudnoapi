import pickle

import scipy
from flask import Flask, request, redirect, url_for, flash, jsonify, render_template
import numpy as np
import pandas as pd
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter

import json

from catboost import CatBoostRegressor

import os

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

from config import Config
from app.forms import VacancyData

app = Flask(__name__)
app.config.from_object(Config)


@app.route('/api', methods=['POST'])
def predict_salary():
    print('Got request')
    data = request.get_json()
    if data is None:  # e.g. data send via form
        data = json.loads(request.form.get('json_data'))

    print('Got data')
    data = {k: [v] for k, v in data.items()}
    print(data)

    df = pd.DataFrame.from_dict(data)
    print("Converted data to dataframe")
    print(df)
    df = process_data(df)
    print("Processed data")
    print(df)
    try:
        prediction = np.array2string(model.predict(df))
    except KeyError:
        return render_template('index.html'), 400
    print("Made prediction")
    print(prediction)

    return jsonify(prediction)


@app.route('/')
def show_form():
    form = VacancyData()
    return render_template('index.html', title='TruDno', form=form)


def process_data(df):
    add_specification_features(df)
    df = spec_modif_ten(df)
    df = add_city_coord_feature(df)
    df = add_len_feature(df)
    final_df = df.copy()
    final_df = tf_idf_as_feature(final_df)
    return df


def parse_date(df, name_column_date):
    series_date_datetime = pd.to_datetime(df[name_column_date])
    df.loc[:, 'year'] = series_date_datetime.apply(lambda x: x.year)
    df.loc[:, 'month'] = series_date_datetime.apply(lambda x: x.month)
    df.loc[:, 'day'] = series_date_datetime.apply(lambda x: x.day)
    df.loc[:, 'hour'] = series_date_datetime.apply(lambda x: x.hour)


def clean_data(df):
    # drop useless columns
    data_vacs_clean = df.drop(columns=['id', 'name', 'area.name', 'company_link', 'salary_currency',
                                       'employment.name', 'schedule.name', 'experience.name',
                                       'description', 'type'])
    parse_date(data_vacs_clean, 'publication_date')
    data_vacs_clean = data_vacs_clean.drop(columns=['publication_date'])
    # simplify key_skills format
    data_vacs_clean.loc[:, "key_skills"] = (data_vacs_clean.key_skills.astype(str) != "nan").astype(int)
    return data_vacs_clean


def spec_modif(elem):
    lst = elem.split()
    for i in range(len(lst)):
        lst[i] = int(lst[i].split('.')[0])

    first_mode = max(set(lst), key=lst.count)
    second_mode = 0

    if len(set(lst)) >= 2:
        lst = list(filter(lambda a: a != first_mode, lst))
        second_mode = max(set(lst), key=lst.count)

    return first_mode, second_mode


def add_specification_features(df):
    df["first_spec"] = 0
    df["second_spec"] = 0
    for index, row in (df.iterrows()):
        df.loc[index, ["first_spec", "second_spec"]] = spec_modif(row["specializations"])
    df["spec_split"] = df["specializations"].apply(lambda x: " ".join(str(x).split('.')))


def spec_modif_ten(df):  # df was from vacs_train_clean_adv.csv. Load it?
    df_new = df.copy()
    # new columns for specializations
    for i in range(10):
        df_new['spec' + str(i) + '**'] = 0

    # fill these columns
    for index, row in (df_new.iterrows()):
        spec_lst = list(row.specializations.split())
        for spec in spec_lst:
            spec_num = int(spec.split('.')[1])
            df_new.loc[index, 'spec' + str(spec_num // 100) + '**'] = spec_num

    # fill empty cities
    df_new = df_new.fillna('unknown')
    return df_new


def add_city_coord_feature(df):  # this is the crapiest function i've ever seen
    df_city = df.copy()
    df_city['city'].fillna(value='Cанкт-Петербург', inplace=True)
    df_city.loc[df_city['city'] == 'unknown', 'city'] = 'Cанкт-Петербург'
    if df_city is None:
        raise AssertionError
    geolocator = Nominatim(user_agent='trudnoapi')
    geocode = RateLimiter(geolocator.geocode, min_delay_seconds=0.1)

    def get_adress(adr):
        loc = geocode(adr)
        if loc is not None:
            return ','.join(map(str, [loc.latitude, loc.longitude]))
        else:
            return '59.9606739,30.1586551'

    uniq_city = pd.Series(df_city['city'].unique())
    location = uniq_city.apply(get_adress)

    mask = pd.Series(location.to_list(), index=uniq_city)
    coord_city = df_city['city'].apply(lambda x: mask[x]).str.split(',', expand=True)
    df_city['coord_lat'] = pd.to_numeric(coord_city[0])
    df_city['coord_lon'] = pd.to_numeric(coord_city[1])
    return df_city


def add_len_feature(df):
    df_len = df.copy()
    df_len["description_len"] = df_len["description.lemm"].apply(len)
    return df_len


def tf_idf_predictions(df):
    X_train_text_1 = vectorizer_1.transform(df.loc[:, "description.lemm"])
    X_train_text_2 = vectorizer_2.transform(df.loc[:, "name.lemm"])
    X_train_text_3 = vectorizer_3.transform(df.loc[:, "city"])
    X_train_text_4 = vectorizer_4.transform(df.loc[:, "company"])
    X_train_text_5 = vectorizer_5.transform(df.loc[:, "key_skills"])
    X_train_text_6 = vectorizer_6.transform(df.loc[:, "specializations.names"])


    X_train_full = scipy.sparse.hstack([X_train_text_1, X_train_text_2, X_train_text_3,
                                        X_train_text_4, X_train_text_5, X_train_text_6]).tocsr()

    print(X_train_full.shape)

    y_pred_tf_idf = tf_idf_catboost.predict(X_train_full)
    return y_pred_tf_idf


def tf_idf_as_feature(df):
    y_pred_tf_idf = tf_idf_predictions(df)  # !!! change iterations
    y_pred_tf_idf = y_pred_tf_idf.reshape((len(y_pred_tf_idf), 1))
    x_wit_tfidf_pred = np.concatenate([df, y_pred_tf_idf], axis=1)
    return x_wit_tfidf_pred


if __name__ == '__main__':
    model_file = 'models/catboost.cbm'
    tf_idf_model_file = 'models/Tf-IdfCatNotFull.cbm'

    vectorizer_1 = pickle.load(open('models/vectorizer_1.pkl', 'rb'))
    vectorizer_2 = pickle.load(open('models/vectorizer_2.pkl', 'rb'))
    vectorizer_3 = pickle.load(open('models/vectorizer_3.pkl', 'rb'))
    vectorizer_4 = pickle.load(open('models/vectorizer_4.pkl', 'rb'))
    vectorizer_5 = pickle.load(open('models/vectorizer_5.pkl', 'rb'))
    vectorizer_6 = pickle.load(open('models/vectorizer_6.pkl', 'rb'))

    model = CatBoostRegressor()
    model.load_model(model_file, "cbm")
    tf_idf_catboost = CatBoostRegressor()
    tf_idf_catboost.load_model(tf_idf_model_file)
    app.run(debug=True, host='0.0.0.0', port=5000)
