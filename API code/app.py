import os
import json
import pickle
import joblib
import pandas as pd
from flask import Flask, jsonify, request
from peewee import (
    SqliteDatabase, PostgresqlDatabase, Model, IntegerField,
    FloatField, TextField, BooleanField,IntegrityError
)
from playhouse.shortcuts import model_to_dict
from playhouse.db_url import connect
from DateTransformer import DateTransformer
from datetime import datetime
from sklearn.base import BaseEstimator, TransformerMixin
import gzip


########################################
# Begin database stuff

# The connect function checks if there is a DATABASE_URL env var.
# If it exists, it uses it to connect to a remote postgres db.
# Otherwise, it connects to a local sqlite db stored in predictions.db.
DB = connect(os.environ.get('DATABASE_URL') or 'sqlite:///predictions.db')
#DB = SqliteDatabase('predictions.db')

class Prediction(Model):
    observation_id = TextField(unique=True)
    observation = TextField()
    predicted_outcome = BooleanField()
    outcome = BooleanField(default=True)

    class Meta:
        database = DB


DB.create_tables([Prediction], safe=True)

# End database stuff
########################################

########################################
# Unpickle the previously-trained model


with open('columns.json') as fh:
    columns = json.load(fh)

with open('pipeline.pickle', 'rb') as f:
    pipeline = joblib.load(f)



with open('dtypes.pickle', 'rb') as fh:
    dtypes = pickle.load(fh)



# End model un-pickling
########################################

########################################
# Begin Checks

def check_request(request):
   
    if "observation_id" not in request:
        error = "Field `observation_id` missing from request: {}".format(request)
        return False, error
    

    if "Date" not in request:
        error = "Field `Date` missing from request: {}".format(request)
        return False, error
    
    if "Object of search" not in request:
        error = "Field `Object of search` missing from request: {}".format(request)
        return False, error
    
    if "station" not in request:
        error = "Field `station` missing from request: {}".format(request)
        return False, error
    
    if "Gender" not in request:
        error = "Field `Gender` missing from request: {}".format(request)
        return False, error
    
    if "Age range" not in request:
        error = "Field `Age range` missing from request: {}".format(request)
        return False, error
    
    if "Officer-defined ethnicity" not in request:
        error = "Field `Officer-defined ethnicity` missing from request: {}".format(request)
        return False, error
    

    return True, ""


def check_date_format(observation):
    date_str = observation.get('Date')
    if not date_str:
        error = "Field `Date` is missing"
        return False, error

    # Try parsing the date string using different format strings until one succeeds
    try:
        date_obj = datetime.strptime(date_str, '%Y-%m-%dT%H:%M:%S+00:00')
    except ValueError:
        try:
            date_obj = datetime.strptime(date_str, '%Y-%m-%d %H:%M:%S')
        except ValueError:
            try:
                date_obj = datetime.strptime(date_str, '%m/%d/%Y %H:%M:%S')
            except ValueError:
                error = "Field `Date` is in the wrong format: {}".format(date_str)
                return False, error

    if not (date_obj.month is not None and date_obj.day is not None and date_obj.hour is not None):
        error = "Field `Date` is missing month, day, or hour"
        return False, error

    if date_obj.hour < 0 or date_obj.hour > 24:
        error = "Field `Hour` year must be between 0 and 24"
        return False, error

    if date_obj.day < 1 or date_obj.day > 31:
        error = "Field `Day` year must be between 1 and 31"
        return False, error

    if date_obj.month < 1 or date_obj.month > 12:
        error = "Field `Month` year must be between 1 and 12" 
        return False, error

    return True, ""

    


def fill_missing_categorical_columns(observation):
    # Define the categorical columns that we want to check for missing values.
    categorical_columns = ['Part of a policing operation', 'Legislation']
    
    # Define the values that we want to use to fill missing categorical values.
    fill_values = {
        'Part of a policing operation': False,
        'Legislation': 'unknown'
    }
    
    # Check if any of the categorical columns are missing from the observation.
    for column in categorical_columns:
        if column not in observation:
            # If the column is missing, fill it with the most common value for that column.
            observation[column] = fill_values[column]
        elif column == 'Part of a policing operation':
            # If the column is present, convert the value to boolean type.
            observation[column] = bool(observation[column])
    
    return observation


def check_num(observation):
    lat = observation['Latitude']
    lon = observation['Longitude']
    
    if lat is None or lon is None:
        error = "Field `Latitude` or 'Longitude' missing"

        return False, error
    
    # Expected range of latitude and longitude for the UK
    lat_range = (49, 60)
    lon_range = (-10, 2)
    
    if not (lat_range[0] <= lat <= lat_range[1]) or not (lon_range[0] <= lon <= lon_range[1]):
        error = "Field `Latitude` and 'Longitude' must be within the range of the UK"
        return False, error
    
    return True, ""



# End Checks
########################################

########################################
# Begin webserver stuff

app = Flask(__name__)


@app.route('/should_search/', methods=['POST'])
def predict():
    # Flask provides a deserialization convenience function called
    # get_json that will work if the mimetype is application/json.
    obs_dict: dict = request.get_json() 
    request_ok, error = check_request(obs_dict)
    if not request_ok:
        response = {'error': error}
        return jsonify(response), 405
    date_ok, error = check_date_format(obs_dict)
    if not date_ok:
        response = {'error': error}
        return jsonify(response), 405
    
    _id = obs_dict['observation_id']
    observation = obs_dict

    observation = fill_missing_categorical_columns(observation)
    
    # Now do what we already learned in the notebooks about how to transform
    # a single observation into a dataframe that will work with a pipeline.
    obs = pd.DataFrame([observation], columns=observation.keys())[columns].astype(dtypes)
         

    # Now get ourselves an actual prediction of the positive class.
    prediction = pipeline.predict_proba(obs)[:, 1][0]
    prediction_value = True if prediction>=0.35 else False


    response = {'outcome': prediction_value}

    p = Prediction(
        observation_id=_id,
        predicted_outcome=prediction_value,
        observation= str(observation),
    )
     
    try:
        p.save()
    except IntegrityError:
        error_msg = 'Observation ID: "{}" already exists'.format(_id)
        response['error'] = error_msg
        DB.rollback() 
        return jsonify(response), 405

    return jsonify(response) 

 
@app.route('/search_result/', methods=['POST'])
def update():
    obs = request.get_json()
    try:
        p = Prediction.get(Prediction.observation_id == obs['observation_id'])
        p.outcome = obs['outcome']
        p.save()

        model_dict = model_to_dict(p)

        model_response_dict = {}
        model_response_dict["observation_id"] = model_dict["observation_id"]
        model_response_dict["outcome"] = model_dict["outcome"]
        model_response_dict["predicted_outcome"] = model_dict["predicted_outcome"]
        return jsonify(model_response_dict)
    except Prediction.DoesNotExist:
        error_msg = 'Observation ID: "{}" does not exist'.format(obs['observation_id'])
        return jsonify({'error': error_msg}), 405



@app.route('/list-db-contents') 
def list_db_contents():
    return jsonify([
        model_to_dict(obs) for obs in Prediction.select()
    ])


# End webserver stuff
########################################

if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=True, port=5000)  

 
