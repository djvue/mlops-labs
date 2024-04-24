#!/bin/sh

python ./data_creation.py && echo "data_creation.py: data created successfully"
python ./model_preprocessing.py && echo "model_preprocessing.py: data preprocessed successfully"
python ./model_preparation.py && echo "model_preparation.py: model trained successfully"
python ./model_testing.py && echo "model_testing.py: model tested successfully"
