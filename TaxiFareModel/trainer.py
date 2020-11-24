# imports
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.base import BaseEstimator,  TransformerMixin
from sklearn.model_selection import train_test_split
from TaxiFareModel.encoders import TimeFeaturesEncoder, DistanceTransformer
from TaxiFareModel.data import get_data,clean_data
from TaxiFareModel.utils import compute_rmse

class Trainer():
    def __init__(self, X, y):
        """
            X: pandas DataFrame
            y: pandas Series
        """
        self.pipeline = None
        self.X = X
        self.y = y

    def set_pipeline(self):
        """defines the pipeline as a class attribute"""
        pipe_distance = make_pipeline(DistanceTransformer(), RobustScaler())
        pipe_time = make_pipeline(TimeFeaturesEncoder(time_column='pickup_datetime'), OneHotEncoder(handle_unknown='ignore'))
        dist_cols = ['pickup_latitude', 'pickup_longitude', 'dropoff_latitude', 'dropoff_longitude']
        time_cols = ['pickup_datetime']
        feat_eng_bloc = ColumnTransformer([('time', pipe_time, time_cols),
                                      ('distance', pipe_distance, dist_cols)]
                                      )
        self.pipeline = Pipeline(steps=[('feat_eng_bloc', feat_eng_bloc),
                                ('regressor', RandomForestRegressor())])
        return self.pipeline


    def run(self):
        """set and train the pipeline"""
        self.pipeline = self.set_pipeline()
        self.pipeline.fit(self.X,self.y)
        return self

    def evaluate(self, X_test, y_test):
        """evaluates the pipeline on df_test and return the RMSE"""
        y_pred = self.pipeline.predict(X_test)
        return compute_rmse(y_pred, y_test)

if __name__ == "__main__":
    df = get_data()
    df = clean_data(df)
    X = df.drop(columns=['fare_amount'])
    y = df.fare_amount
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
    trainer = Trainer(X=X_train,y=y_train)
    trainer.run()
    print("RMSE:",trainer.evaluate(X_test = X_test, y_test = y_test))
