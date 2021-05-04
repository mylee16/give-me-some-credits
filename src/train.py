import numpy as np
import pandas as pd
import yaml
import pickle
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier

from src.utils.pipeline import data_cleaning


class Classifier:
    def __init__(self, config):
        self.config = config
        self.train_data = pd.read_csv(self.config['train_data']).drop(['Unnamed: 0'], axis=1)
        self.test_data = pd.read_csv(self.config['test_data']).drop(['Unnamed: 0'], axis=1)
        self.X = None
        self.y = None
        self.model = None

    def data_cleaning(self):
        self.train_data = data_cleaning(self.train_data, if_test=False)
        self.X = self.train_data.drop(['SeriousDlqin2yrs'], axis=1)
        self.y = self.train_data['SeriousDlqin2yrs']
        print('data cleaned')

    def train(self):
        if self.config["model"] == "XGBoost":
            self.model = GradientBoostingClassifier(
                learning_rate=self.config['XGBoost']['learning_rate'],
                max_features=self.config['XGBoost']['max_features'],
                random_state=self.config['XGBoost']['random_state'],
                n_estimators=self.config['XGBoost']['n_estimators'],
                max_depth=self.config['XGBoost']['max_depth'],
                min_samples_split=self.config['XGBoost']['min_samples_split'],
                min_samples_leaf=self.config['XGBoost']['min_samples_leaf'],
                subsample=self.config['XGBoost']['subsample'])

        elif self.config["model"] == "RandomForestClassifier":
            self.model = RandomForestClassifier(
                n_estimators=self.config['RandomForestClassifier']['n_estimators'],
                criterion=self.config['RandomForestClassifier']['criterion'],
                max_depth=self.config['RandomForestClassifier']['max_depth'],
                min_samples_split=self.config['RandomForestClassifier']['min_samples_split'],
                min_samples_leaf=self.config['RandomForestClassifier']['min_samples_leaf'])
        else:
            raise ValueError("Please use XGBoost or RandomForestClassifier")

        self.model.fit(self.X, self.y)
        print('model fitted')

    def save_model(self):
        model_saved_path = self.config['model_saved_path']
        pickle.dump(self.model, open(model_saved_path, 'wb'))
        print('model saved to %s' % model_saved_path)

    def load_model(self):
        model_saved_path = self.config['model_saved_path']
        self.model = pickle.load(open(model_saved_path, 'rb'))
        print('model loaded from %s' % model_saved_path)

    def get_prediction(self):
        self.test_data = data_cleaning(self.test_data, if_test=True)
        test_pred = self.model.predict_proba(self.test_data)

        results = pd.DataFrame(np.array(test_pred)[:, 1:],
                               columns=['Probability'])
        results.index.name = 'id'
        results.index = results.index+1

        output_path = self.config['output_path']
        results.to_csv(output_path)
        print('prediction saved to %s' % output_path)


if __name__ == "__main__":
    with open("src/config/train.yaml", "r") as file:
        _CONFIG = yaml.safe_load(file)

    model = Classifier(config=_CONFIG)
    model.data_cleaning()
    model.train()
    model.save_model()
    model.load_model()
    model.get_prediction()
