import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler

class Framingham:
    def __init__(self, data_path = '/Volumes/Development/data/deep_learning/farmingham/data.csv', 
               test_size = 0.3, oversample=False, verbose=False):
        df = pd.read_csv(data_path)
        ss = StandardScaler()
        print("scaling with standard scaler")
        scale_features = ["age","cigsPerDay","totChol","sysBP", "diaBP", "BMI", "heartRate", "glucose"]
        df[scale_features] = ss.fit_transform(df[scale_features])
        y = df['TenYearCHD']
        X = df.drop('TenYearCHD', axis=1)
        if verbose:
            print("Loading", len(df), "rows")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=1) # 70% training and 30% test
        if verbose:
            print("split: Train: ", len(X_train), len(y_train), "Test: ", len(X_test), len(y_test))
        if oversample:
            _oversample =  RandomOverSampler(sampling_strategy='minority')
        X_train, y_train = _oversample.fit_resample(X_train, y_train)
        if verbose:
            print("after resample\nsplit: Train: ", len(X_train), len(y_train), "Test: ", len(X_test), len(y_test))
        else:
            print("no resampling")
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.X_val = 0
        self.y_val = 0
    def generate_validation(self, validation_percent_of_training=0.2):
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(self.X_train, self.y_train, test_size=validation_percent_of_training, random_state=1) 
        print("after validation\n Train: ", len(self.X_train), len(self.y_train), "Val: ", len(self.X_val), len(self.y_val), "Test:", len(self.X_test), len(self.y_test))