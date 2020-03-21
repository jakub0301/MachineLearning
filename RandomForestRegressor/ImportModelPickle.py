import pickle
import pandas as pd


with open('model.pkl', 'rb') as f:
    model = pickle.load(f)
# Path testing data
test_data_path = 'home-data-for-ml-course/test.csv'

features = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']

#Training data
test_data = pd.read_csv(test_data_path)

#Testing
test_X = test_data[features]
test_preds = model.predict(test_X)


#Generate output 
output = pd.DataFrame({'Id': test_data.Id,
                       'SalePrice': test_preds})
output.to_csv('submissionPickle.csv', index=False)

