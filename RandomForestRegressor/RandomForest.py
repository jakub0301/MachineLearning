""" Random Forest Kaggle Competition """
import pickle
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
#from sklearn.metrics import mean_absolute_error
#from sklearn.model_selection import train_test_split
#from sklearn.tree import DecisionTreeRegressor


#Path training data
iowa_file_path = 'home-data-for-ml-course/train.csv'

#Training data
home_data = pd.read_csv(iowa_file_path)

#Real prices
y = home_data.SalePrice

#Features
features = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']
X = home_data[features]

#RandomForest
rf_model_on_full_data = RandomForestRegressor(random_state=1)
#Training
rf_model_on_full_data.fit(X, y)

#Path testing data
test_data_path = 'home-data-for-ml-course/test.csv'

#Training data
test_data = pd.read_csv(test_data_path)

#Testing
test_X = test_data[features]
test_preds = rf_model_on_full_data.predict(test_X)


#Generate output 
output = pd.DataFrame({'Id': test_data.Id,
                       'SalePrice': test_preds})
output.to_csv('submission.csv', index=False)

#Save model to pickle 
with open("model.pkl", "wb+") as file:
    pickle.dump(rf_model_on_full_data, file)


