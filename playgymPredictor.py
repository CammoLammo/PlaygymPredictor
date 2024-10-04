import pandas as pd
from meteostat import Point, Hourly
from datetime import datetime
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt
import numpy as np




#Read Square Sales Reports
df2023 = pd.read_csv("2023-01-01-2023-12-311.csv")
df2024 = pd.read_csv("2024-01-01-2024-10-05.csv")
willetton = Point(-32.048, 115.874)
holiday_periods = [
    ("2023-04-07", "2023-04-23"),  # Term 1 break
    ("2023-07-01", "2023-07-16"),  # Term 2 break
    ("2023-09-23", "2023-10-08"),  # Term 3 break
    ("2023-12-14", "2024-01-31"),   # Term 4 break (example for year-end holidays)
    ("2024-03-29", "2024-04-14"),
    ("2024-06-29", "2024-07-14"),
    ("2024-09-21", "2024-10-06"),
    ("2024-12-13", "2024-12-27")
]

holiday_periods = [(datetime.strptime(start, "%Y-%m-%d"), datetime.strptime(end, "%Y-%m-%d")) for start, end in holiday_periods]



#Set up Historical Data DF
columns = ["Date", "Attendance", "Temperature", "Rainfall", "Holiday", "Day"]
dfHistoricalData = pd.DataFrame(columns=columns)

# print(df2023.head())
print(df2024.head())
print(dfHistoricalData)

def getWeatherData(dateDt):
    
    start = datetime(dateDt.year, dateDt.month, dateDt.day, 9)
    end = datetime(dateDt.year, dateDt.month, dateDt.day, 12)
    weatherData = Hourly(willetton, start=start, end=end)
    weatherData = weatherData.fetch()

    temperature = weatherData['temp'].values[0] if not weatherData.empty else None
    rainfall = weatherData['prcp'].values[0] if not weatherData.empty else None

    return temperature, rainfall



#Get total revenue for each day and add to Historical Data DF
def addTotalKids(dfHistoricalData, rawDf):
    for col in rawDf.columns[3:]:
        newDay = {
            "Date": col,
            "Attendance": rawDf[col].sum()
        }

        dfHistoricalData = dfHistoricalData.append(newDay, ignore_index=True)
    return dfHistoricalData

dfHistoricalData = addTotalKids(dfHistoricalData, df2023)
dfHistoricalData = addTotalKids(dfHistoricalData, df2024)
print(dfHistoricalData.describe())
print(dfHistoricalData.sample(10))

def filterEmptyDays(dfHistoricalData):
    dfHistoricalData = dfHistoricalData[dfHistoricalData["Attendance"] != 0]
    dfHistoricalData = dfHistoricalData.reset_index(drop=True)
    return dfHistoricalData

dfHistoricalData = filterEmptyDays(dfHistoricalData)
print(dfHistoricalData)

for index, row in dfHistoricalData.iterrows():
    date = row["Date"]
    dateDt = datetime.strptime(date, "%d/%m/%Y")
    temperature, rainfall = getWeatherData(dateDt)

    dfHistoricalData.at[index, "Holiday"] = 2

    for start, end in holiday_periods:
        if start <= dateDt <= end:
            dfHistoricalData.at[index, "Holiday"] = 1

    dfHistoricalData.at[index, "Day"] = dateDt.weekday()
    dfHistoricalData.at[index, "Temperature"] = temperature
    dfHistoricalData.at[index, "Rainfall"] = rainfall

print("SAMPLES")
print(dfHistoricalData.sample(10))
print(dfHistoricalData.describe())

dfModelTrainer = dfHistoricalData.drop(columns=["Date"])
dfModelTrainer = dfModelTrainer.dropna()

Xset = dfModelTrainer[["Temperature", "Rainfall", "Holiday", "Day"]]
y = dfModelTrainer["Attendance"]

X_train, X_test, y_train, y_test = train_test_split(Xset, y, test_size=0.2, random_state=111)
model = MLPRegressor(hidden_layer_sizes=(100,100), max_iter=1000, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)  # Mean Squared Error
r2 = r2_score(y_test, y_pred)             # R^2 score (coefficient of determination)

model2 = GradientBoostingRegressor(n_estimators=250, learning_rate=0.075, random_state=42, min_samples_split=10)
model2.fit(X_train, y_train)
y_pred2 = model2.predict(X_test)
mse2 = mean_squared_error(y_test, y_pred2)  # Mean Squared Error
r22 = r2_score(y_test, y_pred2)             # R^2 score (coefficient of determination)
#0.17445401913871506
print(f'Mean Squared Error: {mse}')
print(f'R^2 Score: {r2}')
print(f'Mean Squared Error 2: {mse2}')
print(f'R^2 Score 2: {r22}')

param_grid = {
    'n_estimators': [5, 25, 50, 75],
    'learning_rate': [0.05, 0.075, 0.025],
    'min_samples_split': [2, 1.0, 3]
}

grid_search = GridSearchCV(GradientBoostingRegressor(random_state=42), param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(X_train, y_train)

print(grid_search.best_params_)
print(grid_search.best_score_)




plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred2, color='blue', label='Predicted vs Actual')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', label='Ideal fit line')
plt.xlabel('Actual Attendance')
plt.ylabel('Predicted Attendance')
plt.title('Actual vs Predicted Attendance')
plt.legend()
plt.show()

plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, color='blue', label='Predicted vs Actual')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', label='Ideal fit line')
plt.xlabel('Actual Attendance')
plt.ylabel('Predicted Attendance')
plt.title('Actual vs Predicted Attendance')
plt.legend()
plt.show()



# # Get Feature Importances from Random Forest
# importances = model.feature_importances_
# features = ['Temperature', 'Rainfall', 'Holiday', 'Day']

# # Plot Feature Importances
# plt.figure(figsize=(8, 6))
# plt.barh(features, importances, color='green')
# plt.xlabel('Feature Importance')
# plt.title('Feature Importance from Random Forest Model')
# plt.show()

perm_importance = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=42)

# Feature names
feature_names = ['Temperature', 'Rainfall', 'Holiday', 'Day']

# Ensure that the length of feature_names matches the number of features in X_test
if len(feature_names) != len(perm_importance.importances_mean):
    raise ValueError("Length of feature_names does not match the number of features in the model.")

# Sort indices by importance
sorted_idx = perm_importance.importances_mean.argsort()

# Plot horizontal bar chart
plt.figure(figsize=(8, 6))
plt.barh([feature_names[i] for i in sorted_idx], perm_importance.importances_mean[sorted_idx], color='skyblue')
plt.xlabel("Permutation Importance")
plt.title("Feature Importance for MLPRegressor")
plt.show()




# Get Feature Importances from Random Forest
importances = model2.feature_importances_
features = ['Temperature', 'Rainfall', 'Holiday', 'Day']

# Plot Feature Importances
plt.figure(figsize=(8, 6))
plt.barh(features, importances, color='green')
plt.xlabel('Feature Importance')
plt.title('Feature Importance from Random Forest Model')
plt.show()

# Calculate Residuals
residuals = y_test - y_pred

# Plot Residuals
plt.figure(figsize=(8, 6))
plt.scatter(y_pred, residuals, color='purple')
plt.axhline(y=0, color='red', linestyle='--')
plt.xlabel('Predicted Attendance')
plt.ylabel('Residuals')
plt.title('Residuals vs Predicted Attendance')
plt.show()

# Calculate Residuals
residuals = y_test - y_pred2

# Plot Residuals
plt.figure(figsize=(8, 6))
plt.scatter(y_pred2, residuals, color='purple')
plt.axhline(y=0, color='red', linestyle='--')
plt.xlabel('Predicted Attendance')
plt.ylabel('Residuals')
plt.title('Residuals vs Predicted Attendance')
plt.show()