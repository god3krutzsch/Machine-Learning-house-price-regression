import pandas as pd
import numpy as np

# split training set
from sklearn.model_selection import train_test_split

# provide numeric variables for missing numbers from dataset
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt  # Matlab-style plotting

# special encoder that retains ordinal order requirements
from sklearn.preprocessing import OrdinalEncoder

# data visualization library based on matplotlib
import seaborn as sns

color = sns.color_palette()
sns.set_style('darkgrid')

# probability curve
from scipy import stats
import asyncio

# ai model import
from sklearn.linear_model import LinearRegression

# r-square performance measure regression model
from sklearn.metrics import r2_score

file_path = "/Users/godfreykrutzsch/Desktop/ML-Challenge_House Price/train_column.csv"
file_path_test = "/Users/godfreykrutzsch/Desktop/ML-Challenge_House Price/test.csv"

training_data = pd.read_csv(file_path)
test_data = pd.read_csv(file_path_test)

# there are problems with this particular column
training_data.drop(columns='FireplaceQu', inplace=True)
test_data.drop(columns='FireplaceQu', inplace=True)

# 1.0 date exploration note the below does not show enumerated types

print(training_data.isnull().sum())
print(training_data.isna())
print(test_data.isnull().sum())
print(test_data.isna())

print("Shape of training data:", training_data.shape)
print("Shape of test date:", test_data.shape)


# 1.1 visualisation of data

def distribution_graph():
    # distribution plot
    sns.histplot(training_data['SalePrice']).set_title("Distribution of SalePrice")

    # probability plot
    fig = plt.figure()
    res = stats.probplot(training_data['SalePrice'], plot=plt)
    plt.show()


def grLiveArea_Saleprice():
    fig, ax = plt.subplots()
    ax.scatter(x=training_data['GrLivArea'], y=training_data['SalePrice'])
    plt.ylabel('SalePrice', fontsize=13)
    plt.xlabel('GrLivArea', fontsize=13)
    plt.show()

    # def correlation_amongst_variables():
    corr = training_data.corr()
    plt.subplots(figsize=(13, 10))
    sns.heatmap(corr, vmax=0.9, cmap="Blues", square=True)


def totalBsmtSF_SalePrice():
    fig, ax = plt.subplots()
    ax.scatter(x=training_data['GrLivArea'], y=training_data['SalePrice'])
    plt.ylabel('SalePrice', fontsize=13)
    plt.xlabel('TotalBsmtSF', fontsize=13)
    plt.show()

    # def correlation_amongst_variables():
    corr = training_data.corr()
    plt.subplots(figsize=(13, 10))
    sns.heatmap(corr, vmax=0.9, cmap="Blues", square=True)


def garageCars_SalePrice():
    fig, ax = plt.subplots()
    ax.scatter(x=training_data['GrLivArea'], y=training_data['SalePrice'])
    plt.ylabel('SalePrice', fontsize=13)
    plt.xlabel('GarageCars', fontsize=13)
    plt.show()

    # def correlation_amongst_variables():
    corr = training_data.corr()
    plt.subplots(figsize=(13, 10))
    sns.heatmap(corr, vmax=0.9, cmap="Blues", square=True)


def overallLivingQual():
    fig, ax = plt.subplots()
    ax.scatter(x=training_data['OverallQual'], y=training_data['SalePrice'])
    plt.ylabel('SalePrice', fontsize=13)
    plt.xlabel('OverallQual', fontsize=13)
    plt.show()


def log_transformation():
    training_data["SalePrice"] = np.log1p(training_data["SalePrice"])
    sns.histplot(training_data['SalePrice']).set_title("Distribution of Sales Price after Log trans")
    # probability plot
    fig = plt.figure()
    res = stats.probplot(training_data['SalePrice'], plot=plt)
    # plt.show()


# we eliminate outliers from the get go and start log transformation. this changes the shape from 1460 to 1458 as we
# kill to rows with 4676 and 5642 outliers, therefore we cannot meet the submission criteria 1459 rows.
# if we run this we get an error as there is a mismatch with num row in y (untouched) and num rows in x (deleted)

# training_data = training_data.drop(training_data[(training_data['GrLivArea'] > 4000) & (training_data['SalePrice']
# < 300000)].index)

print("Shape of training data after outliers after ignored:", training_data.shape)
print("Shape of test date after outliers after ignored:", test_data.shape)

log_transformation()

y = training_data['SalePrice']

print("The shape of the y target value after outliers ignored", y.shape)

file_path_3 = "/Users/godfreykrutzsch/Desktop/ML-Challenge_House Price/sales_check.csv"
training_data['SalePrice'].to_csv(file_path_3, index=False)

# 1.3 how much missing data is there?

# we merge training and test data so we do not have to do the same operations twice on the data sets
df_combined = pd.concat([training_data.reset_index(drop=True), test_data.reset_index(drop=True)], axis=0)
print("Shape of combined DataFrame:", df_combined.shape)

print("The shape of data set after training and test are merged", df_combined)

# find % of missing values with a df_nan, this works well with columns with shed loads of enumerated types
df_nan = df_combined.isna().sum().reset_index(name="missing_values")
df_nan["percentage"] = round((df_nan["missing_values"] / df_combined.shape[0]) * 100, 2)
print(df_nan.sort_values(by="percentage", ascending=False)[:35])


# 2. Exploratory data analysis

# dealing with a lot of non-numeric missing data
# what is size of task e.g. text columns v integer columns
# what columns are integer and which ones are string


# classify features as integers or text for preprocessing and count the number of each

def get_datatypes():
    # Initialize counters
    int_columns = 0
    str_columns = 0
    sum_list_integer_columns_names = []
    sum_list_str_column_names = []

    for col in df_combined.columns:
        if df_combined[col].dtype == 'int64' or df_combined[col].dtype == 'float64':
            int_columns += 1
            sum_list_integer_columns_names.append(col)
        elif df_combined[col].dtype == 'object':
            str_columns += 1
            sum_list_str_column_names.append(col)

    print("Number of columns with integers:", int_columns)
    print("Number of columns with strings:", str_columns)

    percent_int = (int_columns / (int_columns + str_columns) * 100)
    percent_string = (str_columns / (int_columns + str_columns) * 100)

    print("% of integers are", percent_int)
    print("% of strings are", percent_string)

    print("list of fields that are integers", sum_list_integer_columns_names)
    print("List of fields that are strings", sum_list_str_column_names)


get_datatypes()

# 3 data preprocessing
# Categorical the hunt for NaN and replacement we use df_categorical for string features

df_categorical = df_combined
# df_categorical['FirePlaceQu'] = df_combined['FireplaceQu'].drop(inplace=True)

print(df_categorical.columns)

df_categorical['PoolQC'] = df_categorical['PoolQC'].fillna("NoPool")
df_categorical['MiscFeature'] = df_categorical['MiscFeature'].fillna("None")
df_categorical['Alley'] = df_categorical['Alley'].fillna("NoAlleyAccess")
df_categorical['MasVnrType'] = df_categorical['MasVnrType'].fillna("NoMasonry")
# df_categorical['FirePlaceQu'] = df_categorical['FirePlaceQu'].fillna("NoFirePlace")
df_categorical['GarageFinish'] = df_categorical['GarageFinish'].fillna("NoGarage")
df_categorical['GarageQual'] = df_categorical['GarageQual'].fillna("NoGarage")
df_categorical['GarageCond'] = df_categorical['GarageCond'].fillna("NoGarage")
df_categorical['GarageType'] = df_categorical['GarageType'].fillna("NoGarage")
df_categorical['BsmtExposure'] = df_categorical['BsmtExposure'].fillna("NoBasement")
df_categorical['BsmtCond'] = df_categorical['BsmtCond'].fillna("NoBasement")
df_categorical['BsmtQual'] = df_categorical['BsmtQual'].fillna("NoBasement")
df_categorical['BsmtFinType2'] = df_categorical['BsmtFinType2'].fillna("NoBasement")
df_categorical['BsmtFinType1'] = df_categorical['BsmtFinType1'].fillna("NoBasement")

df_categorical['Fence'] = df_categorical['Fence'].fillna(df_categorical['Fence']).mode()[0]
df_categorical['MSZoning'] = df_categorical['MSZoning'].fillna(df_categorical['MSZoning']).mode()[0]
df_categorical['Functional'] = df_categorical['Functional'].mode()[0]
df_categorical['Utilities'] = df_categorical['Utilities'].mode()[0]
df_categorical['Electrical'] = df_categorical['Electrical'].mode()[0]
df_categorical['KitchenQual'] = df_categorical['KitchenQual'].mode()[0]
df_categorical['SaleType'] = df_categorical['SaleType'].mode()[0]
df_categorical['Exterior2nd'] = df_categorical['Exterior2nd'].mode()[0]
df_categorical['Exterior1st'] = df_categorical['Exterior1st'].mode()[0]

print("The shape of categorical after processing", df_categorical)

# 4.1 Preprocessing integer features
# we can tackle the missing numeric values with some numeric dataframes


df_numeric = df_categorical.select_dtypes(include=['int64', 'float64'])
df_numeric_independent = df_numeric.drop(columns=["SalePrice"], axis=1)
df_numeric_dependent = training_data['SalePrice']
df_misc = df_categorical.select_dtypes(exclude=['int64', 'float64'])

print("The numerics", df_numeric)
print("The independents", df_numeric_independent)
print("The lone dependent", df_numeric_dependent)
print("the insurance", df_misc)

# 5.0  preprocessing lets get rid off the NaNs and missing values in the numerical features
imputer = IterativeImputer(estimator=RandomForestRegressor(random_state=0), max_iter=10, random_state=0)
df_numeric_independent_imputed_v1 = imputer.fit_transform(df_numeric_independent)

print("Did it work....?", df_numeric_independent_imputed_v1)

print("convert back to df from array output of regression")

df_numeric_independent_imputed_v2 = pd.DataFrame(df_numeric_independent_imputed_v1,
                                                 columns=df_numeric_independent.columns)
print(df_numeric_independent_imputed_v2)

df_imputed = pd.concat([df_misc.reset_index(drop=True), df_numeric_independent_imputed_v2.reset_index(drop=True),
                        df_numeric_dependent.reset_index(drop=True)], axis=1)

print("We check the shape of the entire data frame after this merge", df_imputed)

df_check = df_imputed.isna().sum().reset_index(name="missing_values")
df_check['Percentage'] = round((df_check['missing_values'] / df_imputed.shape[0]) * 100, 2)
print(df_check.sort_values(by="Percentage", ascending=False)[:35])

print("Shape of combined of data frame:", df_imputed.shape)

# 4.0 data manipulation one hot encoding for CATEGORICAL, i ignore the multicollinearty features (Garagecars,
# 1stFlrSF, GarageYrBuilt, Totrmsabvgrd (4).

df_misc = pd.get_dummies(df_misc, columns=['MSZoning', 'Street', 'Alley', 'LandContour', 'Utilities',
                                           'LotConfig', 'Neighborhood', 'Condition1', 'Condition2',
                                           'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl',
                                           'Exterior1st', 'Exterior2nd', 'MasVnrType', 'Foundation',
                                           'Heating', 'Electrical', 'Functional', 'GarageType',
                                           'Fence', 'MiscFeature', 'SaleType', 'SaleCondition',
                                           'CentralAir']) * 1

# 5.0 ordinal feature transformation

quality_map = {"Ex": 5, "Gd": 4, "Ta": 3, "Fa": 2, "Po": 1}

basement_fin_map = {"GLQ": 6, "ALQ": 5, "BLQ": 4, "Rec": 3, "LwQ": 2, "Unf": 1, "NoBasement": 0}

basement_exposure = {"Gd": 4, "Av": 3, "Mn": 2, "No": 1, "NoBasement": 0}

garage_quality = {"Ex": 5, "Gd": 4, "TA": 3, "Fa": 2, "Po": 1, "No Garage": 0}

lot_shape = {"Reg": 3, "IR1": 2, "IR2": 1, "IR3": 0}

land_slope = {"Gtl": 2, "Mod": 1, "Sev": 0}

paved_drive = {"Y": 2, "P": 1, "N": 0}

garage_finish = {"Fin": 3, "RFn": 2, "Unf": 1, "NoGarage": 0}

pd.set_option('future.no_silent_downcasting', True)
df_misc.replace({
    'ExterCond': quality_map,
    'ExterQual': quality_map,
    'BsmtQual': quality_map,
    'BsmtCond': quality_map,
    'HeatingQC': quality_map,
    'KitchenQual': quality_map,
    'FireplaceQu': quality_map,
    'GarageCond': quality_map,
    'PoolQC': quality_map,
    'BsmtFinType1': basement_fin_map,
    'BsmtFinType2': basement_fin_map,
    'BsmtExposure': basement_exposure,
    'GarageQual': garage_quality,
    'LotShape': lot_shape,
    'LandSlope': land_slope,
    'PavedDrive': paved_drive,
    'GarageFinish': garage_quality
}, inplace=True)

df_misc = pd.get_dummies(df_misc, columns=["ExterCond", "ExterQual", "BsmtQual", "BsmtCond", "HeatingQC",
                                           "KitchenQual", "GarageCond", "GarageCond",
                                           "BsmtExposure",
                                           "GarageCond", "PoolQC", "BsmtFinType1", "BsmtFinType2",
                                           "BsmtExposure", "GarageQual", "GarageCond", "LotShape",
                                           "LandSlope", "GarageFinish",
                                           "PavedDrive"]) * 1

file_path_impute = "/Users/godfreykrutzsch/Desktop/ML-Challenge_House Price/dependent.csv"
df_numeric_dependent.to_csv(file_path_impute, index=False)

# Split the combined DataFrame back into training and test datasets
# training_data_new = hot_encode_merge.iloc[:num_training_samples]
# test_data_new = hot_encode_merge .iloc[num_training_samples:]


print("Before assigning new values for merge", df_numeric_independent_imputed_v2)
print("check misc value", df_misc)
print("Check y value", y)

df_new_numeric_independent_training_data = df_numeric_independent_imputed_v2.iloc[:1460]
df_new_string_training_data = df_misc.iloc[:1460]
y_value = y.iloc[:1460]

df_new_numeric_independent_training_data.reset_index(drop=True, inplace=True)
df_new_string_training_data.reset_index(drop=True, inplace=True)
y_value.reset_index(drop=True, inplace=True)

print("Numeric Independent Data Shape:", df_new_numeric_independent_training_data.shape)
print("String Training Data Shape:", df_new_string_training_data.shape)
print("Dependent Data Shape:", y_value.shape)

# df_all_features = pd.concat([df_numeric_independent_imputed_v2, df_misc, df_numeric_dependent], axis=1)
df_all_features = pd.concat([df_new_numeric_independent_training_data, df_new_string_training_data], axis=1)

print("Check after super merge of independent and formally text features", df_all_features)

assert df_all_features.shape[0] == y_value.shape[0], "Mismatch in number of rows between features and target"

print("all features Data Shape:", df_all_features.shape)

file_path_impute = "/Users/godfreykrutzsch/Desktop/ML-Challenge_House Price/super_merge.csv"
df_all_features.to_csv(file_path_impute, index=False)

X = df_all_features

print("we check the two parameters shape before splitt")

print("The X value shape", X.shape)
print("The Y value shape", y_value.shape)

# assert len(X) = 1458, "Dataset does not have enough samples for the specified test size."

X_train, X_test, y_train, y_test = train_test_split(X, y_value, test_size=0.1, random_state=42)

# Separate back into training and test sets for submission where rows must be 1459
# X_train = df_all_features.iloc[:1460]
# X_test = df_all_features.iloc[:1460]
# y_train = training_data['SalePrice']
# y_test = df_all_features.iloc[:1460]


print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)

print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)

print("check how many predictions")

lm = LinearRegression()
lm.fit(X_train, y_train)
y_pred = lm.predict(X_test)

print("These must equal each other for a R score")
print("Predictions on test set:", y_pred.shape)
print("the y test", y_test)


# corrmat = pd.concat(
#    [df_numeric_independent_imputed_v2.reset_index(drop=True), df_numeric_dependent.reset_index(drop=True)],
#    axis=1).corr()


# we identify features that are multicollinearity

def show_heatmap_one():
    plt.subplots(figsize=(13, 10))
    sns.heatmap(corrmat, vmax=0.9, cmap="Blues", square=True)
    # Create the heatmap using Matplotlib's imshow function
    plt.imshow(corrmat, cmap='hot', interpolation='nearest')
    plt.colorbar()
    plt.show()


# show_heatmap_one()


# sale price correlation matrix
def showHeatmap_two():
    k = 13  # number of variables for heatmap
    cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index
    cm = np.corrcoef(corrmat[cols].values.T)
    sns.set(font_scale=1.25)
    hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values,
                     xticklabels=cols.values)
    plt.show()


# showHeatmap_two()
# grLiveArea_Saleprice()
# totalBsmtSF_SalePrice()
# garageCars_SalePrice()
# overallLivingQual()


# 5.0 feature engineering

#  a shed load of diagrams some useful others not.

def showAllDiagrms():
    sns.set()
    cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'GarageArea', 'TotalBsmtSF', 'FullBath', 'YearBuilt',
            'TotRmsAbvGrd']
    sns.pairplot(corrmat[cols], height=2.5)
    plt.show()


print("quick check b 4 sub")
print(y_pred)

submission = pd.DataFrame()
X_test['Id'] = X_test['Id'].astype(int)
submission['Id'] = X_test['Id']
submission['SalePrice'] = y_pred
submission.to_csv("/Users/godfreykrutzsch/Desktop/ML-Challenge_House Price/submission_final.csv", index=False)

r2 = r2_score(y_test, y_pred)
print("We print the r-score", r2)
