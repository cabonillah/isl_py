import os
from zipfile import ZipFile
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf
import statsmodels.api as sm
from sklearn.preprocessing import PolynomialFeatures, FunctionTransformer, PowerTransformer, OneHotEncoder


os.chdir("D:/Archivos/Datos_practica/Python")
with ZipFile('../ISLR/ALL+CSV+FILES+-+2nd+Edition+-+corrected.zip') as zf:
    auto = pd.read_csv(zf.open('ALL CSV FILES - 2nd Edition/Auto.csv'), na_values=['?'])
    
#Had to omit na's
auto.dropna(inplace=True)
#Restaurar los índices del DataFrame después de eliminar los missing values
auto.reset_index(inplace=True)
auto.drop('index', axis=1, inplace=True)

#Use statsmodels for regression
fit = smf.ols(formula='mpg~horsepower', data=auto).fit()
fit.summary()

#Predict mpg for 98 HP
fit.predict(pd.DataFrame({'horsepower':[98]}))
#Confidence interval
fit.get_prediction(pd.DataFrame({'horsepower':[98]})).summary_frame()[["mean_ci_lower", "mean_ci_upper"]]
#Prediction interval
fit.get_prediction(pd.DataFrame({'horsepower':[98]})).summary_frame()[["obs_ci_lower", "obs_ci_upper"]]

#Dataframe of predicted response and predictor
predicted_mpg = pd.DataFrame({"pred_mpg": fit.predict(auto), "horsepower": auto["horsepower"]})

#Plot response vs. predictor and then add the regression line
# fig, ax = plt.subplots()
# ax.scatter(auto["horsepower"], auto["mpg"], color="blue")
# ax.plot(predicted_mpg["horsepower"], predicted_mpg["pred_mpg"], color="red")
# plt.xlabel('horsepower')
# plt.ylabel('mpg')
# plt.close()

#Diagnostic plots are not included because there are no popular packages for Python

#Scatterplot matrix for the dataset (excluding "name" column)
#sns.pairplot(auto.loc[:, auto.columns != "name"])
#plt.close()

#Correlation matrix (excluding "name" column)
auto.loc[:, auto.columns != "name"].corr()

#Regression on all variables except for name
formula = "mpg ~ " + " + ".join(i for i in auto.columns if i not in ["mpg", "name"])
fit2 = smf.ols(formula=formula, data=auto).fit()
fit2.summary()

#Diagnostic plots are not included because there are no popular packages for Python

# TODO: Regression on all variables except for name, now including interactions 
# X = todas las variables menos mpg y name

# Función para emplear en FunctionTransformer. Obtiene transformaciones para 
# las columnas especificadas en "x". Las transformaciones consisten en elevar
# cada variable hasta el grado indicado en "pwr"
def my_power(x, pwr):
    result = pd.DataFrame(np.hstack([x**(i+1) for i in range(1, pwr+1)]))
    
    def features(feats):
        feature_names = [str(i) for i in feats]
        feature_names_generated = []
        for i in range(2, pwr+1):
            for j in feature_names:
                feature_names_generated.append(j+"^"+str(i))
        features_complete = feature_names + feature_names_generated
        return features_complete
    
    result.columns = features(x.columns)    
    return result
# TODO: Función generadora de nombres para el argumento "feature_names_out"
# cat = OneHotEncoder(drop='first')
# deg = FunctionTransformer(my_power,
#                           kw_args={'pwr':2}, #Argumento de my_power
#                           feature_names_out=["x"+str(i) for i in range(10)] #Nombres de las columnas obtenidas a partir del transformador
#                           )

X = auto.drop(['mpg', 'name', 'origin'], axis=1)
Y = auto[['mpg']]


    
# transformer = FunctionTransformer(my_power, kw_args={"pwr":4})
# X = np.array([[0, 1, 5], 
#               [2, 3, 1], 
#               [3, 4, 1]])
# np.moveaxis(transformer.transform(X), 1, 0).reshape(X.shape[0], X.shape[0]*3)


       
# X = my_power(X, 2)
# def mypower2(x):
#     result = np.power(x, np.arange(1, 3))
#     return result

# for i in range(3):
#     print(type(X.iloc[:, i]))
    

# transformer2 = FunctionTransformer(np.sqrt)
# transformer3 = FunctionTransformer(lambda x: np.power(x, 3))
# transformer3 = PowerTransformer(exp=2)
# PowerTransformer()
# X2 = X['cylinders']

# TODO: convetir el output de PolynomialFeatures a DataFrame
poly = PolynomialFeatures(interaction_only=True)
cols_inter = poly.get_feature_names_out().tolist()
X_inter = poly.fit_transform(X)

transformer = FunctionTransformer(my_power, kw_args={'pwr':2})
X_power = transformer.fit_transform(X)

# TODO: convetir el output de OneHotEncoder a DataFrame
enc = OneHotEncoder(drop='first', handle_unknown='ignore')
cols_dummies = enc.get_feature_names_out().tolist()
X_dummies = enc.fit_transform(auto[['origin']])
X_dummies.toarray()


fit3 = sm.OLS(Y, X).fit()
fit3.summary()

