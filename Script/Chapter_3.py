import builtins
import os
import sys
from zipfile import ZipFile
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf
import statsmodels.api as sm
from sklearn.pipeline import Pipeline
from sklearn.base import (BaseEstimator, 
                          RegressorMixin)
from sklearn.compose import ColumnTransformer                             
from sklearn.preprocessing import (PolynomialFeatures,
                                   OneHotEncoder)
#sklego es un módulo experimental. Su funcionalidad puede cambiar
from sklego.preprocessing import PatsyTransformer


#  Returns the path where your current main script is located at
def current_dir():
    if "get_ipython" in globals() or "get_ipython" in dir(builtins):        
        return os.getcwd()
    else:
        return os.path.abspath(os.path.dirname(sys.argv[0]))

script_wd = current_dir()
project_wd = os.path.dirname(script_wd)
os.chdir(project_wd)

with ZipFile('Data/ISLR/ALL+CSV+FILES+-+2nd+Edition+-+corrected.zip') as zf:
    auto = pd.read_csv(zf.open('ALL CSV FILES - 2nd Edition/Auto.csv'), na_values=['?'])
    
#Had to omit na's
auto.dropna(inplace=True)
#Restaurar los índices del DataFrame después de eliminar los missing values
auto.reset_index(inplace=True)
auto.drop('index', axis=1, inplace=True)
# Interpret origin column as categorical
auto['origin'] = auto.origin.astype('category')

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


#Regression with interaction effects

X = auto.drop(['mpg', 'name'], axis=1)
y = auto[['mpg']]


poly_transform = PolynomialFeatures(1, interaction_only=True, include_bias=False)

poly_transformer = ColumnTransformer(transformers=[('poly_trans', poly_transform, X.columns)], 
                                              remainder='passthrough')

# TODO: la transformación polinómica no se está ejecutando antes del ols.fit

class SMWrapper(BaseEstimator, RegressorMixin):
    """ A universal sklearn-style wrapper for statsmodels regressors """
    def __init__(self, model_class, fit_intercept=True):
        self.model_class = model_class
        self.fit_intercept = fit_intercept
    def fit(self, X, y):
        if self.fit_intercept:
            X = sm.add_constant(X)
        self.model_ = self.model_class(y, X)
        self.results_ = self.model_.fit()
        return self
    def predict(self, X):
        if self.fit_intercept:
            X = sm.add_constant(X)
        return self.results_.predict(X)

olsFit = SMWrapper(model_class=sm.OLS)

pipe_interac = Pipeline(steps=[('interactions', poly_transformer),
                               ('ols', olsFit)])

pipe_interac.fit(X, y)[1].results_.summary()

sm.OLS(y, X_inter).fit().summary(xname = total_vars_in_fit)


numeric_names = list(X.drop(['origin'], axis=1).columns)
numeric_names_2 = ["I(" + i + "**2)" for i in numeric_names]
numeric_names += numeric_names_2
formula = "+".join(numeric_names)

num_feats = [col for col in X.columns if col not in ['origin']]
cat_feats = ['origin']



pat_trans = PatsyTransformer(formula)
enc_trans = OneHotEncoder(drop='first', handle_unknown='ignore')

transformer_num = Pipeline(steps=[('pat_trans', pat_trans)])
transformer_cat = Pipeline(steps=[('onehotenc', enc_trans)])


transformer = ColumnTransformer(transformers=[('num_trans', transformer_num, num_feats),
                                              ('cat_trans', transformer_cat, cat_feats)], 
                                              remainder='passthrough')


pipe = Pipeline(steps=[('columntrans', transformer)])



num_vars_in_fit = pipe.fit(X).named_steps.columntrans.transformers[0][1].named_steps.pat_trans.fit(X[num_feats]).design_info_.column_names
cat_vars_in_fit = list(pipe.fit(X).named_steps.columntrans.fit(X).transformers[1][1].named_steps.onehotenc.fit(X[cat_feats]).get_feature_names_out())
total_vars_in_fit = num_vars_in_fit + cat_vars_in_fit

X_fit = pipe.fit_transform(X)

sm.OLS(y, X_fit).fit().summary(xname = total_vars_in_fit)


