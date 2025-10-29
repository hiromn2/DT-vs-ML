#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Hello! This is the final version of the code dealing with the MLE. 
Do not forget to change the working directory by choosing the preferred os.chdir().

NOTE: IN THIS CODE, the used data is in a different format. 

This code should be quick to compile

If you have any questions, please do ask me.
"""


if __name__ == '__main__':
    
    
    import os
    import time
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    import statistics
    from statistics import mean
    import numpy as np
    from scipy.stats import norm    
    from functools import reduce  
    import operator
    from scipy.optimize import minimize    
    from numpy import log    
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    from sklearn.model_selection import train_test_split
    # Unused parts of my code
    """
    
    def prod(iterable):
        return reduce(operator.mul, iterable, 1)
        
    """
    
    print(os.getcwd())
    os.chdir('/Users/hiro/DS-ML')
    print(os.getcwd())
    
    
    df = pd.read_csv('data_30countries.csv')
    columns = df.columns
    
    print(df.head())
    print(df.tail())
    print(df.sample(5))
    print(df.info())
    print(df.describe())
    print(df.isnull().sum())
    print(df.duplicated().sum())
    print(df.columns)
    print(df.dtypes)
    
    
    #######
    # 1) Ruling out subjects with incomplete surveys
    print(len(df)) # 81720 observations
    
    """
    indexes_to_drop = []
    
    for i in df['subject_global'].unique():
        df_temporary = df[df['subject_global'] == i]
        if len(df_temporary) != 28:
            indexes_to_drop.append(i)
            
    df = df[~df['subject_global'].isin(indexes_to_drop)]
    """
    print(len(df)) # 78232 observations. Dropped 145 individuals out of 2939
    
    dataframes = {}
    
    for i in range(0,17):
        if i <= 15:
            dataframes[f'df_{i}'] = df.iloc[:, i*10:i*10+9]
        if i == 16:
            dataframes[f'df_{i}'] = df.iloc[:, 160:165]
        
        
    #2) Basically, work with just the choice variables
    # First, let's test only question 3. Later, we could make a 17/16 cross validation procedure
    
    df0 = dataframes['df_0']
    
    #testing = df0[df0['equiv_nr'] == 3] 
    #training = df0[df0['equiv_nr'] != 3]
    
    train_df, test_df = train_test_split(df0, test_size=0.2, random_state=42)
    
    df1 = dataframes['df_1']
    df2 = dataframes['df_2']
    df3 = dataframes['df_3']
    df4 = dataframes['df_4']
    df5 = dataframes['df_5']
    df6 = dataframes['df_6']
    df7 = dataframes['df_7']
    df8 = dataframes['df_8']
    df9 = dataframes['df_9']
    df10 = dataframes['df_10']
    df11 = dataframes['df_11']
    df12 = dataframes['df_12']
    df13 = dataframes['df_13']
    df14 = dataframes['df_14']
    df15 = dataframes['df_15']
    df16 = dataframes['df_16']
    
    # 3) Stochastic returns positive, negative and mixed weights, modeled y, difference between PT and real and log normal to use MLE
    
    def stochastic(params,  df):
        alpha_positive, alpha_negative, beta_positive, beta_negative, lambd, noise = params
        df['w_positive'] = np.exp(-beta_positive * ((-np.log(df['probability'])) ** alpha_positive))
        df['w_negative'] = np.exp(-beta_negative * ((-np.log(df['probability'])) ** alpha_negative))
        df['w_mixed'] = np.exp(-beta_negative * ((-np.log(1-df['probability'])) ** alpha_negative))
        df['modeled_ce'] = np.nan
        df['modeled_y'] = np.nan
        #df['w_negative_complement_p'] = np.exp(-beta_negative * ((-np.log(1-df['probability'])) ** alpha_negative))
        df.loc[(df['high'] > 0) & (df['low'] >= 0) & (df['equiv_nr'] != 44), 'modeled_ce'] = \
            df['w_positive']*df['high'] + ((1-df['w_positive'])*df['low'])
        df.loc[(df['high'] < 0) & (df['low'] <= 0) & (df['equiv_nr'] != 44), 'modeled_ce'] = \
            df['w_negative']*df['high'] + ((1-df['w_negative'])*df['low'])
        #df.loc[(df['equiv_nr'] == 44), 'modeled_y'] = -((df['w_positive'] * df['high'])/(lambd*df['w_negative_complement_p']))
        #df.loc[(df['equiv_nr'] == 44), 'modeled_ce'] = ((df['w_positive']* df['high']) + (df['w_negative_complement_p']*df['modeled_y']))
        
        df.loc[(df['equiv_nr'] == 44), 'modeled_ce'] = -(df['w_positive']*df['high'])/(lambd * df['w_negative'])
        df['difference'] = df['modeled_ce'] - df['equivalent']
        
        #df['lnf'] = np.log(norm.pdf(df['PTdiff'], 0, noise * abs(df['high'] - df['low'])))
        df['lnf'] = np.log(norm.pdf(df['difference'], loc = 0, scale = noise * abs(df['high'] - df['low'])))
        #df['pdf'] = norm.pdf(df['difference']/(noise*abs(df['low']-df['high']))) # if gains
        
            
        df['eut'] = (df['probability']*df['high']) + (1-df['probability'])*df['low']
        
        return df 
    
    def calculate(params, df):
        alpha_positive, alpha_negative, beta_positive, beta_negative, lambd, noise = params
        df['w_positive'] = np.exp(-beta_positive * ((-np.log(df['probability'])) ** alpha_positive))
        df['w_negative'] = np.exp(-beta_negative * ((-np.log(df['probability'])) ** alpha_negative))
        df['w_mixed'] = np.exp(-beta_negative * ((-np.log(1-df['probability'])) ** alpha_negative))
        df['modeled_ce'] = np.nan
        df['modeled_y'] = np.nan
        #df['w_negative_complement_p'] = np.exp(-beta_negative * ((-np.log(1-df['probability'])) ** alpha_negative))
        df.loc[(df['high'] > 0) & (df['low'] >= 0) & (df['equiv_nr'] != 44), 'modeled_ce'] = \
            df['w_positive']*df['high'] + ((1-df['w_positive'])*df['low'])
        df.loc[(df['high'] < 0) & (df['low'] <= 0) & (df['equiv_nr'] != 44), 'modeled_ce'] = \
            df['w_negative']*df['high'] + ((1-df['w_negative'])*df['low'])
        #df.loc[(df['equiv_nr'] == 44), 'modeled_y'] = -((df['w_positive'] * df['high'])/(lambd*df['w_negative_complement_p']))
        #df.loc[(df['equiv_nr'] == 44), 'modeled_ce'] = ((df['w_positive']* df['high']) + (df['w_negative_complement_p']*df['modeled_y']))
        
        df.loc[(df['equiv_nr'] == 44), 'modeled_ce'] = -(df['w_positive']*df['high'])/(lambd * df['w_negative'])
        df['difference'] = df['modeled_ce'] - df['equivalent']
        
        #df['lnf'] = np.log(norm.pdf(df['PTdiff'], 0, noise * abs(df['high'] - df['low'])))
        df['lnf'] = np.log(norm.pdf(df['difference'], loc = 0, scale = noise * abs(df['high'] - df['low'])))
        #df['pdf'] = norm.pdf(df['difference']/(noise*abs(df['low']-df['high']))) # if gains
        
        return -np.sum(df['lnf'])
    
    #print(df0.columns)
    
    init_params = [0.8, 1.0, 0.8, 1.0, 1.5, 0.2]
    
    
    #
    
    def results(init_params,df):
        estimates = minimize(calculate, init_params, args=(df,), method='BFGS', options={'maxiter': 5000, 'disp':True})
        print(f"Paramaters: {np.round(estimates['x'],3)}")
        print(f"Standard errors: {np.round(np.sqrt(np.diag(estimates.hess_inv)), 3)}")

        return estimates
    
    #def predict(params, df):
    #    alpha_positive, alpha_negative, beta_positive, beta_negative, lambd, noise = params
    #    df['w_positive']  = np.exp(-beta_positive * ((-np.log(df['probability'])) ** alpha_positive))
    
    init_params = [0.8, 1.0, 0.8, 1.0, 1.5, 0.2]
    mle_estimates = results(init_params, df = df0)
    
    #hessian_inv = mle_estimates.hess_inv.matmat(np.eye(len(mle_estimates.x)))
    #mle_estimates['hess_inv']
    
    #Check if hessian is positive definite (all eigenvalues are positive)
    np.all(np.linalg.eigvals(mle_estimates['hess_inv']) > 0)
    tolerance = 0.05
    jacobian_small = np.all(np.abs(mle_estimates.jac) < tolerance)
    print(jacobian_small)
    
    
    #mle_estimates = minimize(calculate, init_params, args=(df0,), method='BFGS', options={'maxiter': 5000})
    #standard_errors = np.sqrt(np.diag(mle_estimates.hess_inv))
    
    alpha_positive, alpha_negative, beta_positive, beta_negative, lambd, noise = mle_estimates['x']
    params = {'alpha_positive': alpha_positive, 'alpha_negative': alpha_negative, 'beta_positive': beta_positive, 'beta_negative': beta_negative, \
              'lambd': lambd, 'noise':noise}
    #print(np.round(mle_estimates['x'],3)) # [0.602 0.641 0.908 0.941 1.939 0.23 ]
    #print(np.round(standard_errors, 3)) #[0.002 0.007 0.002 0.004 0.017 0.001]
    df0 = stochastic(init_params, df = df0)
    #The first se is a little bit strange.
    
    
    
    ###########################################################################
    
    
    train_df, test_df = train_test_split(df0, test_size=0.2, random_state=42)
    mle_estimates = results(init_params, df = train_df)
    mle_estimates['x']
    
    
    #mle_estimates = minimize(calculate, init_params, args=(df0,), method='BFGS', options={'maxiter': 5000})
    standard_errors = np.sqrt(np.diag(mle_estimates.hess_inv))
    print(standard_errors)
    test_df = stochastic(mle_estimates['x'], test_df)
    
    test_df.loc[(test_df['equiv_nr'] == 44), 'low'] = test_df['equivalent']
    test_df.loc[(test_df['equiv_nr'] == 44), 'equivalent'] = 0
    test_df.loc[(test_df['equiv_nr'] == 44), 'modeled_ce'] = (test_df['w_positive']*test_df['high']) + (test_df['w_negative']*test_df['low'])
    #train_df = stochastic(mle_estimates['x'], df = train_df)
    
    
    
    
    
    
    test_mse = mean_squared_error(test_df['modeled_ce'], test_df['equivalent']) # 16.40611760667535
    test_mae = mean_absolute_error(test_df['modeled_ce'], test_df['equivalent']) #2.9712378985372765
    
    test_mse_eut = mean_squared_error(test_df['eut'], test_df['equivalent']) #20.701760920623762
    test_mae_eut = mean_absolute_error(test_df['eut'], test_df['equivalent']) #3.1148383076628106
    
    
    
    print(test_mse, test_mae, test_mse_eut, test_mae_eut)