#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 29 16:18:01 2024

@author: hiro
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
    
    from functools import reduce  # Required in Python 3
    import operator
    
    def prod(iterable):
        return reduce(operator.mul, iterable, 1)
    
    from scipy.optimize import minimize
    
    from numpy import log
    
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
    
    
    indexes_to_drop = []
    
    for i in df['subject_global'].unique():
        df_temporary = df[df['subject_global'] == i]
        if len(df_temporary) != 28:
            indexes_to_drop.append(i)
            
    df = df[~df['subject_global'].isin(indexes_to_drop)]
    
    dataframes = {}
    
    for i in range(0,16):
        if i <= 15:
            dataframes[f'df_{i}'] = df.iloc[:, i*10:i*10+9]
        if i == 16:
            dataframes[f'df_{i}'] = df.iloc[:, 160:165]
        
    df0 = dataframes['df_0']
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
    #df16 = dataframes['df_16']
    
    
    df0['weights'] = (df0['equivalent']-df0['low'])/(df0['high']-df0['low'])
    df0['modeled'] = (df0['weights']*df0['low'])+((1-df0['weights'])*df0['high'])
    
    positive_w_half = []
    negative_w_half = []
    lambd = []
    
    for i in df0['subject_global'].unique():
        df_temporary = df0[df0['subject_global'] == i].copy()
        #calculating w+(0.5) as mean of w+(0.5), similarly to w-(0.5)
        df_temporary1 = df_temporary[(df_temporary['probability'] == 0.5) & (df_temporary['high']> 0) & (df_temporary['equiv_nr'] != 44)]
        df_temporary2 = df_temporary[(df_temporary['probability'] == 0.5) & (df_temporary['high'] <= 0)]
        
        positive_mean = df_temporary1['weights'].mean()
        negative_mean = df_temporary2['weights'].mean()
        
        positive_w_half.extend([positive_mean] * len(df_temporary))
        negative_w_half.extend([negative_mean] * len(df_temporary))
        
        df_temporary3 = df_temporary[df_temporary['equiv_nr'] == 44]
        
        if not df_temporary3.empty:
            lamb = (positive_mean * df_temporary3['high']) / (negative_mean * df_temporary3['equivalent'])
            lamb = lamb.values[0] if len(lamb) > 0 else np.nan
        else:
            lamb = np.nan
            
        lambd.extend([lamb] * len(df_temporary))
        
    df0['positive_w_half'] = positive_w_half
    df0['negative_w_half'] = negative_w_half
    df0['lambda'] = lambd
        
    #df0 = df0.dropna()
    #df0['modeled_equivalent_loss'] = (df0[''])
    
    print(mean(df0['lambda']))
    print(df0['lambda'].describe())
    
    print
    
    
    
    df0['lambda'].hist(bins = 5)
    
    
    
    def stochastic(alpha_positive, alpha_negative, beta_positive, beta_negative, lambd, noise, df):
        df['w_positive'] = np.exp(-beta_positive * ((-np.log(df['probability'])) ** alpha_positive))
        df['w_negative'] = np.exp(-beta_negative * ((-np.log(df['probability'])) ** alpha_negative))
        df['w_mixed'] = np.exp(-beta_negative * ((-np.log(1-df['probability'])) ** alpha_negative))
        df['modeled_ce'] = np.nan
        df.loc[(df['high'] > 0) & (df['low'] >= 0) & (df['equiv_nr'] != 44), 'modeled_ce'] = df['w_positive']*df['low'] + ((1-df['w_positive'])*df['high'])
        df.loc[(df['high'] < 0) & (df['low'] <= 0) & (df['equiv_nr'] != 44), 'modeled_ce'] = df['w_negative']*df['low'] + ((1-df['w_negative'])*df['high'])
        df.loc[(df['equiv_nr'] == 44), 'modeled_ce'] = df['w_positive']*df['low'] + ((1-df['w_negative'])*df['high'])
        
        #df['modeled_ce'] = df['w_positive']*df['low'] + ((1-df['w_positive'])*df['high']) #if true and not 44
        #df['modeled_ce'] = df['w_negative']*df['low'] + ((1-df['w_negative'])*df['high']) #if true and not 44
        #df['modeled_ce'] = (df['w_positive']*df['high'])/(lambd*df['w_negative'])
        #Check the signs later
        ## TO BE SUPER PRECISE, THIS IS Y, BUT WE SHALL JUST USE THE SAME

        df['difference'] = df['modeled_ce'] - df['equivalent']
        
        df['lnf'] = np.log(norm.pdf(df['difference']/(noise*abs(df['low']-df['high']))))
        df['pdf'] = norm.pdf(df['difference']/(noise*abs(df['low']-df['high']))) # if gains
        
        return df    
    
    def calculate(params, df):
        alpha_positive, alpha_negative, beta_positive, beta_negative, lambd, noise = params
        df['w_positive'] = np.exp(-beta_positive * ((-np.log(df['probability'])) ** alpha_positive))
        df['w_negative'] = np.exp(-beta_negative * ((-np.log(df['probability'])) ** alpha_negative))
        df['w_mixed'] = np.exp(-beta_negative * ((-np.log(1-df['probability'])) ** alpha_negative))
        df['modeled_ce'] = np.nan
        df.loc[(df['high'] > 0) & (df['low'] >= 0) & (df['equiv_nr'] != 44), 'modeled_ce'] = \
            df['w_positive']*df['high'] + ((1-df['w_positive'])*df['low'])
        df.loc[(df['high'] < 0) & (df['low'] <= 0) & (df['equiv_nr'] != 44), 'modeled_ce'] = \
            df['w_negative']*df['high'] + ((1-df['w_negative'])*df['low'])
        df.loc[(df['equiv_nr'] == 44), 'modeled_ce'] = -(df['w_positive'] * df['high']/lambd*df['w_negative'])
        
        #df['modeled_ce'] = df['w_positive']*df['low'] + ((1-df['w_positive'])*df['high']) #if true and not 44
        #df['modeled_ce'] = df['w_negative']*df['low'] + ((1-df['w_negative'])*df['high']) #if true and not 44
        #df['modeled_ce'] = (df['w_positive']*df['high'])/(lambd*df['w_negative'])
        #Check the signs later
        ## TO BE SUPER PRECISE, THIS IS Y, BUT WE SHALL JUST USE THE SAME
        
        
        df['difference'] = df['modeled_ce'] - df['equivalent']
        
        #df['lnf'] = np.log(norm.pdf(df['PTdiff'], 0, noise * abs(df['high'] - df['low'])))
        df['lnf'] = np.log(norm.pdf(df['difference'], loc = 0, scale = noise * abs(df['high'] - df['low'])))
        #df['pdf'] = norm.pdf(df['difference']/(noise*abs(df['low']-df['high']))) # if gains
        
        return -np.sum(df['lnf'])
    
    #print(df0.columns)
    
    
    init_params = [0.8, 1.0, 0.8, 1.0, 1.5, 0.2]
    calculate(init_params, df = df0)
    
    testing = minimize(calculate, init_params, args=(df0,), method='BFGS', options={'maxiter': 5000})

    # Display the result
    print(testing)
    
    
    
    """
    individual_mle = []
    
    
    import time
    start = time.time()
    
    init_params = [0.8, 1.0, 0.8, 1.0, 1.5, 0.2]
    for i in df0['subject_global'].unique():
        start2 = time.time()
        df_temporary = df0[df0['subject_global'] == i].copy()
        a = minimize(calculate, init_params, args=(df_temporary,), method='BFGS', options={'maxiter': 500})
        print(i)
        individual_mle.append(a)
        end2 = time.time()
        print(end2 - start2)
    end = time.time()
    print(end - start)
    
    individual_mle_values = []
    a = []
    b = []
    c = []
    d = []
    e = []
    f = []
    for i in individual_mle:
        a.append(float(i['x'][0]))
        b.append(float(i['x'][1]))
        c.append(float(i['x'][2]))
        d.append(float(i['x'][3]))
        e.append(float(i['x'][4]))
        f.append(float(i['x'][5]))
    
    # 24237.29720377922 secs = 6.7325 hours
    
    a1, b1, c1, d1, e1, f1 = np.array(a),np.array(b),np.array(c),np.array(d),np.array(e),np.array(f)
    
    mle_df = pd.DataFrame({'alpha_positive': a,
                           'alpha_negative': b,
                           'beta_positive': c,
                           'beta_negative':d,
                           'lambd':e,
                           'noise':f})
    
    #mle_df.columns = ['alpha_positive', 'alpha_negative', 'beta_positive', 'beta_negative', 'lambd', 'noise']
    
    """
    
    import pandas as pd
    
    
    from sklearn.cluster import DBSCAN
    from sklearn.preprocessing import StandardScaler
    
    
    choice_df = np.array
    
    
    a = []
    
    for i in df0['subject_global'].unique():
        df_temporary = df0[df0['subject_global'] == i]
        df_temporary = np.array(df_temporary[['equivalent']].T)
        a.append(df_temporary)
    choice_df = pd.DataFrame(np.concatenate(a))
    
        
    # Initialize DBSCAN with chosen parameters
    dbscan = DBSCAN(eps = 10, min_samples=5)
    
    # Fit DBSCAN to the standardized data
    clusters = dbscan.fit_predict(choice_df)
    clusters.sum()
    # Add the cluster labels to the original DataFrame
    df['cluster'] = clusters
    
    # Display the first few rows with the cluster labels
    print(df.head())
    
    from scipy.spatial.distance import cdist
    
    distance_matrix_manhattan = cdist(choice_df, choice_df, metric = 'cityblock')
    
    distance_df_manhattan = pd.DataFrame(distance_matrix_manhattan, 
                                     columns=[f'row_{i}' for i in range(len(choice_df))], 
                                     index=[f'row_{i}' for i in range(len(choice_df))])
    
    
    a = distance_df_manhattan[:1]
    a = np.array(a)
    a = pd.DataFrame(a)

    a = a.values.tolist()
    a = a[0]
    a = np.array(a)
    a = pd.DataFrame(a)
    plt.show()
    a.hist()
    
    print(distance_df_manhattan.head())
    
    
    
    import hdbscan
    
    np.random.seed(0)
    z = pd.DataFrame(np.random.rand(2794, 5), columns=['col1', 'col2', 'col3', 'col4', 'col5'])

    # Initialize HDBSCAN model
    hdbscan_model = hdbscan.HDBSCAN(min_cluster_size=10, min_samples=1, metric='euclidean')
    
    # Fit the model to your dataset
    hdbscan_model.fit(df)
    
    # Get the labels (cluster assignments)
    labels = hdbscan_model.labels_
    
    # Print the labels for each point
    print(labels)
    
    import seaborn as sns
    from sklearn.decomposition import PCA
    
    # Reduce dimensionality using PCA (for visualization purposes)
    pca = PCA(n_components=2)
    df_pca = pca.fit_transform(df)
    
    # Plot the results
    plt.figure(figsize=(10, 7))
    sns.scatterplot(x=df_pca[:, 0], y=df_pca[:, 1], hue=labels, palette='tab10', s=100)
    plt.title('HDBSCAN Clustering')
    plt.show()
    
    # Cluster membership strength (the higher the value, the stronger the cluster association)
    probabilities = hdbscan_model.probabilities_
    
    # Outlier scores (higher scores indicate that a point is likely an outlier)
    outlier_scores = hdbscan_model.outlier_scores_
    
    # Print probabilities and outlier scores
    print(probabilities)
    print(outlier_scores)


    ### LEAVE ONE-OUT CROSS VALIDATION
    
    
    
    
    a
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    ###################################################################################################

    """
    Index(['subject_global', 'equiv_nr', 'high', 'low', 'probability',
           'equivalent', 'subject', 'country', 'country_name', 'weights',
           'modeled', 'positive_w_half', 'negative_w_half', 'lambda'],
          dtype='object')
    """
    
    
    # Example usage with sample parameters
    params = {
        'alpha_positive': 1.0,
        'alpha_negative': 0.9,
        'beta_positive': 0.6,
        'beta_negative': 0.5,
        'lambd': 1.0,
        'noise': 0.1
    }
    
    alpha_positive = 1.0
    alpha_negative= 0.9
    beta_positive= 0.6
    beta_negative= 0.5
    lambd= 1.0
    noise= 0.1
    
    # Apply the function to your data
    results = stochastic(**params, df=df0)
    
    # Display first few rows of results
    print(results.head())


    """
    # Define the x range
    x = np.linspace(0.0001, 1, 1000)

    # Define the functions for different values of alpha
    y_alpha_09 = np.exp(-1 * (-np.log(x))**0.9)
    y_alpha_07 = np.exp(-1 * (-np.log(x))**0.7)
    y_alpha_05 = np.exp(-1 * (-np.log(x))**0.5)
    y_alpha_03 = np.exp(-1 * (-np.log(x))**0.3)
    y_alpha_01 = np.exp(-1 * (-np.log(x))**0.1)
    y_linear = x

    # Plot the functions
    plt.plot(x, y_alpha_09, color='darkgreen', linewidth=2, label='alpha = 0.9')
    plt.plot(x, y_alpha_07, color='yellow', linewidth=2, label='alpha = 0.7')
    plt.plot(x, y_alpha_05, color='red', linestyle='--', linewidth=2, label='alpha = 0.5')
    plt.plot(x, y_alpha_03, color='green', linewidth=2, label='alpha = 0.3')
    plt.plot(x, y_alpha_01, color='blue', linewidth=2, label='alpha = 0.1')
    plt.plot(x, y_linear, color='black', linestyle='-.', label='linear')

    # Customize the graph
    plt.legend()
    plt.title("Weighting Functions for Different Values of Alpha")
    plt.xlabel("p")
    plt.ylabel("w(p)")
    plt.xticks(np.arange(0, 1.1, 0.1))
    plt.yticks(np.arange(0, 1.1, 0.1))
    plt.grid(True)

    # Show the plot
    plt.show()

    # Define the functions for different values of beta
    y_beta_06 = np.exp(-0.6 * (-np.log(x))**0.65)
    y_beta_08 = np.exp(-0.8 * (-np.log(x))**0.65)
    y_beta_1 = np.exp(-1 * (-np.log(x))**0.65)
    y_beta_12 = np.exp(-1.2 * (-np.log(x))**0.65)
    y_beta_14 = np.exp(-1.4 * (-np.log(x))**0.65)

    # Plot the functions
    plt.plot(x, y_beta_06, color='darkgreen', linewidth=2, label='beta = 0.6')
    plt.plot(x, y_beta_08, color='yellow', linewidth=2, label='beta = 0.8')
    plt.plot(x, y_beta_1, color='red', linestyle='--', linewidth=2, label='beta = 1')
    plt.plot(x, y_beta_12, color='green', linewidth=2, label='beta = 1.2')
    plt.plot(x, y_beta_14, color='blue', linewidth=2, label='beta = 1.4')
    plt.plot(x, y_linear, color='black', linestyle='-.', label='linear')

    # Customize the graph
    plt.legend()
    plt.title("Weighting Functions for Different Values of Beta")
    plt.xlabel("p")
    plt.ylabel("w(p)")
    plt.xticks(np.arange(0, 1.1, 0.1))
    plt.yticks(np.arange(0, 1.1, 0.1))
    plt.grid(True)

    # Show the plot
    plt.show()
    """    
    

    
    """
    def mle(parameters, df):
        individ_likelihood = []
        alpha_positive, alpha_negative, beta_positive, beta_negative, lambd, noise = parameters
        params = {
            'alpha_positive': alpha_positive,
            'alpha_negative': alpha_negative,
            'beta_positive': beta_positive,
            'beta_negative': beta_negative,
            'lambd': lambd,
            'noise': noise
        }
        

        
        for i in df['subject_global'].unique():
            df_temporary = df0[df0['subject_global'] == i].copy()
            result = stochastic(**params, df = df_temporary)
            var = np.log(result['pdf']) #multiply by each person
            individ_likelihood.append(prod(var))
            
        individ_likelihood = np.array(individ_likelihood)
        LL = -np.sum(individ_likelihood)
        
        return LL
    
    """
    
    
    
    
    
    
    def mle(parameters, df):
        individ_likelihood = []
        alpha_positive, alpha_negative, beta_positive, beta_negative, lambd, noise = parameters
        params = {
            'alpha_positive': alpha_positive,
            'alpha_negative': alpha_negative,
            'beta_positive': beta_positive,
            'beta_negative': beta_negative,
            'lambd': lambd,
            'noise': noise
        }
        
        result = stochastic(**params, df = df)

        print(result)
        for i in result['subject_global'].unique():
            df_temporary = result[result['subject_global'] == i].copy()
            var = prod(df_temporary['pdf']) #multiply by each person
            individ_likelihood.append(np.log(var))
            
            
        individ_likelihood = np.array(individ_likelihood)
        LL = -np.sum(individ_likelihood)
        
        return LL
    mle([0.8, 0.8, 1, 1, 1.5, 0.2], df0)
    """
    
    
    
    """
    """
        
        
            
        
        for i in df0['subject_global'].unique():
            df_temporary = df0[df0['subject_global'] == i].copy()
            #calculating w+(0.5) as mean of w+(0.5), similarly to w-(0.5)
            df_temporary1 = df_temporary[(df_temporary['probability'] == 0.5) & (df_temporary['high']> 0) & (df_temporary['equiv_nr'] != 44)]
            df_temporary2 = df_temporary[(df_temporary['probability'] == 0.5) & (df_temporary['high'] <= 0)]
            
            positive_mean = df_temporary1['weights'].mean()
            negative_mean = df_temporary2['weights'].mean()
            
            positive_w_half.extend([positive_mean] * len(df_temporary))
            negative_w_half.extend([negative_mean] * len(df_temporary))
            
        np.sum()
    
    def log_likelihood(params, data):
        alpha, beta, gamma, delta, lambd, noise = params
        # Example: model for gains and losses (this can be adjusted based on your formula)
        likelihood = -np.log(data['probability'])**0.9 * alpha  # Adjust according to your model
        # Penalizing negative likelihood
        return -np.sum(likelihood)
    """
    # Dummy data to simulate
    #data = {'probability': np.random.uniform(0.01, 1, 100)}
    
    # Initial values for parameters
    initial_params = [0.8, 0.8, 1, 1, 1.5, 0.2]
    
    # Maximizing the log-likelihood using BFGS method
    result = minimize(mle, initial_params, args=(df0,), method='BFGS', options={'maxiter': 5000})
    print(result)

        

    # Define a function for parameter estimation (replace this with your actual model)
    def estimate_params(country_data):
        # This is a dummy function to simulate parameter estimation.
        # Replace with your own logic for 'ml model' in Python.
        
        def objective(params):
            alpha_positive, alpha_negative, beta_positive, beta_negative, lambd, sigma = params
            # Simulate some fitting logic here, e.g., likelihood functions
            return np.sum((country_data['probability'] * alpha_positive - country_data['equivalent']) ** 2)

        # Initial guess for the parameters (replace with relevant values)
        initial_guess = [0.8, 1, 0.8, 1, 1.5, 0.2]
        
        # Minimize the objective function to find optimal parameters
        result = minimize(objective, initial_guess, method='BFGS')
        return result.x  # Returns the estimated parameters

    # Create empty lists to store results
    results = {'country': [], 'alpha': [], 'beta': [], 'gamma': [], 'delta': [], 'lambda': [], 'sigma': []}

    # Loop over each country
    for country in data['country'].unique():
        country_data = data[data['country'] == country]
        estimated_params = estimate_params(country_data)
        
        # Append the results
        results['country'].append(country)
        results['alpha'].append(estimated_params[0])
        results['beta'].append(estimated_params[1])
        results['gamma'].append(estimated_params[2])
        results['delta'].append(estimated_params[3])
        results['lambda'].append(estimated_params[4])
        results['sigma'].append(estimated_params[5])


    # Convert results to a DataFrame
    
    results_df = pd.DataFrame(results)

    # Display the results
    print(results_df)

        
    """





































    
    #a = df0.index[df0['lambda'] == df0['lambda'].min()].tolist()
    #z = df0.loc[a]
    #b = 
    
    #df0.drop(df0[df0['lambda'] == df0['lambda'].min()].index, inplace=True)
    #df0.drop(df0[df0['lambda'] == df0['lambda'].min()].index, inplace=True)
    #df0['lambda'].hist(bins = 10)
    #print(df0['lambda'].describe())
    
    #w = (ce-y)/(x-y)
    #l = 
    
    """
    for col in df.select_dtypes(include='object').columns:
        print(f'{col}: {df[col].unique()}')
        
    """
    """
    start_time = time.time()
    sns.pairplot(df)  # For numerical variables
    plt.show()
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Execution time: {execution_time} seconds")
    
    
    
    start_time = time.time()
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm')  # Correlation heatmap
    plt.show()
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Execution time: {execution_time} seconds")
    """
