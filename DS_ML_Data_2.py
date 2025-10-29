#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  2 22:06:49 2024

@author: hiro
"""

def clean2():
    
    import os
    
    
    from sklearn.model_selection import train_test_split
    
    import numpy as np
    import pandas as pd
    
    os.chdir('/Users/hiro/DS-ML')
    from DS_ML_Data import clean
    
    d, train_df, test_df = clean()
    
    ce_list = []
    
    #aa = d.columns.tolist()
    
    #bb = [f"ce_{i}" for i in range(28)]
    
    #aa = aa + bb
    
    
    #for i in bb:
    #    d[i] = np.nan
    
    
    indexes_to_drop = []
    
    for i in d['subject_global'].unique():
        df_temporary = d[d['subject_global'] == i]
        if len(df_temporary) != 28:
            indexes_to_drop.append(i)
            
    d = d[~d['subject_global'].isin(indexes_to_drop)]
    
    updated_rows = []
    
    for i in d['subject_global'].unique():
        df_temporary = d[d['subject_global'] == i].copy()
        df_temporary = df_temporary[['subject_global', 'x', 'y', 'p', 'ce', 'equiv_nr']]
    
        # Create the repeated CE DataFrame
        z = np.array(df_temporary[['ce']].T)[0]
        u = pd.DataFrame(z).T
        k = pd.DataFrame(np.repeat(u, len(df_temporary), axis=0))
        np.fill_diagonal(k.values, np.nan)
        k.columns = [f"ce_{j}" for j in range(28)]
    
        # Concatenate the original df_temporary and the new CE columns
        merged_df = pd.concat([df_temporary.reset_index(drop=True), k.reset_index(drop=True)], axis=1)
    
        # Collect updated rows to reassemble the full DataFrame later
        updated_rows.append(merged_df)
    
    # Step 3: Reconstruct the full DataFrame with updated values
    d_updated = pd.concat(updated_rows, ignore_index=True)

    columns = [col for col in d_updated.columns if col != 'ce'] + ['ce']
    d_updated = d_updated[columns]

    train_df_updated, test_df_updated = train_test_split(d_updated, test_size=0.2, random_state=42)

    return d_updated, train_df_updated, test_df_updated


    
