import numpy as np
import pandas as pd
import networkx as nx
from scipy.special import gammaln
import matplotlib.pyplot as plt
import time

# Function to count occurances
def count_func (input_data,dag):
    
    H = dag
    d = input_data
    df_return = list()

    # For each 'node'
    for i in d.columns:
        # Find predecessors
        ls = list(H.predecessors(i))

        # If there are predecessors
        if len(ls) != 0:
            
            # Pivot along the node and parents.  Count occurrances of values.
            # Return array.
            df = pd.pivot_table(d[[i] + ls],index=ls,columns=[i],
             aggfunc=len).fillna(0).values
        else:
            # If no predecessors, count value of occurances
            d_sub = d[i].value_counts().fillna(0).values
            df = np.array(d_sub)

        #print(df)
        #print(df_return)

        # Append to list
        df_return.append(df)
    
    return df_return

# Calculate the Bayes Score of network
def score_func(counted_data):

    # Initialize value
    bayes_score = 0

    # For each node in the network, calculate score of counted data
    for i,m in enumerate(counted_data):

        # Instantiate alpha array.  Assuming prior is uniform, alpha = 1
        alphas = np.ones(m.shape)
        
        # If no parents (q=1), no sum along column required.
        # Calculate first part
        if m.ndim < 2:
            val_0 = (gammaln(alphas.sum()) - gammaln(alphas.sum() + m.sum())).sum()
        else:
            val_0 = (gammaln(alphas.sum(axis=1)) - gammaln(alphas.sum(axis=1) + m.sum(axis=1))).sum()

        # Calculate second part
        # Sum along rows for all r_i
        val_k = (gammaln(alphas + m) - gammaln(alphas)).sum(axis=0).sum()

        #print(val_0,val_k)

        # Sum both values and add to total score for network.
        bayes_score += (val_0 + val_k)
        #print(bayes_score)

    return bayes_score