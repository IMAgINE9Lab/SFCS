"""
Algorithm: Sequential Datum-wise Feature Acquisition and Classifier Selection (SFCS)
This is an example using Diabetes dataset
"""
import numpy as np
from sklearn.datasets import fetch_openml
from SFCS_lib import SFCS

#%%Step 1:  Select Algorithm to run SFCS-2X or SFCS-3X

ex_count = 3#Define external classifier count
'''
    SFCS-2X: ex_count = 2 
    SFCS-3X: ex_count = 3
'''
if ex_count == 2:
    ex_clf_list = ['NB','SVM']
else:
    ex_clf_list = ['NB','SVM','DT']     
'''
    NB: Nive Bayes 
    SVM: Support Vector Machines
    DT: Decision Tree 
'''
    
#%%Step 2:  Define parameter values 

'''
    C1_w: weight for classifier 1 (0,1)
    C2_w: weight for classifier 2 (0,1)
    C3_w: weight for classifier 3 (0,1)
    f: feature aquistion cost 
    b: number of bins considered when quantizing the feature space
    n: precision parameter used to quantize when finding all possible posterior probabilities
'''
if ex_count == 2: #SFCS-2X
    vs = {
    'C1_w' : 0.3, 
    "C2_w" : 0.3, 
    "f" : 0.0001, 
    "b" : 20, 
    "n" : 10
    }
    weights = [vs['C1_w'],vs['C2_w']]
else: ##SFCS-3X
    vs = { 
    'C1_w' :  0.2, 
    "C2_w" : 0.2, 
    "C3_w" : 0.1, 
    "f" : 0.0001,
    "b" : 50, 
    "n" : 10
    }
    weights = [vs['C1_w'],vs['C2_w'],vs['C3_w']]
        
#%%Step 3: Import Dataset [features, variables]
'''
    Import a dataset and separte features and variables
    var: classifiing variable
    feat: features
'''
dataset = fetch_openml(name='diabetes', as_frame=True, version =1)
feat = dataset.data #features
var = dataset.target #target
var = var.to_frame()
#add values to features with string - this is an optional step depend on your dataset
list_cols = list(var.columns[var.dtypes!=float])
for column in list_cols:
    var[column] = var[column].astype('category').cat.codes[:]
    
split = 0.3 #train_test_split test_size    
#%%Step 4: Run Algorithm 
'''
    Input parameter values and specifications 
    Run SFCS using features and variable as input
'''

#all parameter values as an input to SFCS
config = {'algorithm':ex_count,'feat_cost' : np.full(len(feat.columns),vs['f']), 'bins':vs['b'], 'neta' : vs['n'], 'clf_pool': ex_clf_list, 'clf_weights': weights,'split':split}

clf = SFCS (config)
clf.run(var, feat)

#%%Step 5: Results 
'''
    Print results to see the output
'''
print("\nResults: "+str(clf.summary)+'\n')
