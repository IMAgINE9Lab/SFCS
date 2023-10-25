# SFCS
SFCS is Sequential Datum-wise Feature Acquisition and Classifier Selection Algorithm

# Citation
To cite our paper, please use the following reference:


# Prerequisites
python 3.8  and the following libraries are used
- sklearn
- numpy
- scipy

# Files
- example.py: To run an example 
- SFCS_lib.py:  This contains functions to run SFCS.  Sequential_fs_lib.py is required to run this. 
- Sequential_fs_lib.py: This contains functions for specifications given a dataset, such as cost matrices, variable attributes, feature attributes

## How to use

1. Step 1:  Select Algorithm to run SFCS-2X or SFCS-3X

2. Step 2:  Define Parameter Values 


    - Weight for classifier 1 
    - Weight for classifier 2 
    - Weight for classifier 3 
    - Feature acquisition cost 
    - Number of bins considered when quantizing the feature space 
    - Precision parameter used to quantize when finding all possible posterior probabilities

        
3. Step 3: Import Dataset 

    Import a dataset and separate features and variables
    
4. Step 4: Run Algorithm 

    Run SFCS using features and variable as input
5. Step 5: Results 

    Print results: accuracy, average number of features, training time, and testing time to see the output

## Example 
See example.py file



