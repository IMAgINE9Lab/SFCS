# SFCS
SFCS is a Sequential Datum-wise Feature Acquisition and Classifier Selection Algorithm

# Citation
To cite our paper, please use the following reference:

S. P. Ekanayake and D. -S. Zois, "Sequential Datum–Wise Feature Acquisition and Classifier Selection," IEEE Transactions on Artificial Intelligence, Oct. 2023. (Accepted)

# Prerequisites
python 3.8 and the following libraries are used
- sklearn 0.0
- numpy 1.24.4
- scipy 1.5.0

# Files
- example.py: run SFCS for Diabetes dataset.
- SFCS_lib.py: contains functions to run SFCS. Sequential_fs_lib.py is required to run this file. 
- Sequential_fs_lib.py: contains functions for processing a dataset, such as creating cost matrices and extracting  label variables and feature attributes.

## How to use

1. Step 1:  Select Algorithm to run

      SFCS-2X or SFCS-3X

3. Step 2:  Define Parameter Values 

    - Weight for classifier 1 
    - Weight for classifier 2 
    - Weight for classifier 3 (only if SFCS-3X is selected)
    - Feature acquisition cost 
    - Number of bins considered when quantizing the feature space 
    - Precision parameter used to quantize the belief space (i.e., space of posterior probabilities)
        
4. Step 3: Import Dataset 

    Import a dataset and separate features and labels
    
5. Step 4: Run Algorithm 

    Run SFCS using features and labels as input
   
7. Step 5: Results 

    Print results: accuracy, average number of features acquired, training time, and testing time

## Example 
See example.py file



