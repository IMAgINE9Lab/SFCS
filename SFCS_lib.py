import numpy as np
import warnings
from sklearn.naive_bayes import GaussianNB
from collections import Counter
from sklearn.metrics import accuracy_score
import time
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from sklearn import tree
from Sequential_fs_lib import JFA 
from sklearn.model_selection import train_test_split


warnings.simplefilter(action='ignore', category=FutureWarning)   
warnings.simplefilter(action='ignore', category=RuntimeWarning)  
warnings.simplefilter(action='ignore', category=UserWarning)

np.random.seed(42)

class SFCS(object):     
     """
        SFCS Classs Object
        Initializing Attributes:
            f: feature aquistion cost 
            b: number of bins conisdered when quantizing the feature space 
            n: precision parameter used to quantize when finding all possible posterrior probabilities
            ex_clf_list: external classifier names list
            ex_clf_weight_list: external classifier weights list
            ex_count: number of external classifiers
            split:s
        Functions: 
            run: main function of SFCS object which performs model preprocessing/training and testing. input must be a features and variables
           
     """   
     #initialize values       
     def __init__(self, config):
        self.f = config['feat_cost']
        self.b      = config['bins']
        self.n      = config['neta']
        self.ex_clf_list = config['clf_pool']
        self.ex_clf_weight_list = config['clf_weights']
        self.ex_count = config['algorithm']
        self.split = config['split']

     
     #to run the algorithm for given variables and features  
     def run(self, var, feat):
        #calculate internal classifier weight 
        in_clf_weight = 1 - np.sum(self.ex_clf_weight_list) 
        weights = {'int':in_clf_weight,
                   'ext':self.ex_clf_weight_list}
        
        config = {'feat_cost' : np.full(len(feat.columns),self.f), 'bins':self.b, 'neta' : self.n, 'clf_pool': self.ex_clf_list, 'clf_weights':weights}
        
        #%% Functions
        def find_range(val,edge):
            # find bin number of the given feature value
            for i,f in enumerate(edge):
                if val < f:
                    return i
            return len(edge)-1
        
        #%% Split
        F = feat.values 
        X = var.values
        
        F= F.astype(np.float)
        X= X.astype(np.float)
 
            
        Xtrain, Xtest, ytrain, ytest = train_test_split(F, X, test_size=self.split, random_state=0)

        #print('completed fold - ', folds_count+1)
        #Xtrain, Xtest = F[train_index], F[test_index]
        #ytrain, ytest = X[train_index], X[test_index]
        
        ytrain = ytrain.flatten()
        ytest = ytest.flatten()
        
        data = {'Xtrain': Xtrain, 'Ytrain': ytrain, 'Xtest':Xtest, 'Ytest':ytest}
        
        #create feature aquisition object
        jfs = JFA(config)
        jfs.run(data)
        
        #%% Preprocessing/Training - for external classifier
        
        #1.--------- Train each Classifier 
        forder = jfs.ordering
       
        #make sub dataset ordered by forder
        #train for each sub-dataset and store each model
        #store h_R: probability of error of external classifier using R features for each sub model
        models = {}
        Xtrain_list = []
        Xtest_list = []
        clf_dict = {}
    
        for ex_clf in self.ex_clf_list:
            clf_dict[ex_clf] = {}
            
        #for k = 0, error is 1/L,randomly select class
        error = []
        for c in jfs.C:
            error.append(1/jfs.L)
            
        for ex_clf in clf_dict:
           clf_dict[ex_clf]['clf_model'] = []
           clf_dict[ex_clf]['clf_error'] = []        
           clf_dict[ex_clf]['clf_error'].append(error)
    
        
        train_temp = [] #training time for each classifier with each sub datasets    
        for i in range (len(forder)):
            Xtrain_sub = Xtrain[:,forder[0:i+1]] #sub dataset
            Xtrain_list.append(Xtrain_sub)
            
            Xtest_sub = Xtest[:,forder[0:i+1]] #sub dataset
            Xtest_list.append(Xtest_sub)
              
            #for each cassifier in classifier pool
            #-train classifier 
            #-find classifier error # P(Error|True)
            #--store error for each classifier in the pool
            #--store classifier for each set of features 
            
            #TODO: If you would like to run for different external classifiers, you need to update below
            for ex_clf in self.ex_clf_list:
                start = time.time()
                if ex_clf == 'SVM':
                    clf = SVC(kernel= 'rbf',C=0.1,random_state=0).fit(Xtrain_sub, ytrain)
                elif ex_clf == 'DT':
                    clf = tree.DecisionTreeClassifier(random_state=0).fit(Xtrain_sub, ytrain)
                elif ex_clf == 'NB':    
                    clf = GaussianNB().fit(Xtrain_sub, ytrain)
                else:
                    raise ValueError('Wait!! No Classifier...!')
                train_temp.append(time.time() - start)
    
                clf_dict[ex_clf]['clf_model'].append(clf)
                
                # P(Error|True)
                ypred_train = clf.predict(Xtrain_sub) 
                check = ypred_train==ytrain #compare true label and predicted
                ytrain_count = dict(Counter(ytrain)) #find total instances for each True label 
                
                #for each class find P_error
                error = []
                for c in jfs.C:
                    f = check[np.where(ytrain==c)] #filter Check by each class
                    e_count = dict(Counter(f)) #count false of P(mismatch|class)
                    try: #calculate P(Error|True Class)
                        e = np.divide(e_count[False],ytrain_count[c])
                    except  KeyError: #when no False
                        e = float(0) #no error
                    error.append(e)
                
                clf_dict[ex_clf]['clf_error'].append(error)
    
        train_time_clf = np.sum(train_temp) #total training time for all external classifiers
        
        models['sub_Xtrain'] = Xtrain_list
        models['sub_Xtest'] = Xtest_list
        models['clf'] = clf_dict
        models['data'] = data
                    
        #1.--------- 
        train_time_A = jfs.fit(models['clf']) #To find stopping cost, cont cost and optimum cost at stage k
        
        #%% Testing 
        predictions = [] 
        n_feat = []
        pin_update = []    
        
        decision_track=[]
        start = time.time() # start time
        for z in range(np.size(Xtest,axis=0)):    
            obs = Xtest[z,:]  # test instance
            pin =  np.ones(jfs.L)/jfs.L # initial belief 
            
            temp_p = []
            for k in range(jfs.K):
                
                f_order = jfs.ordering[k]  # features  
                f = obs[f_order]  # observing a feature assignment
                f_index = find_range(f,jfs.edges[f_order][1:])   # index after discreterizing the feature    
    
                pin = np.divide(np.multiply(pin, jfs.feat_prob[f_order,:,f_index]), # belief update
                                   np.sum(np.multiply(pin,jfs.feat_prob[f_order,:,f_index])))         
                temp_p.append(pin)
                diff = np.abs(jfs.W - pin)
                pi_approx_ind = np.where(np.sum(diff,axis=1) == np.min(np.sum(diff,axis=1)))[0][0]
                if jfs.l[k][pi_approx_ind] <= jfs.A[k][pi_approx_ind]:
                    break
    
            n_feat.append(k+1) # number of features used    
            pin_update.append(temp_p) # posterior update for each instance
          
            l = [in_clf_weight*jfs.g_w[pi_approx_ind]]
            for c in self.ex_clf_list:               
                H_c = np.dot(models['clf'][c]['clf_error'][k],pin) #calculate total error proability = P(Error/True)P(True)
                l.append(self.ex_clf_weight_list[self.ex_clf_list.index(c)]*H_c)
                
            U_opt = np.argmin(np.array(l)) #with weights
    
            if U_opt >= 1: #External classifier 
                decision_track.append(U_opt) #Yes    
                ## Select decision from classifier 
                #       Use trained models relevent to features
                if k==0:
                    ypredict  = models['clf'][self.ex_clf_list[U_opt-1]]['clf_model'][k].predict(models['sub_Xtest'][k][z].reshape(-1,1))
                else:
                    ypredict  = models['clf'][self.ex_clf_list[U_opt-1]]['clf_model'][k].predict(models['sub_Xtest'][k][z].reshape(1,-1))
                predictions.append(ypredict[0])  
            else: #Internal classifier
                decision_track.append(0) #No
                D_opt = np.argmin(np.dot(pin,jfs.MC))
                ypredict = D_opt
                predictions.append(jfs.C[ypredict])
                
        test_time =  time.time()- start  # test time  
    
        # %% Summary 
        fold1 = {}
        
        #1. accuracy 
        acc = []
        accuracy = accuracy_score(ytest,predictions)
        acc.append(accuracy)
        fold1['accuracy'] = acc
         
        #--------------------------------------------------------------------------
        #2. avg. no. of features
        feat_avg = []
        feat_avg.append(np.mean(n_feat))
            
        fold1['no_of_features_avg']   =  feat_avg
        
        #--------------------------------------------------------------------------
        #3. train/test time
        temp=[]
        temp.append(train_time_A)
        temp.append(train_time_clf)
        fold1['loop_train_time'] = np.sum(temp)
        fold1['loop_test_time'] = test_time 
        
        #summary report 
        self.summary = {'accuracy': np.round(fold1['accuracy'][0],3), 'avg_feat': np.round(fold1['no_of_features_avg'][0],3),
                        'training_time': np.round(fold1['loop_train_time'],3), 'testing_time': np.round(fold1['loop_test_time'],3)}
        
            
        
            