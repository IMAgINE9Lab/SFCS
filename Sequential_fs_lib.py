import numpy as np
from scipy.linalg import toeplitz
import time, itertools

class JFA(object):
    '''
     JFA Classs Object
        Initializing Attributes:
            feat_cost: feature aquistion cost 
            bins: number of bins conisdered when quantizing the feature space 
            neta: precision parameter used to quantize when finding all possible posterrior probabilities
            ext_clf: external classifier names list
            ext_v_list: external classifier weights list
            int_v: internal classifier weight
        Functions: 
            run: To find spesifications using given dataset e.g., misclassification cost matrix, feature ordering  
            fit: To find cost matrix for all possible posteriror probabilities         
    '''
    def __init__(self, config):
        self.feat_cost  = config['feat_cost']
        self.bins       = config['bins']
        self.neta       = config['neta']
        self.ext_clf    = config['clf_pool']
        self.ext_v_list = config['clf_weights']['ext']
        self.int_v      = config['clf_weights']['int']
        
    def run(self, data):        
        # preprocessing
        self.preprocess(data['Xtrain'], data['Ytrain'])     
        
    def fit(self,clf):  # training - to find cost matrix for all possible posteriror probabilities
    
        self.clf = clf #external clf list models and errors
        self.l = l(self.W,self.K,self.g_w,self.clf,self.ext_v_list, self.int_v) # compute l(pi_R,Q_R)
        
        self.J = np.zeros(shape=(self.K+1,len(self.W)))
        self.A =  np.zeros(shape=(self.K+1,len(self.W)))
        self.J[self.K] = self.l[self.K]  
    
        start = time.time() # start time
        for i in range(self.K-1,-1,-1):
            sigma= np.zeros(len(self.W))
            f_order = self.ordering[i]
    
            for j in range(self.bins): 
                np.seterr(divide='ignore', invalid='ignore')
                a = np.sum(np.multiply(self.W, self.feat_prob[f_order,:,j]),axis =1)
                b = np.divide(np.multiply(self.W, self.feat_prob[f_order,:,j]),a[:,None])
    
                if not(np.any(np.isnan(b))):
    
                    I = np.zeros(len(self.W))
                    for ind in range(len(b)):
                        diff = np.abs(self.W - b[ind])
                        I[ind] = np.where(np.sum(diff,axis=1) == np.min(np.sum(diff,axis=1)))[0][0]
    
                    sigma = sigma + np.multiply(a, self.J[i+1][I.astype(int)])
    
    
            self.A[i] = np.add(self.feat_cost[i], sigma) #continuing cost
            self.J[i] = np.minimum(self.l[i], self.A[i]) #cost at a stage 
      
        train_time = time.time() - start # training time
        return train_time
    
    def preprocess(self, Xtrain, Ytrain): 
  
        self.C = list(set(Ytrain))       # classes
        self.L = len(self.C)             # number of classes
        self.K = np.size(Xtrain,axis=1)  # number of features

        self.W = quantize_simplex(self.L, self.neta) # uniformly quantizing the probability simplex

        self.MC = cost_matrix(self.L) # misclassification cost matrix

        self.g_w = g(self.W, self.L, self.MC)  # optimum decision cost on each quantized point of the probability simplex

        # feature distributions and corresponding bin edges
        self.feat_prob, self.edges = compute_feat_prob(Xtrain, Ytrain, self.C, self.K, self.L, self.bins) 

        self.ordering = get_feat_ord(self.feat_prob, self.K, self.feat_cost) # feature ordering 

        
def quantize_simplex(L, neta):
    # uniformly quantizing the probability simplex
    w = np.linspace(0,1,neta+1)
    W = [seq for seq in itertools.product(w, repeat=L) if sum(seq) == 1]
    W = np.array(W)
    
    return W

def cost_matrix(L):
    # misclassification cost matrix
    arr = np.ones(L)
    arr[0] = 0
    MC = toeplitz(arr)
    
    return MC

def g(w,L,MC):
    g = []
    for j in range(L):
        g_j = np.sum(np.multiply(MC[j],w),axis =1)
        g.append(g_j)
    g_w = np.amin(g, axis=0)
    return g_w

def l(w,K,g_w,clf,ext_clf_weights,int_clf_weight):
   l = np.zeros(shape=(K+1,len(w)))
   for k in range(K+1):
       for p in range(len(g_w)):
           H = []
           for c in clf:               
               H.append(np.dot(clf[c]['clf_error'][k],w[p])) #calculate Total error proability = P(Error/True)P(True)
           l[k][p] = np.minimum(int_clf_weight*g_w[p],np.dot(ext_clf_weights,H))
   return l


def compute_feat_prob(Xtrain, Ytrain, C, K, L, bins):
    # compute feature distributions
    edges = np.zeros((K, bins+1))
    feat_prob = np.zeros((K, L, bins))
    
    for i in range(K):
        # discreterizing feature space
        min_r = np.floor(Xtrain[:,i].min())
        max_r = np.ceil(Xtrain[:,i].max())
        edges[i] = np.linspace(min_r, max_r, num = bins+1)
    
        for j in range(L):
            # feature ditributions conditioned on class C_j
            cpd = np.histogram(Xtrain[:,i][Ytrain == C[j]], bins=edges[i])[0]
            feat_prob[i,j,:] = (cpd+1)/(sum(cpd) + bins)
            
    return feat_prob, edges

def get_feat_ord(feat_prob, K,feat_cost): 
    # get feature ordering based on feature variance 
    PE_var = np.zeros(K) 
    # feature ordering             
    for mi in range(K):
        # error vector for features
        PE_var[mi] = sum(np.var(feat_prob[mi,:,:],axis =0))/feat_cost[mi] 
    ordering = sorted(range(len(PE_var)), key=lambda k:PE_var[k],reverse=True) 
    
    return ordering
    


