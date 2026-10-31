import numpy as np

class CS229_SVM:
    def __init__(self, learning_rate=0.01, lambda_param=0.01, n_iters=1000):
        self.learning_rate = learning_rate
        self.lambda_param = lambda_param
        self.n_iters = n_iters
        self.w = None
        self.b = None
    def fit(self,x,y):
        y_= np.where(y <= 0, -1, 1)#0 and 1 to -1 and 1
        n_samples,n_features = x.shape #recognize the shape of the input data low and columm
        self.w = np.zeros(n_features) #initialize the weight vector
        self.b = 0 #initialize the bias term
        for _ in range(self.n_iters): #iterate for the number of iterations
            for idx,x_i in enumerate(x):
                condition=y_[idx]*(np.dot(x_i,self.w)-self.b)>=1 #check the condition for the SVM
                if condition:
                    self.w-=self.learning_rate*(2*self.lambda_param*self.w) #update the weight vector
                else:
                    self.w-=self.learning_rate*(2*self.lambda_param*self.w-np.dot(x_i,y_[idx])) #update the weight vector
                    self.b-=self.learning_rate*y_[idx] #update the bias term
    def predict(self,x):
        approx=np.dot(x,self.w)-self.b #calculate the approximate value
        return np.sign(approx) #return the sign of the approximate value
    
    
    
                    