import numpy as np
class deciscionstump:
    def __init__(self):
        self.polarity=1
        self.feature_idx=None
        self.thresthold=None
        self.alpha=None

    def    predict(self,x):
        n_samples=x.shape[0]#Wfind the row length
        x_col=x[ :, self.features_idx]#pick certain index  so 

        predictions=np.ones(n_samples)#makle array length n fill with 1




        if self.polarity==1:
            predictions[x_col<self.thresthold]=-1
        else:#polarity=-1
            predictions[x_col>self.thresthold]=-1

        return predictions
class  adaboost:
    def __int__(self,n_clf=5):
        self.n_clf=n_clf
        self.n_clf=[]



        def fit(self,x,y):
            n_samples,n_features=x.shape
            #initial weight give 1of n
            w=np.full(n_samples,(1/n_samples))
            for i in range(self.n_clf):
                clf=deciscionstump()#make objectg
                min_error=float('inf')#fil with infinity value

                for feature_i  in range(n_features): #n_features rowㅁ으 ㅣ 열 데이터마ㅓㄴ 섵냇
                    x_column=x[:,feature_i]
                    thresholds=np.unique(x_column)

                    for threshold in thresholds:
                        for polarity in [1,-1]:
                            predictions=np.ones(n_samples)
                            if polarity==1:
                                predictions[x_column<threshold]=-1
                            else:
                                predictions[x_column>threshold]=-1

                            error=sum(w[y!=predictions])
                            if error  <min_error:
                                min_error=error
                                clf.polarity=polarity
                                clf.thresthold=threshold
                                clf.feature_idx=feature_i


            eps=1e-10
            clf.alpha=-.5*np.log((1.0-min_error+eps)/(min_error+eps))
            predictions=clf.predict(x)
            w*=np.exp(-clf.alpha*y*predictions)
 
            w/=np.sum(w)
            self.clfs.append(clf)

    def predict(self,x):
        clf_preds=[clf.alpha*clf.predict(x)for clf in self.clf]
        y_pred=np.sum(clf_preds,axis=0)
        retrun np.sign(y_pred)