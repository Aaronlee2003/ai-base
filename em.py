import numpy as np
np.random.seed(42)
data1=np.random.normal(loc=0,scale=1,size=300)#loc is average, scale is standard deviation, size is number of data points
data2=np.random.normal(loc=5,scale=1.5,size=700)

x=np.half

n=len(x)
k=2


mu=np.array([-1.0,1.0])
sigma=np.array([1.0,1.0])
pi=np.array([0.5,0.5])


def gaussain(x,mean,var):
    return (1/np.sqrt(2*np.pi*var))*np.exp(-((x-mean)**2)/(2*var))


max_iter=100
tol=1e-6#if the log likelihood is less than this value, we will stop the algorithm
prev_log_likelihood=0 
#log likehood is 현재 모델 파라미터(평균, 분산, 비중)가 주어진 전체 데이터 1,000개를 만들어냈을 확률(가능성)"입니다.

for step in range(max_iter):
    likelihoods=np.zeros((n,k))#we find the
    for j in range(k):
        likelihoods[:,j]=pi[j]*gaussain(x,mu[j],sigma[j])# this matrix 1000 by 2 matrix   p[k] means  kth group 차지하는  ratio  gaussian is  data x is  how many near k th gaussian distribution size of this group* possibiklity of data from this group  


    sum_likelihoods=np.sum(likelihoods,axis=1,keepdims=True)
    r=likelihoods/sum_likelihoods
    log_likelihood=np.sum(np.log(sum_likelihoods))
    n_k=np.sum(r,axis=0)# 각 그룹에서  x번 데이터가 나올 확률

    for  i in range(k):
        mu[i]=np.sum(r[:,i]*x)/n_k[i]#분모는 x가 k번째 그룹에 속할 확률/k그룹에 속학 하는 데이터의 수 k에 가까울수록 반영비 업
        sigma[i]=np.sum(r[:,i]*(x-mu[i])**2)/n_k[i]
        pi[i]=n_k[i]/n


    if np.abs(log_likelihood-prev_log_likelihood)<tol:
        break
    prev_log_likelihood=log_likelihood
