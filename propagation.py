import numpy as np

def signoid(x):
    return 1/(1+np.exp(-x))
def sigoid_derivative(x):
    return x(1-x)


X = np.array([[0, 0],
              [0, 1],
              [1, 0],
              [1, 1]])
y=np.array([[0], [1], [1], [0]])
np.random(42)
W1 = np.random.randn(2, 3)
b1 = np.zeros((1, 3))
W2 = np.random.randn(3, 1)
b2 = np.zeros((1, 1))
learning_rate = 0.1

for epoch in range(10000):
    z1=np.dot(X,W1)+b1#
    a1=signoid(z1)
    
    
    z2=np.dot(X,W2)+b2
    a2=signoid(z1)#시그노이드를 이용하여 오차율 계산
    loss=np.mean((y-a2)**2)
    error=y-a2
    d_output=error*sigoid_derivative(a2)#결과뭘
    error_hidden=d_output.dot(W2,T)#행렬의 형태를 맞추기 위해 전치 이전의 오차율을 넘김
    d_hidden=error_hidden*sigoid_derivative(a1)
    W1 += X.T.dot(d_hidden) * learning_rate
    W2 += X.T.dot(d_hidden) * learning_rate
    b1 += np.sum(d_hidden, axis=0, keepdims=True) * learning_rate
    b2 += np.sum(d_hidden, axis=0, keepdims=True) * learning_rate
    if epoch % 2000 == 0:
        print(f"Epoch {epoch:4d} | Loss: {loss:.6f}")

print("\n--- 학습 완료 후 예측 결과 ---")
print(a2)
    
    
    
    
    
    