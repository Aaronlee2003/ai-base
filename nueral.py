import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    return x * (1 - x)

class NeuralNetwork:
    #Weight from 정규분포#
    def __init__(self, input_size, hidden_size, output_size):
        self.w1=np.random.randn(input_size, hidden_size)  # Weight for input to hidden layer
        self.w2=np.random.randn(hidden_size, output_size)  # Weight for hidden to output layer
        self.b1=np.random.randn(hidden_size)  # Bias for hidden layer
        self.b2=np.random.randn(output_size)  # Bias for output layer
    
    def forward(self, X):
        self.z1=np.dot(X,self.w1)+self.b1  # Linear combination for hidden layer
        self.a1=sigmoid(self.z1)  # Activation for hidden layer
        self.z2=np.dot(self.a1,self.w2)+self.b2  # Linear combination for output layer layer ussing result a1 
        return self.a2
        
    def backward(self, X, y, output, learning_rate):
        error=output-y  # Calculate error
        d_output=error*sigmoid_derivative(output)  # Derivative of output layer
        dw2=np.dot(self.a1.T,d_output)  # Weight update for hidden to output layer
        db2=np.sum(d_output,axis=0,keepdims=True)  # Bias update for output layer
        d_hidden=np.dot(d_output,self.w2.T)*sigmoid_derivative(self.a1)  # Derivative of hidden layer
        dw1=np.dot(X.T,d_hidden)  # Weight update for input to hidden layer
        db1=np.sum(d_hidden,axis=0,keepdims=True)  # Bias update for hidden layer
        self.w1-=learning_rate*dw1  # Update weights for input to hidden layer
        self.b1-=learning_rate*db1  # Update bias for hidden layer
        self.w2-=learning_rate*dw2  # Update weights for hidden to output layer
        self.b2-=learning_rate*db2  # Update bias for output layer
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
# 정답 라벨 (4x1 행렬)
y = np.array([[0], [1], [1], [0]])

# 4. 모델 생성 및 학습 (입력 2, 은닉 4, 출력 1)
nn = NeuralNetwork(input_size=2, hidden_size=4, output_size=1)
epochs = 10000
lr = 0.1

for epoch in range(epochs):
    # 순전파
    output = nn.forward(X)
    # 역전파 및 업데이트
    nn.backward(X, y, output, lr)
    
    if epoch % 2000 == 0:
        loss = np.mean(np.square(y - output)) # MSE 계산
        print(f"Epoch {epoch:5d} | Loss: {loss:.6f}")

# 5. 테스트 결과 출력
print("\n[최종 예측 결과]")
print(nn.forward(X))
        
  