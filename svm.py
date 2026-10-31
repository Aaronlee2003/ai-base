import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


iris = datasets.load_iris()
X = iris.data[:, :2]  # Use only the first two features for visualization
y= iris.target


binary_mask = y != 2
X_binary = X[binary_mask]
y_binary = y[binary_mask]


scaler = StandardScaler()
X_binary_scaled = scaler.fit_transform(X_binary)


xtrain, xtest, ytrain, ytest = train_test_split(X_binary_scaled, y_binary, test_size=0.2, random_state=42)


svm_model = SVC(kernel='linear', C=1.0)
svm_model.fit(xtrain, ytrain)


y_pred = svm_model.predict(xtest)

print("Accuracy:", accuracy_score(ytest, y_pred))

plt.figure(figsize=(8, 6))


ax=plt.gca()
plt.scatter(X_binary_scaled[:, 0], X_binary_scaled[:, 1], c=y_binary, cmap='coolwarm', edgecolors='k', s=50)

xlim = ax.get_xlim()
ylim = ax.get_ylim()
xx,yy=np.meshgrid(np.linspace(xlim[0], xlim[1], 50), np.linspace(ylim[0], ylim[1], 50))

Z = svm_model.decision_function(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
ax.contour(xx, yy, Z, colors='k', levels=[-1, 0, 1], alpha=0.5, linestyles=['--', '-', '--'])
ax.scatter(svm_model.support_vectors_[:, 0], svm_model.support_vectors_[:, 1], s=150,
           facecolors='none', edgecolors='k', label='Support Vectors')

plt.title("SVM Linear Kernel Decision Boundary")
plt.xlabel("Sepal Length (Scaled)")
plt.ylabel("Sepal Width (Scaled)")
plt.legend()
plt.show()

