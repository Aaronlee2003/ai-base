from sklearn.datasets import load_iris
from sklearn.decomposition import PCA   

iris = load_iris()
X = iris.data

pca=PCA(n_components=2)#4 dimesion to 2 dimension "PCA 알고리즘아, 데이터를 2개 특성(차원)으로 압축할 준비를 해줘
x_reduced=pca.fit_transform(X)#x축을 찾고  축을 기준으로 2차원 변환해줘


print("--- [차원 축소 결과] ---")
print(f"원래 데이터 형태: {X.shape} (150개 데이터, 4개 특성)")
print(f"축소된 데이터 형태: {X_reduced.shape} (150개 데이터, 2개 특성)")
print("\n압축된 첫 번째 데이터 예시:", X_reduced[0])
print(f"정보 보존율(설명된 분산 비율): {sum(pca.explained_variance_ratio_) * 100:.2f}%")