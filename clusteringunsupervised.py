from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
import pandas as pd

# 1. 데이터 로드 (정답 y는 사용하지 않고 특성 X만 사용합니다)
iris = load_iris()
X = iris.data

# 2. K-Means 모델 생성 (3개 그룹으로 묶어줘!)
# n_clusters=3 : 몇 개의 그룹으로 나눌지 지정합니다.
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
kmeans.fit(X) # 정답(y) 없이 X만 학습!

# 3. 각 데이터가 몇 번 그룹(0, 1, 2)으로 분류되었는지 확인
labels = kmeans.labels_

# 보기 좋게 DataFrame으로 정리
df = pd.DataFrame(X, columns=iris.feature_names)
df['Cluster'] = labels
df['Actual_Species'] = iris.target # 실제 정답과 비교용 (학습엔 안 썼음)

print("--- [군집화 결과 샘플] ---")
print(df.sample(7, random_state=42))