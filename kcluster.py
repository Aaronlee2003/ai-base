import numpy as np

class KMeans:
    def __init__(self, k=3, max_iters=100, tol=1e-4):
        self.k = k                  # 군집 개수
        self.max_iters = max_iters  # 최대 반복 횟수
        self.tol = tol              # 수렴 판정 임계값 (이동 거리가 이 값보다 작으면 정지)
        self.centroids = None       # 각 군집의 중심점 좌표들

    def fit(self, X):
        """데이터 X (shape: n_samples, n_features)를 군집화합니다."""
        n_samples, n_features = X.shape#data and dimension

        # 1. 초기 중심점 설정: 데이터 포인트 중 무작위로 k개 선택
        random_indices = np.random.choice(n_samples, self.k, replace=False)
        self.centroids = X[random_indices].astype(float)

        for iteration in range(self.max_iters):
            # 2. 할당 단계: 모든 데이터 포인트에서 각 중심점까지의 거리 계산
            # 브로드캐스팅을 활용해 유클리드 거리(L2 Norm) 계산
            # X[:, np.newaxis] shape: (n_samples, 1, n_features) n,d matrix add 1 axis make  n,1,d matrix
            # self.centroids shape: (1, k, n_features)
            distances = np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=2)#throught numpy broad cast,we can calculate distance between each data and each centroid, find norm and we cnat get n,k matrix
            
            # 가장 가까운 중심점의 인덱스를 클러스터 라벨로 할당 (shape: n_samples,)
            labels = np.argmin(distances, axis=1)

            # 3. 업데이트 단계: 각 군집에 속한 데이터들의 평균 좌표로 중심점 갱신
            new_centroids = np.zeros_like(self.centroids)#새로 갱신될 좌표
            for cluster_idx in range(self.k):
                cluster_points = X[labels == cluster_idx]#현재 군집에 속한 데이터들만 추리는 것
                
                # 빈 클러스터가 생기는 예외 상황 처리
                if len(cluster_points) > 0:
                    new_centroids[cluster_idx] = cluster_points.mean(axis=0)#평균 좌표로 중심점 갱신
                else:
                    # 데이터가 하나도 안 잡히면 랜덤 데이터 하나로 재초기화
                    new_centroids[cluster_idx] = X[np.random.choice(n_samples)]

            # 4. 수렴 확인: 중심점들의 이동 거리가 tol 이하이면 조기 종료
            shift = np.linalg.norm(new_centroids - self.centroids)
            self.centroids = new_centroids
            
            if shift < self.tol:
                print(f"[{iteration + 1}번째 반복] 중심점이 수렴하여 조기 종료했습니다.")
                break

        return labels

    def predict(self, X):
        """새로운 데이터가 어느 군집에 속하는지 예측합니다."""
        distances = np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=2)
        return np.argmin(distances, axis=1)
    
# broad cast about it can expand the array same shape,if n=100,k=3,d=2,we can expand 100,1,2 to 100,3,2    1,3,2 to 100,3,2 