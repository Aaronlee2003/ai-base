import numpy as np



def create_bootstrap_samples(X, y, n_estimators):
    #x,y original data,n boostrap smaple  want to create n_estimators bootstrap samples`
    n_samples = X.shape[0]#row count of original data
    bootstrap= []
    
    for _ in range(n_estimators):
        # 중복 추출(replace=True)로 원본 데이터 크기만큼 인덱스를 뽑습니다.
        incices=np.random.choice(n_samples,size=n_samples,replace=True) #pick randomly  size=n_samples with replacement and return true 는 복원 추출
        oob_indices = np.array([i for i in range(n_samples) if i not in np.indices])#unpicked indices if incidies not picke 3  oodincieds append 3 testcase
        oob_indices = np.setdiff1d(np.arange(n_samples), np.indices)
        bootstrap.append((X[np.indices], y[np.indices], oob_indices))#pick index store as a tuple of X,y,oob_indices and append to bootstrap list
    return bootstrap

X_dummy = np.arange(10).reshape(-1, 1)
y_dummy = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])

samples = create_bootstrap_samples(X_dummy, y_dummy, n_estimators=2)

for idx, sample in enumerate(samples):
    print(f"\n[트리 {idx+1}번을 위한 데이터]")
    print("추출된 X:", sample["X_resampled"].flatten())
    print("선택받지 못한 OOB 인덱스:", sample["oob_indices"])
