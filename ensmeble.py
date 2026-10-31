import numpy as np

#여러 모델의 예측값 중 가장 많이 나온 클래스(최빈값)를 최종 결정으로 내리는 방식입니다.
def hard_voting_predict(model_predictions):
    predictions_per_sample=model_predictions.T
    final_predictions = []

    for sample in predictions_per_sample:
        # 각 샘플마다 가장 많이 등장한 클래스(최빈값)를 찾습니다.
        classes, counts = np.unique(sample, return_counts=True)
        best_class = classes[np.argmax(counts)]
        final_predictions.append(best_class)
        
    return np.array(final_predictions) 
test_preds = np.array([
    [0, 1, 1, 0, 1],
    [0, 0, 1, 1, 1],
    [1, 0, 1, 0, 0]
])

print("최종 투표 결과:", hard_voting_predict(test_preds))
# 결과 예상: [0, 0, 1, 0, 1] (각 열에서 다수결)