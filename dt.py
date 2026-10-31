import math
from collections import Counter
from unittest import result


def calculate_entropy(y):
    total_samples = len(y)
    if total_samples == 0:
        return 0.0
    count=Counter(y)
    entropy = 0.0   
    for count in count.values():
        p=count/total_samples
        entropy -= p * math.log2(p)
    return entropy
    
def calculate_information_gain(X, y, feature_index):
    #before split
    paretn_entropy = calculate_entropy(y)
    #by certain feature data split
    feature_valeus=[sameple[feature_index] for sample in X]
    unique_values=set(feature_valeus)
    
    weighted_entropy=0.0
    total_samples=len(y)
    
    for value in unique_values:
        subset_y=[y[i] for i in range(total_samples) if X[i][feature_index]==value]
        subset_entropy=calculate_entropy(subset_y)
        probability=len(subset_y)/total_samples
        #applied weighted entropy formula  because we need to consider the proportion of samples in each subset when calculating the overall entropy after the split. This ensures that the information gain reflects the actual distribution of the data across the different feature values.
        #shoud applied sample proportion can get mora accurative weighted entropy
        weighted_entropy+=probability*subset_entropy
        #erase certain feature data split
    return paretn_entropy-weighted_entropy
class Node:
    def __init__(self, feature=None, children=None, result=None, value=None):
        self.feature = feature  # Index of the feature to split on certeria of branching
        self.children=children or {}     # Threshold value for the split
        self.result=result                # Right child node
        self.value = value                    # Class label for leaf nodes
        
class DecisionTree:
    def __init__(self):
        self.root = None
    def fit(self, X, y):
        self.root = self._build_tree(X, y)
    
    def _build_tree(self, X, y,allowed_features=None,paretn_value=None):
        #just one optionn
        if len(set(y))==1:
            return Node(result=y[0],value=paretn_value)
        
        #no more features to split on
        if not X or not X[0] and allowed_features is not None:
            # get most frequnt
            most_common_value=Counter(y).most_common(1)[0][0]
            return Node(result=most_common_value,value=paretn_value)
        #fill all features to split on
        #check if allowed_features is None, is root node, if not None, is child node
        if allowed_features is None:
            #how many features in X, get all features index except used features
            allowed_features = list(range(len(X[0])))
        besst_gain = -1
        best_feature = None
        for feature_index in allowed_features:  
            gain = calculate_information_gain(X, y, feature_index)
            if gain > besst_gain:
                besst_gain = gain
                best_feature = feature_index
        #certaian and complexity any improvement, same pareent entropy sanme count, no more gain, stop split,prevent overfitting, same graph height
        if best_feature <=0:
            most_common = Counter(y).most_common(1)[0][0]
            return Node(results=most_common, value=paretn_value)
        node=Node(feature=best_feature, value=paretn_value)
        feature_values=[sample[best_feature] for sample in X]
        unique_values=set(feature_values)
        
        # used feature except next split, prevent overfitting, same graph height
        remaining_features=[f for f in allowed_features if f!=best_feature]
        feature_values = [sample[best_feature] for sample in X] 
        unique_values = set(feature_values)
        for val in unique_values:
    # 1. 조건에 맞는 실제 데이터 subset 분리 why use this way now stage have to make real decisiotn branch  and child node get complete data to make decision,so child can  split so can implement recursive call
          subset_X = [X[i] for i in range(len(X)) if X[i][best_feature] == val]
          subset_y = [y[i] for i in range(len(y)) if X[i][best_feature] == val]
       
    # 2. 자식 트리 생성 (재귀 호출)
        child_node = self._build_tree(subset_X, subset_y, remaining_features, parent_value=val)
    
    # 3. 현재 노드의 자식 딕셔너리에 연결
        node.children[val] = child_node
    def predict_sample(self, node,sample):
        if node.result is not None:
            return node.result
        #check samnple value in child node
        val = sample[node.feature]
        #특성값을 검사하고 해당 값에 대한 자식 노드가 존재하면 재귀적으로 예측을 수행합니다. 만약 자식 노드가 존재하지 않으면 None을 반환합니다.
        if val in node.children:
            return self.predict_sample(node.children[val], sample)                           
        else:
            return None

    def predict(self, X):
        return [self.predict_sample(self.root, sample) for sample in X]
            
        
            
        
    