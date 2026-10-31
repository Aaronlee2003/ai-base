import numpy as np

# 1. 환경 및 MDP 매개변수 설정
GRID_SIZE = (3, 3)
ACTIONS = {'Up': (-1, 0), 'Down': (1, 0), 'Left': (0, -1), 'Right': (0, 1)}
ACTION_NAMES = list(ACTIONS.keys())

# 보상 마스크 정의
REWARDS = np.full(GRID_SIZE, -0.04)
REWARDS[2, 2] = 1.0   # 목표 지점 (Goal)
REWARDS[1, 1] = -1.0  # 위험 지점 (Trap)

TERMINAL_STATES = [(2, 2), (1, 1)]

GAMMA = 0.99         # 할인율 (Discount factor)
THETA = 0.0001       # 수렴 조건 임계값

def get_next_state(state,action):
    r,c = state
    dr,dc = ACTIONS[action]
    next_r,next_c = r+dr, c+dc
    if 0 <= next_r < GRID_SIZE[0] and 0 <= next_c < GRID_SIZE[1]:
        return (next_r,next_c)
    return state  # 경계 밖으로 나가면 현재 상태 유지
def value_iteration():
    v=np.zeros(GRID_SIZE)
    while True:
        delta=0
        new_v=np.copy(v)
        for r in range(GRID_SIZE[0]):
            for c in range(GRID_SIZE[1]):
                state=(r,c)
            if state in TERMINAL_STATES:
                    new_v[state] = REWARDS[state]
                    continue
            q_values = []
            for action in ACTIONS:
                 next_s=get_next_state(state,action)
                 q_value=REWARDS[state]+GAMMA*v[next_s]
                 q_values.append(q_value)
                 max_v = max(q_values)#most valuable action
                 delta = max(delta, abs(max_v - v[state]))#if differece is small we choose big delta  this data cant converge to optimal value
                 new_v[state] = max_v
                 v=new_v
            if delta < THETA:
             break
        policy = np.empty(GRID_SIZE, dtype=object)
        for r in range(GRID_SIZE[0]):
            for c in range(GRID_SIZE[1]):
               state = (r, c)
            if state in TERMINAL_STATES:
                policy[state] = '  Goal  ' if state == (2, 2) else '  Trap  '
                continue
                
            best_action = None
            best_value = float('-inf')
            for action in ACTIONS:
                next_s = get_next_state(state, action)
                q_value = REWARDS[state] + GAMMA * v[next_s]
                if q_value > best_value:
                    best_value = q_value
                    best_action = action
            policy[state] = best_action
        return v, policy

                 

