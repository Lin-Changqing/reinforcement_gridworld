import numpy as np

# 全局变量
"""
设置棋盘大小,左上角坐标(0,0).行数列数可调,目标点可调
"""
BOARD_ROWS = 3 # 行数
BOARD_COLS = 4 # 列数
WIN_STATE = (0, 3) # 目标点(成功)坐标
LOSE_STATE = (1, 3) # 目标点(失败)坐标
START = (2, 0) # 起点坐标
DETERMINISTIC = True # 做动作一定有对应状态转移

class State:
    """
    board 存储每一个位置的状态,0表示没有障碍物可以通过,-1表示障碍,无法通过.
    state 表示当前位置,用二元组表示
    isEnd 表示是否到达最终位置
    determine 表示环境模型是否是 deterministic
    """
    def __init__(self,state=START) -> None:
        self.board = np.zeros([BOARD_ROWS, BOARD_COLS]) # 棋盘大小用numpy存储,0表示可以通行
        self.board[1, 1] = -1 # -1表示障碍
        self.state = state
        self.isEnd = False
        self.determine = DETERMINISTIC

    def giveReward(self,st):
        """
        此处的reward获取只与转移后状态,也就是s'有关,后续可以修改成与(s,a)相关
        """
        if st == WIN_STATE:
            return 1 # WIN_STATE 奖励为1
        elif st == LOSE_STATE:
            return -1 # LOSE_STATE 奖励为-1
        else:
            return 0 # 其余无奖励
        
    def isEndFunc(self):
        """
        判断是否到达终止状态
        """
        if (self.state == WIN_STATE) or (self.state == LOSE_STATE):
            self.isEnd = True

    def nxtPosition(self,state,action):
        """
        根据action和space(上下左右)更换位置,返回下一个位置
        模型是否deterministic对这个函数有影响
        """
        if self.determine:
            if action == "up":
                nxtState = (state[0] - 1, state[1])
            elif action == "down":
                nxtState = (state[0] + 1, state[1])
            elif action == "left":
                nxtState = (state[0], state[1] - 1)
            elif action == "right":
                nxtState = (state[0],state[1] + 1)
            else:
                nxtState = (state[0], state[1])
            # if next state legal
            if (nxtState[0] >= 0) and (nxtState[0] <= (BOARD_ROWS -1)):
                if (nxtState[1] >= 0) and (nxtState[1] <= (BOARD_COLS -1)):
                    # 如果nxtState不为障碍,则返回
                    x,y = nxtState
                    if self.board[x,y]!=-1:
                        return nxtState
            return state
        
    def showBoard(self):
        self.board[self.state] = 1
        for i in range(0, BOARD_ROWS):
            print('-----------------')
            out = '| '
            for j in range(0, BOARD_COLS):
                if self.board[i, j] == 1:
                    token = '*'
                if self.board[i, j] == -1:
                    token = 'z'
                if self.board[i, j] == 0:
                    token = '0'
                out += token + ' | '
            print(out)
        print('-----------------')

class Agent:
    """
    数据成员:
    states:存放一整条状态变化历史
    actions:动作空间
    State:上述状态,以及gridworld相关参数
    lr:
    exp_rate: discounted rate
    policy: 一大张表格,里面有对应每个位置每个动作的概率
    state_values: 一张表格,里面存放每个位置的state value
    action_value: 一张表格,里面存放每个位置每个动作的value
    """
    def __init__(self):
        self.states = []
        self.actions = ["up", "down", "left", "right", "none"]
        self.State = State()
        self.lr = 0.2
        self.exp_rate = 0.3
        self.policy = np.full((BOARD_ROWS,BOARD_COLS,len(self.actions)),1.0/len(self.actions)) # 初始policy随机
        self.action_values = np.zeros((BOARD_ROWS,BOARD_COLS,len(self.actions)))
        self.state_values = np.zeros((BOARD_ROWS, BOARD_COLS))

    def chooseAction(self):
        """
        对应书中policy,根据policy,state选出对应动作.
        """
        x,y = self.State.state # 获取当前坐标
        probabilities = self.policy[x][y]
        action = np.random.choice(self.actions,p=probabilities)
        return action

    def takeAction(self, action):
        position = self.State.nxtPosition(action)
        return State(state=position)

    def reset(self):
        self.states = []
        self.State = State()

    def showValues(self):
        for i in range(0, BOARD_ROWS):
            print('----------------------------------')
            out = '| '
            for j in range(0, BOARD_COLS):
                out += "{:.2f}".format(self.state_values[(i, j)]).ljust(6) + ' | '
            print(out)
        print('----------------------------------')

    def showPolicy(self):
        """
        打印policy,每个位置概率最大的被打印出来,障碍的就用x表示
        """
        for i in range(0, BOARD_ROWS):
            print('-----------------')
            out = '| '
            for j in range(0, BOARD_COLS):
                token=""
                # 选出概率最大的
                possibilities = self.policy[i][j]
                action = self.actions[np.argmax(possibilities)]
                if action == 'up':
                    token = '↑'
                elif action == 'down':
                    token = '↓'
                elif action == 'left':
                    token = '←'
                elif action == 'right':
                    token = '→'   
                else:
                    token = '⭕'            
                if self.State.board[i][j] == -1:
                    token = '■'
                elif (i,j) == WIN_STATE:
                    token = '★'
                elif (i,j) == LOSE_STATE:
                    token = 'X'
                out += token + ' | '
            print(out)
        print('-----------------')

    def valueIteration(self):
        # for every state
        while(True):
            previous_state_values = self.state_values.copy()
            for i in range(0, BOARD_ROWS):
                for j in range(0, BOARD_COLS):
                    # calculate action values
                    # for every action
                    for k in range(len(self.actions)):
                        nxtState = self.State.nxtPosition((i,j),self.actions[k]) # 获取下一个状态,根据这个算奖励
                        self.action_values[i][j][k] = self.State.giveReward(nxtState) + self.state_values[nxtState]*self.exp_rate
                    # 选择最大的action value
                    best_action = np.argmax(self.action_values[i,j])
                    # 更新policy,只有最大的概率为1,其余为0
                    self.policy[i,j] = np.zeros(len(self.actions))
                    self.policy[i,j,best_action] = 1.0
                    # 更新state value
                    self.state_values[i,j] = self.action_values[i,j,best_action]
            # 如果state value不再变化,则停止
            if np.sum(np.abs(previous_state_values - self.state_values)) < 1e-6:
                break





if __name__ == "__main__":
    a = Agent()
    a.valueIteration()
    a.showValues()
    a.showPolicy()
