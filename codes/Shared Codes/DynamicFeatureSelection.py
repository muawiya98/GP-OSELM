from sklearn.metrics import recall_score, precision_score, f1_score, roc_auc_score,accuracy_score
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler 
from sklearn.tree import DecisionTreeClassifier
from collections import deque, defaultdict
from warnings import filterwarnings
from xgboost import XGBClassifier
from scipy.special import softmax
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
from keras import backend as K
from typing import Callable 
import tensorflow as tf
from sklearn import svm
from typing import List
import seaborn as sns
import pandas as pd 
import numpy as np 
import requests
import random
import keras 
import copy
import json
import os
sns.set(rc = {'figure.figsize':(22,12)}, style="whitegrid")



def epsilon_greedy(expected_reward, epsilon=0.97) -> int:
    """
    expected_reward: list of expected rewards for each possible action
    epsilon: .
    """
    if np.random.rand() <= epsilon:
        return np.random.choice(list(range(len(expected_reward))))
    else:
        return np.argmax(expected_reward)

PolicyFunction  = Callable[[np.ndarray, float], int]




LEARNING_RATE = 0.001

def create_model(input_dim):
    K.clear_session()
    model = keras.models.Sequential()
    model.add(keras.layers.Input(shape=(input_dim,)))
    model.add(keras.layers.Dense(32, kernel_initializer='he_uniform', activation='relu'))
    model.add(keras.layers.Dense(16, kernel_initializer='he_uniform', activation='relu'))
    model.add(keras.layers.Dense(2))
    model.compile(loss='mse', optimizer='adam')
    return model



class Agents:
    def __init__(self, evaluation_network,number_of_featuer, buffer_size: int = 800):

        self.evaluation_network = evaluation_network
        self.target_network = copy.deepcopy(self.evaluation_network)
        self.buffer_size = buffer_size
        self.fitted = False
        self.number_of_featuer = number_of_featuer
        # reply buffer is a list of tuples each tuples contains the following
        # (St, At, St+1, Rt+1)
        # (Current state, Action was made, New state, Reward)
        self.reply_buffer = deque(maxlen=self.buffer_size)
        self.contrbution = np.random.rand()

    def make_action(self, curr_state: np.ndarray, policy_function: PolicyFunction, epsilon) -> int:
        # q_values represents the expected rewards for each possible action
        if self.fitted:
            q_values = self.evaluation_network.predict(curr_state.reshape(-1, self.number_of_featuer),verbose=0)
            action = policy_function(q_values, epsilon)
        else:
            action = policy_function([0, 1], 1)
        return action
    
    def update_target_network(self):
        self.target_network = copy.deepcopy(self.evaluation_network)
        return
        
        
class AgentsSoftmax(Agents):
    # class variable
    agent_count = 0
    def __init__(self, evaluation_network,number_of_featuer, buffer_size=800):
        self.agent_id = AgentsSoftmax.agent_count
        AgentsSoftmax.agent_count += 1
        super().__init__(evaluation_network,number_of_featuer, buffer_size)


    def update_evaluation_network(self, batch_size=32, epochs=5, discount_factor=0.995):
        # select random batch from the reply buffer
        batch = random.sample(self.reply_buffer, batch_size)

        # inintilize some lists to store transition information
        Q1, actions, Q2, rewards = [], [], [], []

        # from each transition extract its values
        for transition in batch:
            Q1.append(transition[0])
            actions.append(transition[1])
            Q2.append(transition[2])
            rewards.append(transition[3])
        # X_train will be the states from
        X_train = np.array(Q1).reshape(-1, self.number_of_featuer)

        expected_reward = self.evaluation_network.predict(np.array(Q1).reshape(-1, self.number_of_featuer),verbose=0)
        Q2 = self.target_network.predict(np.array(Q2).reshape(-1, self.number_of_featuer),verbose=0)

        # update expected rewards using biliman equation

        for i, act in enumerate(actions[:-1]):
            expected_reward[i, act] = rewards[i] + (discount_factor * np.argmax(Q2[i]))

        y_train = expected_reward.copy()

        # calculate the change frequency of the agent decision to use it as its contrbution in get total reward

        change_frequency = 0
        for state, next_state, reward, next_reward in zip(X_train[:-1], X_train[1:], rewards[: -1], rewards[1:]):
            #print(state, next_state, reward, next_reward)
            print()
            if np.abs(state[self.agent_id] - next_state[self.agent_id]) == 1:
                self.contrbution += np.abs(reward - next_reward)
                change_frequency += 1
        self.contrbution /= change_frequency

        # train the DQN evaluation network.
        self.evaluation_network.fit(X_train, y_train, epochs=epochs, verbose=0)
        self.fitted = True
        return        
        
        
class AgentsRegression(Agents):
    # class variable
    agent_count = 0
    def __init__(self, evaluation_network,number_of_featuer, buffer_size=800):
        self.agent_id = AgentsRegression.agent_count
        AgentsRegression.agent_count += 1
        super().__init__(evaluation_network,number_of_featuer, buffer_size)
   
    def update_evaluation_network(self, batch_size=32, epochs=5, discount_factor=0.995):
        # select random batch from the reply buffer
        batch = random.sample(self.reply_buffer, batch_size)

        # inintilize some lists to store transition information
        Q1, actions, Q2, rewards = [], [], [], []

        # from each transition extract its values
        for transition in batch:
            Q1.append(transition[0])
            actions.append(transition[1])
            Q2.append(transition[2])
            rewards.append(transition[3])
        # X_train will be the states from
        X_train = np.array(Q1).reshape(-1, self.number_of_featuer)

        expected_reward = self.evaluation_network.predict(np.array(Q1).reshape(-1, self.number_of_featuer),verbose=0)
        Q2 = self.target_network.predict(np.array(Q2).reshape(-1, self.number_of_featuer),verbose=0)

        # update expected rewards using biliman equation
        for i, act in enumerate(actions[:-1]):
            expected_reward[i, act] = rewards[i] + (discount_factor * np.argmax(Q2[i]))
            
        y_train = expected_reward.copy()
        # train the DQN evaluation network.
        self.evaluation_network.fit(X_train, y_train, epochs=epochs, verbose=0)
        self.fitted = True
        return


class AgentsAverage(Agents):
    # class variable
    agent_count = 0
    def __init__(self, evaluation_network,number_of_featuer, buffer_size=800):
        self.agent_id = AgentsAverage.agent_count
        AgentsAverage.agent_count += 1
        super().__init__(evaluation_network,number_of_featuer, buffer_size)

    def update_evaluation_network(self, batch_size=32, epochs=5, discount_factor=0.995):
        # select random batch from the reply buffer
        batch = random.sample(self.reply_buffer, batch_size)

        # inintilize some lists to store transition information 
        Q1, actions, Q2, rewards = [], [], [], []

        # from each transition extract its values 
        for transition in batch:
            Q1.append(transition[0])
            actions.append(transition[1])
            Q2.append(transition[2])
            rewards.append(transition[3])
        # X_train will be the states from
        X_train = np.array(Q1).reshape(-1, self.number_of_featuer)

        expected_reward = self.evaluation_network.predict(np.array(Q1).reshape(-1, self.number_of_featuer),verbose=0)
        Q2 = self.target_network.predict(np.array(Q2).reshape(-1, self.number_of_featuer),verbose=0)

        # update expected rewards using biliman equation

        for i, act in enumerate(actions[:-1]):
            expected_reward[i, act] = rewards[i] + (discount_factor * np.argmax(Q2[i]))

        y_train = expected_reward.copy()

        WINDOW_SIZE = 4
        X_train_ = np.zeros((X_train.shape[0] // WINDOW_SIZE, X_train.shape[1]))
        y_train_ = []
        j = 0
        for i in range(0, batch_size, WINDOW_SIZE):
            window_of_states = X_train[i: i + WINDOW_SIZE].sum(axis=0) / WINDOW_SIZE
            window_of_rewards = sum(rewards[i: i + WINDOW_SIZE])
            r = window_of_rewards * window_of_states[self.agent_id]
             # Rounding state 
            X_train_[j, :] = np.around(window_of_states)
            if window_of_states[self.agent_id] == 0:
                if window_of_rewards > 0.6:
                    r = window_of_rewards
                else:
                    r = window_of_rewards / WINDOW_SIZE
            y_train_.append(r)
            j += 1

        X_train = X_train_
        y_train = np.array(y_train_)
        # train the DQN evaluation network.
        self.evaluation_network.fit(X_train, y_train, epochs=epochs, verbose=0)
        self.fitted = True
        return


class AgentsSingle(Agents):
    # class variable
    agent_count = 0
    def __init__(self, evaluation_network,number_of_featuer, buffer_size=800):
        self.agent_id = AgentsSingle.agent_count
        AgentsSingle.agent_count += 1
        super().__init__(evaluation_network,number_of_featuer, buffer_size)

    def update_evaluation_network(self, batch_size=32, epochs=5, discount_factor=0.995):
        # select random batch from the reply buffer
        batch = random.sample(self.reply_buffer, batch_size)

        # inintilize some lists to store transition information
        Q1, actions, Q2, rewards = [], [], [], []

        # from each transition extract its values
        for transition in batch:
            Q1.append(transition[0])
            actions.append(transition[1])
            Q2.append(transition[2])
            rewards.append(transition[3])
        # X_train will be the states from
        X_train = np.array(Q1).reshape(-1, self.number_of_featuer)

        expected_reward = self.evaluation_network.predict(np.array(Q1).reshape(-1, self.number_of_featuer),verbose=0)
        Q2 = self.target_network.predict(np.array(Q2).reshape(-1, self.number_of_featuer),verbose=0)

        # update expected rewards using biliman equation
        for i, act in enumerate(actions[:-1]):
            expected_reward[i, act] = rewards[i] + (discount_factor * np.argmax(Q2[i]))
            
        y_train = expected_reward.copy()
        # train the DQN evaluation network.
        self.evaluation_network.fit(X_train, y_train, epochs=epochs, verbose=0)
        self.fitted = True
        return


# test set percentage 
TESTSIZE=0.2

def get_reward(X,Y ,subset_features):
    global TESTSIZE
    # index of selected features
    subset_features = np.where(np.array(subset_features) == 1)[0]
    if subset_features.shape[0] == 0:return 0
    # train test split 
    X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state=42, test_size=TESTSIZE)
    # data normalization.
    std = StandardScaler()
    X_train = std.fit_transform(X_train)
    X_test = std.transform(X_test)
    # classification and model evaluation 
    classifier = LogisticRegression(max_iter=500)
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    return acc


def reward_strategy(time_step: int, accuracy: float, accuracy_history: list, subset_features: list, error_rate: float,
                    beta: float = 0.99):
    if sum(subset_features) == len(subset_features):
        return -5
    elif accuracy > max(accuracy_history):
        return 0.5
    elif accuracy < max(accuracy_history):
        return -0.1
    else:
        return -1 * (beta * error_rate + ((1 - beta) * (sum(subset_features) / len(subset_features))))




import pickle 

class Results:
    def __init__(self, method_name, dataset_name, chunk_id, feature_space):
        self.dataset_name = dataset_name
        self.method_name = method_name
        self.chunk_id = chunk_id
        self.feature_space = feature_space
        self.feature_space_size = len(feature_space)
        self.result_information = {}
    
    def set_chunk_id(self, chunk_id: int):
        self.chunk_id = chunk_id
    
    def set_feature_space(self, feature_space: list):
        self.feature_space = feature_space
        self.feature_space_size = len(feature_space)
    
    def add_result(self, model_type:str, result:dict):
        self.result_information[model_type] = result
    
    def save(self, path='feature_selection_results'):
        file_name = self.method_name + '_' + self.dataset_name + '_' + '{}'.format(self.chunk_id) + '.pkl'
        with open(os.path.join(path, file_name), 'wb') as file_:
            pickle.dump(self, file_, pickle.HIGHEST_PROTOCOL)
        return     



class Telegram:

    def __init__(self, bot_token):
        self.end_point = 'https://api.telegram.org/bot'
        self.token = bot_token
        self.full_endpoint = self.end_point + self.token + '/'

    def __repr__(self):
        return 'your token is {}'.format(self.full_endpoint)

    def send_message(self, chat_id, message):
        send_text = self.full_endpoint + 'sendMessage'
        data = {'chat_id': chat_id, 'text': message}
        response = requests.get(send_text, data=data)
        return response

    def send_photo(self, chat_id, photo):
        url = self.full_endpoint + 'sendPhoto'
        data = {'chat_id': chat_id}
        files = {'photo': open(photo, 'rb')}
        response = requests.post(url, data=data, files=files)
        return response

    def send_document(self, chat_id, file):
        url = self.full_endpoint + 'sendDocument'
        data = {'chat_id': chat_id}
        files = {'document':open(file, 'rb')}
        response = requests.post(url, data = data, files = files)
        return response
    def get_updates(self):
        url = self.full_endpoint + 'getUpdates'
        response = requests.get(url)
        return response
    
    def get_file_information(self, file_id):
        url = f'https://api.telegram.org/bot{self.token}/getFile'
        response = requests.post(url,data = {"file_id":file_id})
        if response.status_code != 200:
            return {"ok":"False"}
        json_response = response.json()
        if json_response['ok'] == False:
            return {"ok":"False"}
        file_path = json_response['result']['file_path']
        file_information = requests.get(f'https://api.telegram.org/file/bot{self.token}/{file_path}')
        return file_information.text



def send_results(telegram_api, results):
    telegram.send_message(1021388563, 'dataset name : {}'.format(results.dataset_name))
    telegram.send_message(1021388563, 'chunk id : {}'.format(results.chunk_id))
    telegram.send_message(1021388563, 'selected features : {}'.format(results.feature_space))
    telegram.send_message(1021388563, 'results')
    for key in  results.result_information.keys():
        telegram.send_message(1021388563, 'model tpye : {}'.format(key))
        telegram.send_message(1021388563, '{}'.format(results.result_information[key]))



def softmax_distrbution(agents):
    contrbutions = []
    for agent in agents:
        contrbutions.append(agent.contrbution)
    return softmax(contrbutions)


def random_forest_distrbution(X_train,Y_train,num_of_agents, num_of_samples=10000):
    X = []
    y = []
    for i in range(num_of_samples):
        features_space = np.random.choice([0, 1], size=(num_of_agents,)).tolist()
        accuracy = get_reward(X_train,Y_train, features_space)
        X.append(features_space)
        y.append(accuracy)
    X = np.array(X)
    y = np.array(y)
    rf = RandomForestRegressor(n_estimators=15)
    rf.fit(X, y)
    return rf.feature_importances_.tolist()


def feature_selection(algo_type, agents, X, Y, eposide=200):
    """
    """
    # column_names = list(range(dataset.shape[1]))
    # column_names[-1] = 'class'
    # dataset.columns = column_names
    epsilon = 0.01
    features_space = []
    NUMBER_OF_AGENTS = X.shape[1]
    
    # get contrbutions 
    contrbutions = []
    if algo_type == 'random_forest':
        contrbutions = random_forest_distrbution(X,Y,NUMBER_OF_AGENTS)
    elif algo_type in ['single_agent', 'average']:
        contrbutions = [1] * NUMBER_OF_AGENTS
    
    for i in range(eposide):
        # define the initial space 
        features_space = np.random.choice([0, 1], size=(NUMBER_OF_AGENTS,)).tolist()
        # rewards history 
        rewards = [0]

        # get action of each agent to create new feature space 
        next_feature_space = []
        # contrbution of each agent 
        
        if algo_type == 'softmax':
            contrbutions = softmax_distrbution(agents)
            
        for t in range(0, NUMBER_OF_AGENTS):
            action = agents[t].make_action(np.array(features_space.copy()), epsilon_greedy, epsilon)
            next_feature_space.append(action)
            if algo_type == 'single_agent':
                features_space[t] = action

        
        # calculate the total accuracy of new state (new feature space) and distrbute it using softmax
        
        # 1- get the accuracy using machine learning model trained in the current subset feature 
        reward_as_accuracy = get_reward(X,Y, next_feature_space)
        
        # 2- using the reward strategy map the accuracy value (reward_as_accuracy) to new reward value (reward_at_time_t)
        reward_at_time_t = reward_strategy(t, reward_as_accuracy, rewards, next_feature_space, 1 - reward_as_accuracy)
        # add the accuray of machine learning model to rewards list to use it in the mapping reward strategy.
        rewards.append(reward_as_accuracy)
        
        # total reward = reward after mapping 
        total_reward = reward_at_time_t
        

        
        # add state and actions to agent buffer reply  and the reward which equals to contrbution of the agent*total reward
        transition = []
        for t in range(0, NUMBER_OF_AGENTS):
            transition.clear()
            # add current state (current feature space )
            feature_space_copy = features_space.copy() 
            transition.append(feature_space_copy)
            # add agent's action to the transition
            action = next_feature_space[t]
            transition.append(action)
            # add new state (new feature space) into transition
            transition.append(next_feature_space)
            # add distrbuted reward to the transition
            transition.append(total_reward * contrbutions[t])
            # add the transition to reply buffer 
            agents[t].reply_buffer.append(transition)
            if len(agents[t].reply_buffer) > 32 and i % 32 == 0:
                agents[t].update_evaluation_network()
        if i % 64 == 0:
            for agent in agents:
                if agent.fitted:
                    agent.update_target_network()
        epsilon = 0.97 * epsilon
    return next_feature_space


def model_evaluation(chunk_X, chunk_Y, selected_features):
    models = [
              ('SVM', svm.SVC(kernel='rbf', max_iter=8000, C=0.2, probability=True)),
              ('KNN', KNeighborsClassifier(5)),
              ('DecisionTree', DecisionTreeClassifier(random_state=42)),
              ('RandomForest', RandomForestClassifier()),
              ('LogRegression', LogisticRegression(max_iter=500))
     ]
    # index of selected features
    subset_features = np.where(np.array(selected_features) == 1)[0]
    if subset_features.shape[0] == 0:return 0
    # train test split 
    X, Y = chunk_X, chunk_Y
    X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state=42, test_size=TESTSIZE)
    # data normalization.
    std = StandardScaler()
    X_train = std.fit_transform(X_train)
    X_test = std.transform(X_test)
    # classification and model evaluation 
    res = []
    for model in models:
        model_name, model_obj = model[0], model[1]
        model_obj.fit(X_train, y_train)
        y_pred = model_obj.predict(X_test)
        f1 = f1_score(y_test, y_pred)
        res.append(f1)
    return np.mean(res)


def dynamic_feature_selection(chunk_X, chunk_Y,algorithm_type=['softmax','average','single_agent','random_forest']):  
  AgentsSoftmax.agent_count,AgentsAverage.agent_count,AgentsSingle.agent_count,AgentsRegression.agent_count = 0,0,0,0
  softmax_agents,average_agents,single_agent_agents,random_forest_agents,result=[],[],[],[],[]
  NUM_OF_FEATURES = chunk_X.shape[1] 
  for i in range(NUM_OF_FEATURES):
    if 'softmax' in algorithm_type:
      softmax_agents.append(AgentsSoftmax(create_model(NUM_OF_FEATURES),NUM_OF_FEATURES))
    if 'average' in algorithm_type:
      average_agents.append(AgentsAverage(create_model(NUM_OF_FEATURES),NUM_OF_FEATURES))  
    if 'single_agent' in algorithm_type:
      single_agent_agents.append(AgentsSingle(create_model(NUM_OF_FEATURES),NUM_OF_FEATURES))
    if 'random_forest' in algorithm_type:
      random_forest_agents.append(AgentsRegression(create_model(NUM_OF_FEATURES),NUM_OF_FEATURES))
  if 'softmax' in algorithm_type: 
    softmax_result = feature_selection('softmax', softmax_agents,chunk_X, chunk_Y)
    sr=model_evaluation(chunk_X=chunk_X,chunk_Y=chunk_Y,selected_features=softmax_result)
  if 'average' in algorithm_type:
    average_result = feature_selection('average', average_agents,chunk_X, chunk_Y)
    ar=model_evaluation(chunk_X=chunk_X,chunk_Y=chunk_Y,selected_features=average_result)
  if 'single_agent' in algorithm_type:
    single_agent_result = feature_selection('single_agent', single_agent_agents,chunk_X, chunk_Y)
    sar=model_evaluation(chunk_X=chunk_X,chunk_Y=chunk_Y,selected_features=single_agent_result)
  if 'random_forest' in algorithm_type:
    random_forest_result = feature_selection('random_forest', random_forest_agents,chunk_X, chunk_Y)
    rfr=model_evaluation(chunk_X=chunk_X,chunk_Y=chunk_Y,selected_features=random_forest_result)
  
  for softmax,average,single,random in zip(softmax_result,average_result,single_agent_result,random_forest_result):
    sum_votes = sum([softmax,average,single,random])
    if sum_votes > (len(algorithm_type) // 2):result.append(1)
    elif sum_votes == (len(algorithm_type) // 2):
      rand = np.random.uniform(low=0,high=1)
      if rand >0.5:result.append(1)
      else:result.append(0)
    else:result.append(0)
  
  
  vr = model_evaluation(chunk_X, chunk_Y, result)
  over_all = [sr,ar,sar,rfr,vr]
  re_all = [softmax_result,average_result,single_agent_result,random_forest_result,result]
  
  print(re_all[over_all.index(max(over_all))])
  return re_all[over_all.index(max(over_all))]
























