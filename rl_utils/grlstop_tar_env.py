import numpy as np
import gymnasium as gym
from gymnasium import spaces


# reward functions to handle different powers m, and n
def cumulative_reward(x, T, B, m, n):
    if x <= T:
        return (x / T) ** m
    else:
        return ((B - x) / (B - T)) ** n

def stepwise_reward(x, T, B, m, n):
    if x <= T:
        return (1 / T ** m) * (x ** m - (x - 1) ** m)
    else:
        return (1 / (B - T) ** n) * ((B - x) ** n - (B - (x - 1)) ** n)
		
		


class TAREnv(gym.Env):



    def __init__(self, target_recall = None, topics_list = None, topic_id= None, size=100 , render_mode=None):



        self.size = size  # The size of the ranking relv vector

        #observation is 1D np array size array of relv vector
        self.observation_space = spaces.Box(-1,  1, shape=(size+2,), dtype=np.float32) #with clf

        # 2 actions, corresponding to "next", "stop"
        self.action_space = spaces.Discrete(2)

        # Set up some properties
        self.done = False
        self.reward = 0
        self.hit, self.miss = 0, 0



        # Set up the TAR
        self.vector_size = size



        # current position and stop position
        self._agent_location = 0 # -1: target,
        self._target_location = -1 #dummy value

        # keep predicted recall so far
        self.recall = 0
        self.target_recall = target_recall

        # topic data

        self.topics_list = topics_list
        self.topic_id = topic_id # for single env

        self.windows = 0
        self.window_size = 0

        #for vec env
        if topic_id is None:
          # checking whether the generated random number is not repeated
          while ( len(SELECTED_TOPICS) <= len(topics_list)):
            t = random.choice(topics_list)
            if t not in SELECTED_TOPICS:
              # include target recall with topic
              if TRAINING:
                # appending the random number to the resultant list, if the condition is true
                SELECTED_TOPICS.append(t)
                self.topic_id = t.split('_',1)[0]
                self.target_recall = float(t.split('_',1)[1])
              #if testing no need for target recall with topic
              else:
                # appending the random number to the resultant list, if the condition is true
                SELECTED_TOPICS.append(t)
                self.topic_id = t
                self.target_recall = target_recall
              break


          #ACTIVATE on TRAINING
          # use same ordered list of topics across diffreent runs
          if TRAINING:
            global SELECTED_TOPICS_ORDERERD_INDEX
            t = SELECTED_TOPICS_ORDERERD[SELECTED_TOPICS_ORDERERD_INDEX]
            self.topic_id = t.split('_',1)[0]
            self.target_recall = float(t.split('_',1)[1])
            SELECTED_TOPICS_ORDERERD_INDEX += 1
        else:
           self.topic_id = topic_id # for single env



        self.n_docs = 0
        self.rel_cnt = []
        self.rel_rate = []
        self.n_samp_docs = 0
        self.n_samp_docs_after_target =  0
        self.n_docs_wins = []
        self.rel_list = 0
        self.text_list = ''
        self.tfidf_list = []
        self.all_vectors = []
        self.all_vectors_target = []
        self.all_vectors_prediction = []

        # Define constants for clearer code
        self.NEXT = 0
        self.STOP = 1


        self.load_data_flag = True
        self.load_data(self.topic_id)

        self.first_step_flag = True

        #initialize environment each time for each topic
        self.reset()


    def load_data(self, topic_id):

      # load data only once when self._agent_location ==0
      if self._agent_location == 0 :

        all_vectors = [[-1]*self.vector_size for i in range(self.vector_size)]
        all_vectors_target = [[-1]*(self.vector_size+2) for i in range((self.vector_size))] # +2 for current index &target, DONT ADD IT FOR ROWS ONLY COLS
        all_vectors_prediction = [[-1]*(self.vector_size+2) for i in range((self.vector_size))] # +2 for current index &target, DONT ADD IT FOR ROWS ONLY COLS

        topic_id = self.topic_id


        n_docs = len(doc_rank_dic[topic_id])  # total n. docs in topic
        rel_list = rank_rel_dic[topic_id]  # list binary rel of ranked docs
        text_list = rank_text_dic[topic_id]  # list text feature of ranked docs
        tfidf_list = rank_tfidf_dic[topic_id]  # list text feature of ranked docs


        # get batches
        windows = make_windows(self.vector_size, n_docs)

        window_size = windows[0][1]

        # calculate batches
        rel_cnt,rel_rate, n_docs_wins = get_rel_cnt_rate(windows, window_size, rel_list)


        self.n_docs = n_docs
        self.rel_cnt = rel_cnt
        self.rel_rate = rel_rate
        self.n_docs_wins = n_docs_wins
        self.rel_list = rel_list
        self.text_list = text_list
        self.tfidf_list = tfidf_list
        self.windows = windows
        self.window_size = window_size

        #update all vector with all possible examined states
        for i in range(self.vector_size):
          all_vectors[i][0:i+1] = rel_rate[0:i+1] # update examined part

          all_vectors_target[i][-1] = self.target_recall # include target recall as last element
          all_vectors_target[i][-2] = i # mark current examined index

          #if clf not used
          all_vectors_target[i][0:i+1] = rel_rate[0:i+1] # update examined part only

          #if clf used
          if self.topic_id not in SELECTED_TOPICS_WITHOUT_TARGET:
            #run clf only once
            all_vectors_target[i][0:-2] = self.get_clf_predictions(i) # update examined with tl & non-examined part with clf predictions
          else:
            saved_all_vectors_target = ALL_VECTORS_PREDICTIONS_DIC[self.topic_id]
            all_vectors_target[i][0:-2] = saved_all_vectors_target[i][0:-2]

          #calculate target recall stopping pos
          #mark only 1st recall achieved stopping position
          if (sum(self.rel_cnt[0:i+1]) / sum(self.rel_cnt)) >= self.target_recall and self._target_location == -1 and i < self.vector_size: #7-10-24: i< self.vector_size
            self._target_location = i


        #update after passing through all batches
        SELECTED_TOPICS_WITHOUT_TARGET.append(self.topic_id)
        ALL_VECTORS_PREDICTIONS_DIC[self.topic_id] = all_vectors_target

        self.all_vectors = all_vectors
        self.all_vectors_target = all_vectors_target

    def get_clf_predictions(self,i):

      # Initialise count of documents in sample
      tmp_n_samp_docs = int(np.sum(self.n_docs_wins[0:i+1]))

      clf_name = 'LR-TFIDF'
      dataset_imbalance_handle = 'cost_sensitive_manual'
      acc, f1, predictions = run_classification_model_tfidf(self.topic_id, tmp_n_samp_docs, self.n_docs, self.rel_list, self.tfidf_list, clf_name, dataset_imbalance_handle)

      rel_pred_list = list(self.rel_list[0:tmp_n_samp_docs+1])+list(predictions)
      rel_cnt_wins, rel_pred_wins, n_docs_wins  = get_rel_cnt_rate(self.windows, self.window_size, rel_pred_list)

      return rel_pred_wins # update examined&non-examined part with tl & clf predictions

#######################################################################################


    def _get_obs(self):
        return  np.array(self.all_vectors_target[self._agent_location], dtype=np.float32)




    def _get_info(self):



        return {
                "topic_id": self.topic_id,
                "recall": round((self.recall),3),
                "cost": round(((self._agent_location +1 )/100),3), # each vec pos == 1% of collection +1 bc 1st loc [0] is 1% cost
                "e_cost": (round((((self._agent_location)-(self._target_location))/100),3)), # CostDiff
                "distance": (self._agent_location - self._target_location),
                "agent": (self._agent_location),
                "target": (self._target_location),
                "agent_vector": np.array(self.all_vectors_target[self._agent_location]),
                "terminal_observation": np.array(self.all_vectors_target[self._target_location])} # target_vector named terminal_observation needed for SB3 vec_env
#######################################################################################


    def reset(self,seed=0):

        # re-load data 1st time for vec_env
        if self.load_data_flag:
          self.load_data(self.topic_id)
          self.load_data_flag = False

        self._agent_location = 0
        self.n_samp_docs =  sum(self.n_docs_wins[0:self._agent_location+1])
        self.n_samp_docs_after_target =  sum(self.n_docs_wins[self._target_location:self._agent_location+1])
        self.recall = sum(self.rel_cnt[0:self._agent_location+1]) / sum(self.rel_cnt)

        state = self.all_vectors[self._agent_location]

        observation = self._get_obs()
        info = self._get_info()

        self.reward = 0

        #return state
        return observation, info

#######################################################################################



    def step(self, action):
        truncated = False
        terminated = False

        if self._agent_location >= self.vector_size-1:
          self.done = True
          truncated = True
          self.reward = 0

        if self._agent_location >= self.vector_size-2 and action == self.NEXT:
          self.done = True
          truncated = True
          self.reward = 0

        if action == self.STOP:
          terminated = True
          self.reward = 0


        if action == self.NEXT:
            if self.first_step_flag:
              self._agent_location = self._agent_location # dont move next, examine 1st portion at pos [0]
              self.first_step_flag = False
            else:
              self._agent_location += 1 # move to next portion (examined)

            self.n_samp_docs =  sum(self.n_docs_wins[0:self._agent_location+1])
            self.n_samp_docs_after_target =  sum(self.n_docs_wins[self._target_location:self._agent_location+1])
            self.recall = sum(self.rel_cnt[0:self._agent_location+1]) / sum(self.rel_cnt)

        observation = self._get_obs()
        info = self._get_info()

        # to get more easy readable formula
        reward_target_loc = self._target_location+1
        reward_agent_loc = self._agent_location+1
        reward_vector_size = self.vector_size


        global m,n #reusable for linear vs nonlinear
        self.reward = stepwise_reward(reward_agent_loc, reward_target_loc, reward_vector_size, m, n)


        return observation, self.reward, terminated, truncated, info




    def close(self):
        # we dont need close
        return




#early stopping
class EarlyStoppingCallback(BaseCallback):
    def __init__(self, patience: int, min_delta: float = 0.001, verbose: int = 0):
        super(EarlyStoppingCallback, self).__init__(verbose)
        self.patience = patience
        self.min_delta = min_delta
        self.best_mean_reward = -np.inf
        self.no_improvement_steps = 0

    def _on_step(self) -> bool:
        # check if an episode has ended
        if 'episode' in self.locals['infos'][-1]:
            # get latest episode reward mean
            mean_reward = np.mean(self.locals['infos'][-1]['episode']['r'])

            # check if the mean reward has improved
            if mean_reward > self.best_mean_reward + self.min_delta:
                # improvement, update the best reward
                self.best_mean_reward = mean_reward
                self.no_improvement_steps = 0
            else:
                # no improvement
                self.no_improvement_steps += 1

            # if no improvement for `self.patience` steps, stop training
            if self.no_improvement_steps > self.patience:
                if self.verbose > 0:
                    print("Early stopping triggered: Training stopped at step {}".format(self.num_timesteps))
                return False  # stop the training

        return True  # continue training



