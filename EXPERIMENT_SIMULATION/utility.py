'''
This script contains some functions used by the Model Based and Model Free agents, and the metaController scripts. The names of the functions
are explicit : 
set_* = edit the contents of dictionary
compute_* = do a computation on an element of a data structure and return the result
get_* = collect a value in a data structure
'''

import random
import numpy as np

INF = 1000000000000000000000000

def rargmax(vector):
    """ Argmax that chooses randomly among eligible maximum indices. """
    m = np.amax(vector)
    indices = np.nonzero(vector == m)[0]
    return random.choice(indices)


def softmax_actions_prob(qvalues, beta):
	"""
	This version will shrink or augment the value of beta in case the returned action probabilities are problematic (all 0 or a NAN value)
	:param qvalues: the agent's q values
	:param beta: the initial agent's delta
	:return: the action probabilities
	"""
	value_NAN, values_zero = True, True # This helps removing infinite values (or every proba to 0)
	while value_NAN or values_zero :
		value_NAN, values_zero, actions_prob = softmax_actions_prob_beta(qvalues, beta)
		if value_NAN: # beta too high
			beta = 2*beta/3 # TODO: see if there are other more efficient ways to shrink beta... or turn it into a parameter?
		elif values_zero: # beta too small
			beta = 5*beta/4
	return actions_prob # No NAN values

def softmax_actions_prob_beta(qvalues, beta):
	actions_prob = dict()
	sum_probs = 0
	# -----------------------------------------------------------------------
	for key, value in qvalues.items():
		# -------------------------------------------------------------------
		actions_prob[str(key)] = np.exp(value*beta) # /!\ If value*beta is too big (for highest qvalues), inf !!! (especially with many replays ... ) ---> Leads to NAN later....
		# -------------------------------------------------------------------
		sum_probs += actions_prob[str(key)]
	# -----------------------------------------------------------------------
	values_zero = True # check if
	values_NAN = False # check if
	for key, value in qvalues.items():
		actions_prob[str(key)] = actions_prob[str(key)]/sum_probs
		if np.isnan(actions_prob[str(key)]): # Too big beta detection!
			values_NAN = True
		if actions_prob[str(key)] != 0: # Too small beta detection!
			values_zero = False
	# -----------------------------------------------------------------------
	return values_NAN, values_zero, actions_prob
# ---------------------------------------------------------------------------




# ---------------------------------------------------------------------------
def softmax_decision(actions_prob, actions):
	cum_actions_prob = list()
	previous_value = 0
	# -----------------------------------------------------------------------
	for key, value in actions_prob.items():
		cum_actions_prob.append([key,value + previous_value])
		previous_value = cum_actions_prob[-1][1]
	# -----------------------------------------------------------------------
	randval = np.random.rand()
	decision = dict()
	# -----------------------------------------------------------------------
	for key_value in cum_actions_prob:
		if randval < key_value[1]:
			decision = key_value[0]
			action = actions[key_value[0]]
			break
	return decision, action
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
def set_visit(dict_qvalues, state):# ??
	dict_qvalues["values"][state]["visits"]+= 1


def get_visit(dict_qvalues, state):
	visit = dict_qvalues["values"][state]["visits"]
	return visit


# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
def set_deltaQ(dict_qvalues, state, deltaQ):
	for dictStateValues in dict_qvalues["values"]:
		if dictStateValues["state"] == state:
			dictStateValues["deltaQ"] = deltaQ
			break

def get_deltaQ(dict_qvalues, state):
	for dictStateValues in dict_qvalues["values"]:
		if dictStateValues["state"] == state:
			return dictStateValues["deltaQ"]
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
def set_RPE(dict_qvalues, state, RPE):
	for dictStateValues in dict_qvalues["values"]:
		if dictStateValues["state"] == state:
			dictStateValues["RPE"] = RPE
			break

def get_RPE(dict_qvalues, state):
	for dictStateValues in dict_qvalues["values"]:
		if dictStateValues["state"] == state:
			return dictStateValues["RPE"]
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
def set_delta_prob(dict_delta_prob, state, delta_prob):
	dict_delta_prob["values"]["state"]["delta_prob"]= delta_prob

def get_delta_prob(dict_delta_prob, state):
	return dict_delta_prob["values"]["state"]["delta_prob"]


# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
def set_actions_prob(dict_probs, state, list_probs):
	#change probas of actions for the state current state
	dict_probs["values"][state]["actions_prob"]=list_probs

def set_filtered_prob(dict_probs, state, list_probs):
	#print('oui',dict_probs["values"])
	dict_probs["values"][state]["filtered_prob"] = list_probs
	

def get_filtered_prob(dict_probs, state):

	list_probs= dict_probs["values"][state]["filtered_prob"]
			
	return list_probs

# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
def set_decided_action(dict_decision, state, list_actions):
	for dictStateValues in dict_decision["values"]:
		if dictStateValues["state"] == state:
			dictStateValues["history_decisions"] = list_actions
			break

def get_decided_action(dict_decision, state):
	for dictStateValues in dict_decision["values"]:
		if dictStateValues["state"] == state:
			list_actions = dictStateValues["history_decisions"]
			return list_actions

def set_history_decision(dict_decision, state, decided_action, window_size):
	# [0,1,0,1,0,0,0] means that, for a state s, the action a was used two steps ago, and also 4 steps ago, but not the
	# other step times in this state, as other actions were therefore used instead.
	# 'steps' taken into account are only steps when the agent is in state s, of course.
	for dictStateValues in dict_decision["values"]:
		if dictStateValues == state:
			for action in range(0, len(dict_decision["values"][dictStateValues]["history_decisions"])):
				if action == decided_action:
					value_to_add = 1
				else:
					value_to_add = 0
				pointer = window_size-1
				while pointer >= 0:
					if pointer == 0:
						dict_decision["values"][dictStateValues]["history_decisions"][action][pointer] = value_to_add
					# The last done action, stored in history
					else:
						dict_decision["values"][dictStateValues]["history_decisions"][action][pointer] =\
							dict_decision["values"][dictStateValues]["history_decisions"][action][pointer-1]
					# history shift : [0,1,0] --> [0,0,1] to indicate that the action was used 2 steps before instead
					# of 1
					pointer = pointer - 1
			break
			
# ---------------------------------------------------------------------------
def set_duration(dict_duration, state, duration):
	
	dict_duration["values"][state]["duration"]=duration
	

def get_duration(dict_duration, state):
	
	duration=dict_duration["values"][state]["duration"]
	return duration

# ---------------------------------------------------------------------------
def set_qval(dict_qvalues, state, action, value):
	for dictStateValues in dict_qvalues["values"]:
		if dictStateValues["state"] == state:
			for dictActionValue in dictStateValues["values"]:
				if dictActionValue["action"] == action:
					dictActionValue["value"] = value
					break
			break

def get_qval(dict_qvalues, state, action):
	for dictStateValues in dict_qvalues["values"]:
		if dictStateValues["state"] == state:
			for dictActionValue in dictStateValues["values"]:
				if dictActionValue["action"] == action:
					qvalue = dictActionValue["value"]
					return qvalue

def get_qvals(dict_qvalues, state):
	qvalues = dict()
	for dictStateValues in dict_qvalues["values"]:
		if dictStateValues["state"] == state:
			for dictActionValue in dictStateValues["values"]:
				action = dictActionValue["action"]
				value = dictActionValue["value"]
				qvalues[action] = value
			return qvalues

def get_vval(dict_qvalues, state):
	for dictStateValues in dict_qvalues["values"]:
		if dictStateValues["state"] == state:
			vVal = 0.0
			for dictActionValue in dictStateValues["values"]:
				qvalue = dictActionValue["value"]
				if qvalue > vVal:
					vVal = qvalue
			return vVal
# ---------------------------------------------------------------------------
def get_reward(dict_rewards, state, action):
	reward = dict_rewards["transitionActions"][state][action]
	return reward

def set_reward(dict_rewards, state, action, reward):
	dict_rewards["transitionActions"][state][action]= reward
	
	
def initialize_rewards(dict_rewards, state, ACTIONSPACE): # provides mean reward, for a state, for each action
	dict_rewards["transitionActions"][state]=[0]*ACTIONSPACE
	for a in range(0,ACTIONSPACE):
		dict_rewards["transitionActions"][state][a]= 0.0 # donne un valeur de reward 

# ---------------------------------------------------------------------------

					
def get_transition_prob(dict_transitions, start, action, arrival):#modified_fct
	
	return dict_transitions["transitionActions"][start]["transition"][action][arrival]["prob"]



			
def get_transition_probs(dict_transitions, start, action):#modified_fct

	dictProbs = dict()
	
	for arrival in dict_transitions["transitionActions"][start]["transition"][action]: # en comprehension
		
		dictProbs[arrival] = dict_transitions["transitionActions"][start]["transition"][action][arrival]["prob"]
	
	return dictProbs
	

def set_transition_prob(dict_transitions, start, action, arrival, prob):#modified_fct
	
	dict_transitions["transitionActions"][start]["transition"][action][arrival]["prob"]=prob
	
# ---------------------------------------------------------------------------



def initialize_transition(dict_transitions, start, action, arrival, prob, window_size):#modified_fct

	dict_transitions["transitionActions"][start]={"transition":{action:{arrival:{"prob":prob,"window": [1]+(window_size-1)*[0]}}}}



def add_transition(dict_transitions, start, action, arrival, prob, window_size):#modified_fct
	
	
	if action not in dict_transitions["transitionActions"][start]["transition"]:
		dict_transitions["transitionActions"][start]["transition"][action]=dict()

	if arrival not in dict_transitions["transitionActions"][start]["transition"][action]:
		dict_transitions["transitionActions"][start]["transition"][action][arrival]=dict()
		
	dict_transitions["transitionActions"][start]["transition"][action][arrival]["prob"]=prob
	dict_transitions["transitionActions"][start]["transition"][action][arrival]["window"]=[1]+(window_size-1)*[0]
	



def get_number_transitions(dict_transitions, start, action):#modified_fct
	

	dictProbs = dict()
	for arrival in dict_transitions["transitionActions"][start]["transition"][action]: # en comprehension
		dictProbs[arrival]=  sum(dict_transitions["transitionActions"][start]["transition"][action][arrival]["window"])
	return dictProbs



			
def set_transitions_window(dict_transitions, start, action, arrival, window_size):#modified_fct
	
	for arrivalit in dict_transitions["transitionActions"][start]["transition"][action]:
		
		if arrivalit == arrival:
			dict_transitions["transitionActions"][start]["transition"][action][arrivalit]["window"]=[1]+dict_transitions["transitionActions"][start]["transition"][action][arrivalit]["window"][:- 1]
		else:		
			dict_transitions["transitionActions"][start]["transition"][action][arrivalit]["window"]=[0]+dict_transitions["transitionActions"][start]["transition"][action][arrivalit]["window"][:- 1]
			
				
				
			
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
def low_pass_filter(alpha, old_value, new_value):
	"""
	Apply a low-pass filter
	"""
	# ---------------------------------------------------------------------------
	filtered_value = (1.0 - alpha) * old_value + alpha * new_value
	return filtered_value
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
def shanon_entropy(source_vector):
	entropy = 0
	for element in source_vector:
		if element == 0:
			element = 0.0000001
		entropy += element*np.log2(element)
	return -(entropy)
# ---------------------------------------------------------------------------


#----------------------------------------------------------------------------
def check_buffer(this_state, action,replay_buffer):
	"""
	:param this_state:
	:param action:
	:param replay_buffer:
	:return: The index (if present) of the transition (this_state, action), and a boolean to indicate it's presence.
	"""
	index = None
	check = False
	if len(replay_buffer) > 0:
		itemindex1 = np.where(replay_buffer[0] == str(this_state))
		itemindex2 = np.where(replay_buffer[1] == str(action))
		candidate = np.intersect1d(itemindex1, itemindex2)

		if len(candidate) != 0 :
			check = True
			index = candidate[0]

	return index, check


# ------------- Priority queue implementation --------------------------------------------------------------------------
import bisect

class PriorityQueue:
	def __init__(self):
		self._q = []
		self.keys = []

	def add(self, value, priority=0):
		bisect.insort(self._q, [priority]+ value) # sorts according to priority

	def add2(self, value, priority = 0):
		"""
		Same aim as add, but add provokes bugs for dqn algorithms.
		This second version might be better for some modules.
		Therefore, it's up to the module whether *add* or *add2* is used.
		:param value:
		:param priority:
		:return:
		"""

		index = bisect.bisect(self.keys, priority)
		self._q.insert(index, [priority] + value)
		self.keys.insert(index, priority)
		#bisect.insort(self._q, (priority, value))

	def pop(self):
		if len(self.keys) > 0:  # Means that add2 was used
			self.keys.pop()
		return self._q.pop()

	def clean(self, max_size = 1000):
		del self._q[:max(0,len(self) - max_size)]
		if len(self.keys) > 0: #Means that add2 was used
			del self.keys[:max(0, len(self.keys) - max_size)]



	def sample(self, batch_size):
		"""
		Samples the experiences with the highest probabilities
		/!\ Should be used with add2 function (as keys also are updated)
		:param batch_size:
		:return:
		"""
		batch = self._q[- min(len(self), batch_size):]
		del self._q[- min(len(self), batch_size):]
		del self.keys[- min(len(self.keys), batch_size):]
		return batch

	def sampleRandom(self, batch_size, beta):
		"""
		Samples according to priority proba.
		Returns also importance-sampling weight wj (see "Double DQN with proportional prioritization",  for details)
		This version does not delete the samples!
		:param batch_size:
		:return:
		"""
		batch = []
		weights = []
		batch_index  = []
		if len(self.keys) > 0:
			indexes = range(len(self.keys))
			eps = 1e-5 #To avoid having some probabilities equal to 0
			probas = (np.array(self.keys) + eps) / sum(np.array(self.keys) + eps)
			batch_index = np.random.choice(indexes, min(batch_size, len(self)), replace= False, p = probas)
			batch_index[::-1].sort() # important to sort batch list if removal is done just after sampling (errors of indexes if not inverse of sort)
			divider = (min(probas)*len(batch_index))**(-beta) # to normalize weights
			for index in batch_index :
				batch.append(self._q[index])
				weights.append(((probas[index]*len(batch_index))**(-beta) )/ divider)
		return  batch, np.array(weights), batch_index

	def __len__(self):
		return len(self._q)


