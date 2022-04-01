'''
This script permits with the other ones in the folder to simulate an Erwan Renaudo's 
experiment : 
"Renaudo, E. (2016). Des comportements flexibles aux comportements habituels: 
Meta-apprentissage neuro-inspiré pour la robotique autonome (Doctoral dissertation, 
Université Pierre et Marie Curie (Paris 6))". 

With this script, the simulated agent use a q-learning algorithm (model-free behavior) 
to learn the task.
'''

from utility import *
from collections import deque
import numpy as np
import json
import datetime

VERSION = 1


class QLearningReplayBudget:
	"""
    This class implements a MODEL-FREE MODULE, ie learning algorithm (q-learning-replay-budget).
    """

	def __init__(self, experiment, map_file, initial_variables, action_space, boundaries_exp, parameters, options_log):
		"""
		Iinitialise values and models
		"""

		# self.epsilon = 0.2
		self.num_states = 38

		self.action_space = action_space
		self.new_deltas = np.zeros((self.action_space, self.num_states))

		# --- KEEP TRACK OF REPLAY CYCLES ---
		self.replay_cycles = 0
		self.replay_time = 0

		# --- REPLAY VALIDITY TIME ---
		self.time_window = 100

		self.experiment = experiment
		self.max_reward = boundaries_exp["max_reward"]
		self.duration = boundaries_exp["duration"]
		self.window_size = boundaries_exp["window_size"]
		self.alpha = parameters["alpha"]
		self.gamma = parameters["gamma"]
		self.beta = parameters["beta"]
		self.log = options_log["log"]
		self.summary = options_log["summary"]
		self.epsilon = 0.1

		action_count = initial_variables["action_count"]
		decided_action = initial_variables["decided_action"]
		previous_state = initial_variables["previous_state"]
		current_state = initial_variables["current_state"]
		self.init_qvalue = initial_variables["qvalue"]
		init_reward = initial_variables["reward"]
		init_delta = initial_variables["delta"]
		init_plan_time = initial_variables["plan_time"]
		init_actions_prob = initial_variables["actions_prob"]  # at first, uniform action prob
		self.not_learn = False
		
		# // List and dicts for store data //
		# Create a dict that contains the qvalues of the expert
		self.dict_qvalues = dict()
		# Create a dict that contains the probability of actions for each states
		self.dict_actions_prob = dict()
		self.dict_actions_prob["actioncount"] = action_count
		self.dict_actions_prob["values"] = dict()
		# Create a dict that contains the probability of actions for each states
		self.dict_decision = dict()
		self.dict_decision["actioncount"] = action_count
		self.dict_decision["values"] = dict()
		# Create a dict that contains the time of planification for each states
		self.dict_duration = dict()
		self.dict_duration["actioncount"] = action_count
		self.dict_duration["values"] = dict()
		
		# Load the transition model which will be used as map
		with open(map_file,'r') as file2:
			self.map = json.load(file2)
		# For each state of the map : 
		for state in self.map["transitionActions"]:
			s = str(state["state"])
			t = state["transitions"]
			
			self.dict_qvalues[(s,"qvals")] = [self.init_qvalue]*8
			self.dict_qvalues[(s,"visits")] = 0
			
			# - initialise the probabilties of actions
			self.dict_actions_prob["values"][s]={ "actions_prob": [init_actions_prob]*8, "filtered_prob": [init_actions_prob]*8}

			# - initialise the duration dict
			self.dict_duration["values"][s]={ "duration": 0.0}

		# Experience replays: we have to keep previously encountered transitions in buffer
		self.replay_buffer = deque()  # here will be put the samples
		self.replay_budget = parameters['replay_budget']

		self.base_it_replay = 100
		self.tau = np.power(self.epsilon / self.base_it_replay, 1 / self.base_it_replay)

		self.num_states = 38

		self.new_deltas = np.zeros((self.action_space, self.num_states))



	def __del__(self):
		"""
		Close all log files
		"""

	def get_actions_prob(self, current_state):
		"""
		Get the probabilities of actions of the current state
		"""
		
		return get_filtered_prob(self.dict_actions_prob, current_state)
		

	def get_plan_time(self, current_state):
		"""
		Get the time of planification for the current state
		"""
		
		return get_duration(self.dict_duration, current_state)


	def decide(self, current_state, qvalues):
		"""
		Choose the next action using soft-max policy
		"""
		
		actions = dict()
		qvals = dict()
		
		for a in range(0,8):
			actions[str(a)] = a
			qvals[str(a)] = qvalues[a]
		
		# Soft-max function
		actions_prob = softmax_actions_prob(qvals, self.beta)
		new_probs = list()
		for action, prob in actions_prob.items():
			new_probs.append(prob)
		set_actions_prob(self.dict_actions_prob, current_state, new_probs)
		
		# For each action, sum the probabilitie of selection with a low pass filter
		old_probs = get_filtered_prob(self.dict_actions_prob, current_state)
		filtered_actions_prob = list()
		for a in range(0,len(new_probs)):
			filtered_actions_prob.append(low_pass_filter(self.alpha, old_probs[a], new_probs[a]))
		set_filtered_prob(self.dict_actions_prob, current_state, filtered_actions_prob)
		
		# The end of the soft-max function
		decision, choosen_action = softmax_decision(actions_prob, actions)
		
		return choosen_action, actions_prob
		

	def infer(self, current_state):
		"""
		In the MF expert, the process of inference consists to read the q-values table.
		(this process is useless. It's to be symetric with MB expert)
		"""
		
		return self.dict_qvalues[(str(current_state),"qvals")]
		


	def learn(self, previous_state, action, current_state, reward_obtained):
		"""
		Update q-values using Q-learning
		"""
		
		# Compute the deltaQ to send at the MC (criterion for the trade-off)
		qvalue_previous_state = self.dict_qvalues[str(previous_state),"qvals"][int(action)]
		qvalues_current_state = self.dict_qvalues[str(current_state),"qvals"]
		max_qvalues_current_state = max(qvalues_current_state)
		
		# Compute q-value
		new_RPE = reward_obtained + self.gamma * max_qvalues_current_state - qvalue_previous_state
		new_qvalue = qvalue_previous_state + self.alpha * new_RPE
		self.dict_qvalues[str(previous_state),"qvals"][int(action)] = new_qvalue

		return new_qvalue, qvalue_previous_state

	

	def replayExperience(self):
		"""
		This is the 'experience replay' part.
		"""
		# Keep track of replay time values
		start = datetime.datetime.now()
		cycle = 0
		ci = []

		deltas = np.zeros((self.action_space, self.num_states))
		for act in range(self.action_space):
			for state in range(self.num_states):
				deltas[act][state] = 1 - self.dict_qvalues[str(state), "qvals"][int(act)]

		activated = np.zeros((self.action_space, self.num_states))

		# 2) Do as many replays as possible
		# for i in range(1, self.replay_budget+1): # as there is always at least one experiment
		while True:
			if len(self.replay_buffer) == 0 :
				self.replay_time = (datetime.datetime.now() - start).total_seconds()
				self.replay_cycles = cycle
				break

			cycle +=1

			rand = random.randrange(0, len(self.replay_buffer)) # Take randomly an experience
			replay_previous_state = self.replay_buffer[rand][0]
			replay_action = self.replay_buffer[rand][1]
			replay_reward_obtained = self.replay_buffer[rand][2]
			replay_current_state = self.replay_buffer[rand][3]
			new_qvalue, qvalue_previous_state = self.learn(replay_previous_state, replay_action, replay_current_state,
														   replay_reward_obtained)

			self.new_deltas[replay_action][int(replay_previous_state)] = np.abs(new_qvalue - qvalue_previous_state)

			if activated[replay_action][int(replay_previous_state)] == 0:
				activated[replay_action][int(replay_previous_state)] = self.base_it_replay

			activated = activated * self.tau
		
			# mask = activated + np.ones((self.action_space,  self.num_states)) * np.sign(activated)
			# mask = activated * np.sign(activated)
			crit = np.sum(np.multiply(self.new_deltas, activated))

			# crit = np.max(self.new_deltas)
			ci.append(crit)

			#if cycle >= self.replay_budget or crit < self.epsilon:
				#break
			# if crit < self.epsilon or cycle >= self.replay_budget:
			if crit < self.epsilon or cycle >= self.replay_budget:
				break

		self.replay_time = (datetime.datetime.now() - start).total_seconds()
		self.replay_cycles = cycle
		#if self.replay_cycles > 1:
		# print(f"crit: {ci[-1]}")
		# print(f"replay_cycles: {self.replay_cycles}")
		# print(f"replay_cycles: {self.replay_cycles}")

	def run(self, action_count, cumulated_reward, reward_obtained, previous_state, decided_action, current_state, do_we_plan): 
		"""
		Run the model-free system
		"""

		# Update the actioncount and the number of the visits for the previous state
		self.dict_duration["actioncount"] = action_count
		self.dict_actions_prob["actioncount"] = action_count
		self.dict_qvalues[(str(current_state),"actioncount")] = action_count
		self.dict_qvalues[(str(previous_state),"visits")] += 1
		
		if self.not_learn == False:
			# ---- Update the q-values of the previous state using Q-learning ---
			self.new_qvalue = self.learn(previous_state, decided_action, current_state, reward_obtained)

			# --- Update replay buffer ONLY IF NOT SELF.NOT_LEARN ---
			# 1) Update buffer with new experience
			self.replay_buffer.append([previous_state, decided_action, reward_obtained, current_state])  # Buffer updated

			# We also make sure to avoid memory saturation (and privilege the most recent experiences)
			while len(self.replay_buffer) > self.time_window:
				self.replay_buffer.popleft() # FIFO

		# If the expert was choosen to plan, compute the news probabilities of actions
		if do_we_plan:
			# ---------- INFER (nothing in MF mode) -----------
			
			old_time = datetime.datetime.now()
			
			# Run the process of inference
			qvalues = self.infer(current_state)
			

			# ---------- REPLAYS ---------------------------------
			# Do some experience replay
			self.replayExperience()

			# Sum the duration of planning process with a low pass filter
			current_time = datetime.datetime.now()
			new_plan_time = (current_time - old_time).total_seconds()
			old_plan_time = get_duration(self.dict_duration, current_state)
			filtered_time = low_pass_filter(self.alpha, old_plan_time, new_plan_time)
			set_duration(self.dict_duration, current_state, filtered_time)
		else:
			qvalues = self.dict_qvalues[(str(current_state),"qvals")]
			
		decided_action, actions_prob = self.decide(current_state, qvalues)
		
		# Maj the history of the decisions
		set_history_decision(self.dict_decision, current_state, decided_action, self.window_size)
		prefered_action = [0]*8
		for action in range(0,len(prefered_action)):
			for dictStateValues in self.dict_decision["values"]:
				if dictStateValues["state"] == current_state:
					prefered_action[action] = sum(dictStateValues["history_decisions"][action])
		
		# Prepare data to return 
		plan_time = get_duration(self.dict_duration, current_state)
		selection_prob = get_filtered_prob(self.dict_actions_prob, current_state)
		
		if reward_obtained > 0.0:
			self.not_learn = True

			# self.replayExperience()

			for a in range(0,8):
				self.dict_qvalues[(current_state,"qvals")] = [0.0]*8
			# reset where reward is found ( as no anymore move )
		else:
			self.not_learn = False

		return decided_action
		



