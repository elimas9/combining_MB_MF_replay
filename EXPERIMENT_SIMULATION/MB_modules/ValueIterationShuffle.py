'''
This script permits with the other ones in the folder to simulate an Erwan Renaudo's 
experiment : 
"Renaudo, E. (2016). Des comportements flexibles aux comportements habituels: 
Meta-apprentissage neuro-inspiré pour la robotique autonome (Doctoral dissertation, 
Université Pierre et Marie Curie (Paris 6))". 

With this script, the simulated agent use a value-iteration algorithm (model-based 
behavior) to learn the task.
'''

from utility import *
import numpy as np
import json
import datetime


VERSION = 1


class ValueIterationShuffle:
	"""
	This class implements a MODEL-BASED MODULE, ie learning algorithm (value-iteration-shuffle).
	In this version, the infer budget is per state-action update.
	The order of the states to loop in infer phase is random (shuffled at each loop)
    """

	def __init__(self, experiment, map_file, initial_variables, action_space, boundaries_exp, parameters, options_log):
		"""
		Initialise values and models
		"""
		
		# --- KEEP TRACK OF INFER & SIMULATION REPLAY CYCLES ---
		self.infer_cycles = 0 # keeps track of the number of random infer cycles
		self.infer_cycles_tot = 0
		# /!\ IN THIS MODULE, THE PARAMETERS BELOW FOR REPLAYS ARE USELESS, BUT STILL INITIALIZED FOR CONSISTENCY
		#Replays for the simulation replays of the prioritized sweeping phase:
		self.replay_cycles = 0 # keeps track of the number of cycles at each simulation replay session
		self.replay_time = 0 # keeps track of the replay time at each simulation replay session
		#Replays for the forward phase of the bidirectional algorithm :
		self.bi_replay_cycles = 0 # keeps track of the number of cycles at each forward simulation replay session
		self.bi_replay_time = 0 # keeps track of the replay time at each forward simulation replay session

		# --- VARIABLE FOR RANDOM STATE-ACTION SELECTION ---
		self.shuffled_list_states_actions = []

		# --- VARIABLE TO KEEP TRACK OF THE ESTIMATION ERROR ---
		self.deltas = [{} for a in range(action_space)] # Will keep track of the deltas (per state-action)

		self.experiment = experiment
		self.max_reward = boundaries_exp["max_reward"]
		self.duration = boundaries_exp["duration"]
		self.window_size = boundaries_exp["window_size"]
		self.epsilon = 0.01  # 0.001  # 0.01 # boundaries_exp["epsilon"]
		self.alpha = parameters["alpha"]
		self.gamma = parameters["gamma"]
		self.beta = parameters["beta"]
		self.log = options_log["log"]
		self.summary = options_log["summary"]
		action_count = initial_variables["action_count"]
		decided_action = initial_variables["decided_action"]
		previous_state = initial_variables["previous_state"]
		current_state = initial_variables["current_state"]
		self.init_qvalue = initial_variables["qvalue"]
		init_reward = initial_variables["reward"]
		init_delta = initial_variables["delta"]
		init_plan_time = initial_variables["plan_time"]
		self.action_space = action_space
		init_actions_prob = initial_variables["actions_prob"]
		self.not_learn = False
		
		# // List and dicts for store data //
		# Create the list of states
		self.list_states = list()
		# Create the list of known action
		self.list_actions = dict()
		# Create a dict that contains the qvalues of the expert
		self.dict_qvalues = dict()
		# Create the transitions dict according to the type of log used by Erwan
		self.dict_transitions = dict()
		self.dict_transitions["actioncount"] = action_count
		self.dict_transitions["transitionActions"] = dict() # modified_fct
		# Create the rewards dict according to the type of log used by Erwan

		self.dict_rewards = dict()
		self.dict_rewards["actioncount"] = action_count
		self.dict_rewards["transitionActions"] = dict()
		# Create a dict that contains the probability of actions for each states
		self.dict_actions_prob = dict()
		self.dict_actions_prob["actioncount"] = action_count
		self.dict_actions_prob["values"] = dict()
		# Create the dict of neighbour reward states
		self.dict_goals = dict()
		self.dict_goals["values"] = list()
		# Create a dict that contains the delta prob for each state
		self.dict_delta_prob = dict()
		self.dict_delta_prob["actioncount"] = action_count
		self.dict_delta_prob["values"] = dict()
		# Create a dict that contains the time of planification for each states
		self.dict_duration = dict()
		self.dict_duration["actioncount"] = action_count
		self.dict_duration["values"] = dict()

		self.num_states = 38

		self.new_deltas = np.zeros((self.action_space, self.num_states))

		self.base_it_replay = 500
		self.tau = np.power(self.epsilon / self.base_it_replay, 1 / self.base_it_replay)

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
			self.dict_actions_prob["values"][s]={ "actions_prob": [init_actions_prob]*8,
												  "filtered_prob": [init_actions_prob]*8}

			# - initialise the delta prob dict
			self.dict_delta_prob["values"][s]={ "delta_prob": init_delta}
			
			# - initialise the duration dict
			 
			self.dict_duration["values"][s]={"duration": 0.0} 


		# Initialise logs
		if self.log == True:
			self.directory_flag = False
			try:
				os.stat("logs")
			except:
				os.mkdir("logs") 
			os.chdir("logs")
			try:
				os.stat("MB")
			except:
				os.mkdir("MB") 
			os.chdir("MB")

			directory = "exp"+str(self.experiment)+"_gamma"+str(self.gamma)+"_beta"+str(self.beta)
			if not os.path.exists(directory):
				os.makedirs(directory)
			os.chdir(directory) 
			self.directory_flag = True
			
			prefixe = "v"+str(VERSION)+"_TBMB_exp"+str(self.experiment)+"_"
			
			self.reward_log = open(prefixe+'reward_log.dat', 'w')
			self.reward_log.write("timecount"+" "+str(action_count)+" "+str(init_reward)+" "+"currentTime-nodeStartTime"+" "+"currentTime"+"\n")
			
			self.states_evolution_log = open(prefixe+'statesEvolution_log.dat', 'w')
			self.states_evolution_log.write("timecount"+" "+str(action_count)+" "+current_state+" "+previous_state+" "+"currentContactState"+ \
				" "+"currentViewState"+" "+str(decided_action)+"currentTime-nodeStartTime"+" "+"currentTime"+"\n")

			self.actions_evolution_log = open(prefixe+'actions_evolution_log.dat', 'w')
			self.actions_evolution_log.write('{\n"logs" :\n['+json.dumps(self.dict_actions_prob))
			
			self.monitoring_values_log = open(prefixe+'monitoring_values_log.dat', 'w')
			self.monitoring_values_log.write(str(action_count)+" "+str(init_plan_time)+" "+str(init_delta)+" "+str(init_delta)+" "+str(init_delta)+"\n")
		
			os.chdir("../../../")
		


	def __del__(self):
		"""
		Close all log files
		"""
		
		if self.log == True:
			self.reward_log.close()
			self.actions_evolution_log.close()
			self.states_evolution_log.close()
			self.monitoring_values_log.close()
		


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
		
		# For each action, sum the probabilities of selection with a low pass filter
		old_probs = get_filtered_prob(self.dict_actions_prob, current_state)
		filtered_actions_prob = list()
		for a in range(0,len(new_probs)):
			filtered_actions_prob.append(low_pass_filter(self.alpha, old_probs[a], new_probs[a]))
		set_filtered_prob(self.dict_actions_prob, current_state, filtered_actions_prob)
		
		# The end of the soft-max function
		decision, choosen_action = softmax_decision(actions_prob, actions)
		
		return choosen_action, actions_prob
		


	def update_qvalue(self, this_state, this_action):#modified_fct
		"""
		Value iteration update, for just 1 state-action couple
		
		"""
		
		flag = False
		accu = 0.0
		reward = get_reward(self.dict_rewards, this_state, this_action)

		# loop througth the transitions
		
		for arrival in self.dict_transitions["transitionActions"][this_state]["transition"][this_action]:
	
			prob = self.dict_transitions["transitionActions"][this_state]["transition"][this_action][arrival]["prob"]
			vValue = max(self.dict_qvalues[(str(arrival), "qvals")])
			flag = True
			accu += (prob * (reward + self.gamma * vValue))
			
		if flag == True:
			self.dict_qvalues[(str(this_state),"qvals")][int(this_action)] = accu

	


#################################################################################
# _____________________ Functions for the  delta estimations ____________________

# -------- 1) ONLY for the initialisation
	def get_first_infer_deltas(self):
		"""
		Allows to get an estimate of the convergence indicator, without making any new update.
		Stacks all of the delta values (not absolute values) for each state, action couple.
		Should be used once at the beginning of each new step, only once at the beginning of self.infer.
		"""
		for this_state in self.list_states:
			# Get estimated delta of state:
			self.update_deltas_one_state(this_state)

	def update_deltas_one_state(self, this_state):
		"""
		This function is a helper function to update the list of deltas
		Similarly to update_qvalues, but it does not update at all the actual Q-value,
		 it is just used for convergence estimation, by updating self.deltas
		:param this_state: a state of the environment
		"""
		for action in self.list_actions[this_state]:  # LOOP ACTIONS
			self.deltas[action][this_state] = self.get_state_action_delta(this_state, action)


	def get_state_action_delta(self,this_state, action):
		"""
		This function is a helper function to update the list of deltas
		:param this_state: a state of the environment
		:param action: a possible action for the provided state
		:return: the estimation error
		"""
		action = int(action)
		
		previous_qvalue = self.dict_qvalues[(str(this_state), "qvals")][action]
		accu = 0.0
		reward = get_reward(self.dict_rewards, this_state, action)
		# loop througth the transitions a changer
		for arrival in self.dict_transitions["transitionActions"][this_state]["transition"][action]:
	
			prob = self.dict_transitions["transitionActions"][this_state]["transition"][action][arrival]["prob"]
			vValue = max(self.dict_qvalues[(str(arrival), "qvals")])
			accu += (prob * (reward + self.gamma * vValue))
		return previous_qvalue - accu



# -------- 2 ) AFTER the initialisation ONLY
	def update_deltas(self, modified_state, previous_value, new_value, modified_action, previous_qvalue, new_qvalue):#modified_fct
		"""
		This is the only function needed at each cycle (ie update of a state-action couple), after the initialization phase.

		:param modified_state: the updated state of the updated state-action couple
		:param previous_value: the previous value of the updated state (maximum among actions)
		:param new_value:  the new value of the updated state (maximum among actions)
		:param modified_action: the updated action of the updated state-action couple
		:param previous_qvalue: the previous q value of the updated state-action couple
		:param new_qvalue: the new q value of the updated state action couple
		"""
		difference_modified_state = previous_value - new_value # only value needed for every state-action excepted (s,a)

		if difference_modified_state != 0 : # Economy of time
			for this_state in self.dict_transitions["transitionActions"]:
				# this_state[modified_state]
				for link in range(self.action_space):
					if link in self.dict_transitions["transitionActions"][this_state]["transition"]:
						if modified_state in self.dict_transitions["transitionActions"][this_state]["transition"][link]:
							prob = self.dict_transitions["transitionActions"][this_state]["transition"][link][modified_state]["prob"]
							self.deltas[link][this_state] += prob * self.gamma * difference_modified_state
		self.deltas[modified_action][modified_state] += new_qvalue - previous_qvalue	



	def get_convergence_indicator(self):#modified_fct
		"""
		This computes the convergence indicator, out of the delta list
		:return: convergence indicatorIntelligents
		"""
		
		state_value = np.array([v for action in range(self.action_space) for k,v in self.deltas[action].items()])  #check dtype
		return np.sum(np.abs(state_value))


	def infer(self, current_state):
		"""
		In the MB expert, the process of inference consists to do planning using models of the world.
		/!\ One cycle corresponds to only ONE application of self.update_qvalue(state, action)
		"""

		# Initialize delta list
		self.get_first_infer_deltas()
		# Initialize the number of cycles
		cycle = 0
		ci = []

		activated = np.zeros((self.action_space, self.num_states))

		while True:
			# The number of states-actions in the list is constant in one single call to self.infer function
			if len(self.shuffled_list_states_actions) == 0 or cycle%len(self.shuffled_list_states_actions) == 0 :
				self.shuffled_list_states_actions = []
				for s in self.list_states:
					for a in self.list_actions[s]:
						self.shuffled_list_states_actions.append([s, a])

				np.random.shuffle(self.shuffled_list_states_actions)

			# Use low pass filter on the sum of deltaQ
			s,a = self.shuffled_list_states_actions[cycle%len(self.shuffled_list_states_actions)]

			# 1) Keep previous highest q-value in memory, for later deltas update
			previous_value = max(self.dict_qvalues[(str(s), "qvals")]) # Max among actions
			previous_qvalue = self.dict_qvalues[(s, "qvals")][a]

			# 2) Update the actual q value of this chosen state-action
			self.update_qvalue(s,a)
			new_value = max(self.dict_qvalues[(str(s), "qvals")])  # Max among actions
			new_qvalue = self.dict_qvalues[(s, "qvals")][a]

			# ORIGINAL CONVERGENCE CRITERION #

			# 3) Update new deltas list
			self.update_deltas(s, previous_value, new_value, a, previous_qvalue, new_qvalue)
			convergence_indicator = self.get_convergence_indicator()

			####################################
			'''
			# SAME CONVERGENCE CRITERION AS MF #
			self.new_deltas[a][int(s)] = np.abs(new_qvalue - previous_qvalue)

			if activated[a][int(s)] == 0:
				activated[a][int(s)] = self.base_it_replay

			activated = activated * self.tau
			convergence_indicator = np.sum(np.multiply(self.new_deltas, activated))
			'''
			####################################

			cycle += 1

			self.infer_cycles = cycle
			self.infer_cycles_tot +=cycle

			# Stop VI when convergence
			# Stop VI when convergence
			# self.new_deltas[a][int(s)] = np.abs(new_qvalue - previous_qvalue)
			# print(self.new_deltas[a][int(s)])

			'''if activated[replay_action][int(replay_previous_state)] == 0:
                activated[replay_action][int(replay_previous_state)] = self.base_it_replay

            activated = activated * self.tau'''

			#mask = activated + np.ones((self.action_space,  self.num_states)) * np.sign(activated)
			#mask = activated * np.sign(activated)
			#crit = np.sum(np.abs(np.multiply(self.new_deltas, activated)))
			# crit = np.max(self.new_deltas)

			ci.append(convergence_indicator)
			if convergence_indicator < self.epsilon:
				break
		return self.dict_qvalues[(str(current_state),"qvals")]



	def update_reward(self, current_state, reward_obtained):
		"""
		Update the the model of reward
		"""
		
		for state in self.dict_goals["values"]:
			
			# Change potentially the reward of the rewarded state
			if state["state"] == current_state:
				state["reward"] = reward_obtained
			
			for link in state["links"]:
				action = link[0]
				previous_state = link[1]
				prob = get_transition_prob(self.dict_transitions, previous_state, action, state["state"])
				relative_reward = prob  * state["reward"]
				set_reward(self.dict_rewards, previous_state, action, relative_reward)



	def update_prob(self, previous_state, action, current_state):
		"""
		Update the the model of transition (by updating the probabilities to the new ones directly estimated from the
		window of previously encountered transitions.
		Therefore, the greater the probability window is, the better the proba estimation is.
		"""
		delta_prob = 0.0
		nb_transitions = get_number_transitions(self.dict_transitions, previous_state, action)
		sum_nb_transitions = sum(nb_transitions.values())
		probs = get_transition_probs(self.dict_transitions, previous_state, action)
		for arrival, old_prob in probs.items():
			new_prob = nb_transitions[arrival]/sum_nb_transitions
			set_transition_prob(self.dict_transitions, previous_state, action, arrival, new_prob)
			delta_prob += abs(new_prob - old_prob)
		probs = get_transition_probs(self.dict_transitions, previous_state, action)


	def learn(self, previous_state, action, current_state, reward_obtained):
		"""
		Update the contents of the rewards and the transitions model
		"""

		# // Update the transition model //
		self.update_prob(previous_state, action, current_state)
		
		# // Update the reward model //
		self.update_reward(current_state, reward_obtained)


	def update_data_structure(self, action_count, previous_state, action, current_state, reward_obtained):#modified_fct
		"""
		Update the data structure of the rewards and the transitions models
		"""
		self.dict_qvalues[(str(current_state),"actioncount")] = action_count
		self.dict_transitions["actioncount"] = action_count
		self.dict_rewards["actioncount"] = action_count
		# For the modelof qvalues, update only the number of visit
		self.dict_qvalues[(str(previous_state),"visits")] += 1
		# If the previous state is unknown, add it in the states model, in the reward model and in the transition model
		# (not needful for the model of q-values because it has already is final size)
		if previous_state not in self.list_states:#change this
			# do not do for rewarded state
			if self.not_learn == False:
				self.list_states.append(previous_state)
				self.list_actions[previous_state] = [action]
				initialize_rewards(self.dict_rewards, previous_state, self.action_space)
				initialize_transition(self.dict_transitions, previous_state, action, current_state, 1, self.window_size)
		else:
			# do not do for rewarded state
			if self.not_learn == False:
				if action not in self.list_actions[previous_state]:
					self.list_actions[previous_state].append(action)
					add_transition(self.dict_transitions, previous_state, action, current_state, 1, self.window_size)
				else:
					# Check if the transition "previous state -> action -> current state" has already been experimented
					if (previous_state in self.dict_transitions["transitionActions"]) and (action in self.dict_transitions["transitionActions"][previous_state]['transition']) and (current_state in self.dict_transitions["transitionActions"][previous_state]['transition'][action]):
						# If it exist, update the window of transitions
						set_transitions_window(self.dict_transitions, previous_state, action, current_state, self.window_size)
					# If the transition doesn't exist, add it
					else:
						add_transition(self.dict_transitions, previous_state, action, current_state, 0, self.window_size)

		# Check if the agent already known goals and neighbours
		
		if reward_obtained != 0.0:
			# If none goal is known, add it in the dict with this neighbour
			if not self.dict_goals["values"]:
				self.dict_goals["values"].append({"state": current_state, "reward": reward_obtained, "links": [(action, previous_state)]})
			# Check if this goal is already known
			known_goal = False
			for state in self.dict_goals["values"]:
				if state["state"] == current_state:
					known_goal = True
					known_link = False
					for link in state["links"]:
						if link[0] == action and link[1] == previous_state:
							known_link = True
							break
					if known_link == False:
						state["links"].append((action, previous_state))
					break
			if known_goal == False:
				self.dict_goals["values"].append({"state": current_state, "reward": reward_obtained, "links": [(action, previous_state)]})
				# delete transitions and possible actions for current_state

				if current_state in self.dict_transitions["transitionActions"]:
					self.dict_transitions["transitionActions"][current_state]["transition"]=dict()
				for i in range (self.action_space):
					self.list_actions[current_state] = []


	def run(self, action_count, cumulated_reward, reward_obtained, previous_state, decided_action, current_state, do_we_plan): 
		"""
		Run the model-based system
		"""
		# Update the actioncount and the number of the visits for the previous state
		self.dict_duration["actioncount"] = action_count
		self.dict_delta_prob["actioncount"] = action_count
		self.dict_qvalues[(str(current_state),"actioncount")] = action_count
		self.dict_qvalues[(str(previous_state),"visits")] += 1
		# Update the data structure of the models (states, rewards, transitions, qvalues)
		self.update_data_structure(action_count, previous_state, decided_action, current_state, reward_obtained)
		if self.not_learn == False:
		# Update the transition model and the reward model according to the learning.
			self.learn(previous_state, decided_action, current_state, reward_obtained)
		# If the expert was chosen to plan, update all the q-values by planning
		
		if do_we_plan: 
			old_time = datetime.datetime.now()
			# Run the planning process
			qvalues = self.infer(current_state) # INFERENCE & SIMULATION
			# Sum the duration of planification with a low pass filter
			current_time = datetime.datetime.now()
			new_plan_time = (current_time - old_time).total_seconds()
			old_plan_time = get_duration(self.dict_duration, current_state)
			filtered_time = low_pass_filter(self.alpha, old_plan_time, new_plan_time)
			set_duration(self.dict_duration, current_state, filtered_time)
		else:
			qvalues = self.dict_qvalues[(str(current_state),"qvals")]
		# Choose the next action to do from the current state using soft-max policy.
		decided_action, actions_prob = self.decide(current_state, qvalues)
		plan_time = get_duration(self.dict_duration, current_state)
		selection_prob = get_filtered_prob(self.dict_actions_prob, current_state)
		if reward_obtained > 0.0:
			self.not_learn = True
			for a in range(self.action_space):
				self.dict_qvalues[(current_state,"qvals")] = [0.0]*self.action_space
		else:
			self.not_learn = False      
		# Logs 
		if self.log == True:
			self.reward_log.write("timecount"+" "+str(action_count)+" "+str(reward_obtained)+" currentTime-nodeStartTime"+" currentTime"+"\n")
			self.qvalues_evolution_log.write(",\n"+json.dumps(self.dict_qvalues))
			self.actions_evolution_log.write(",\n"+json.dumps(self.dict_actions_prob))
			self.monitoring_values_log.write(str(action_count)+" "+str(decided_action)+" "+str(plan_time)+" "+str(selection_prob)+" "+str(prefered_action)+"\n")
		#Finish the logging at the end of the simulation (duration or max reward)
		if (action_count == self.duration) or (cumulated_reward == self.max_reward):
			if self.log == True:
				self.actions_evolution_log.write('],\n"name" : "Actions"\n}')
			# Build the summary file
			if self.summary == True:
				if self.directory_flag == True:
					os.chdir("../")
				prefixe = 'v%d_TBMB_'%(VERSION)
				self.summary_log = open(prefixe+'summary_log.dat', 'a')
				self.summary_log.write(str(self.gamma)+" "+str(self.beta)+" "+str(cumulated_reward)+"\n")
		return decided_action



