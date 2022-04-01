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
import json
import datetime

VERSION = 1


class QLearning:
	"""
	This class implements a MODEL-FREE MODULE, ie learning algorithm (q-learning).
    """

	def __init__(self, experiment, map_file, initial_variables, boundaries_exp, parameters, options_log):
		"""
		Iinitialise values and models
		"""

		# --- KEEP TRACK OF REPLAY CYCLES ---
		self.replay_cycles = 0
		self.replay_time = 0
		# --- REPLAY BUDGET, INITIALIZED BUT NOT USED ---
		self.replay_budget = 0
		

		self.experiment = experiment
		self.max_reward = boundaries_exp["max_reward"]
		self.duration = boundaries_exp["duration"]
		self.window_size = boundaries_exp["window_size"]
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
			
			self.dict_qvalues[(s,"qvals")] = [self.init_qvalue]*8   # one qvalue per each one of the 8 actions
			self.dict_qvalues[(s,"visits")] = 0
			
			# - initialise the "probabilties of actions" dict
			self.dict_actions_prob["values"][s]={ "actions_prob": [init_actions_prob]*8, "filtered_prob": [init_actions_prob]*8}
			

			# - initialise the duration dict
			self.dict_duration["values"][s]={ "duration": 0.0}

		# I COMMENTED THIS PART BELOW FOR THE LOGS BECAUSE I DID NOT USE IT
		
		# Initialise logs
		if self.log == True:
			self.directory_flag = False
			if not os.path.exists("logs"):
				os.mkdir("logs") 
			os.chdir("logs")
			if not os.path.exists("MF"):
				os.mkdir("MF") 
			os.chdir("MF")
			directory = "exp"+str(self.experiment)+"_alpha"+str(self.alpha)+"_gamma"+str(self.gamma)+"_beta"+str(self.beta)
			if not os.path.exists(directory):
				os.makedirs(directory)
			os.chdir(directory)
			self.directory_flag = True
			
			prefixe = "v"+str(VERSION)+"_TBMF_exp"+str(self.experiment)+"_"
			
			self.reward_log = open(prefixe+'reward_log.dat', 'w')
			self.reward_log.write("timecount"+" "+str(action_count)+" "+str(init_reward)+" currentTime-nodeStartTime"+" currentTime"+"\n")
			
			self.states_evolution_log = open(prefixe+'statesEvolution_log.dat', 'w')
			self.states_evolution_log.write("timecount"+" "+str(action_count)+" "+current_state+" "+previous_state+ \
				" currentContactState"+" currentViewState"+" "+str(decided_action)+" currentTime-nodeStartTime"+" currentTime"+"\n")
			
			
			self.qvalues_evolution_log = open(prefixe+'qvaluesEvolution_log.dat', 'w')
			self.qvalues_evolution_log.write('{\n"logs" :\n['+json.dumps(self.dict_qvalues))
			#error key = tuple doesnt work in json
			
			
			self.actions_evolution_log = open(prefixe+'actions_evolution_log.dat', 'w')
			
			self.actions_evolution_log.write('{\n"logs" :\n['+json.dumps(self.dict_actions_prob))
			self.actions_evolution_log.close()

			self.monitoring_values_log = open(prefixe+'monitoring_values_log.dat', 'w')
			self.monitoring_values_log.write(str(action_count)+" "+str(init_plan_time)+" "+str(abs(init_delta))+" "+str(init_delta)+" "+str(init_delta)+"\n")
		
			os.chdir("../../../")
		


	def __del__(self):
		"""
		Close all log files
		"""
		
		if self.log == True:
			self.reward_log.close()
			self.qvalues_evolution_log.close() #error key = tuple doesnt work in json
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
		
		return new_qvalue
		


	def run(self, action_count, cumulated_reward, reward_obtained, previous_state, decided_action, current_state,
			do_we_plan):
		"""
		Run the model-free system
		"""

		
		# Update the actioncount and the number of the visits for the previous state
		self.dict_duration["actioncount"] = action_count
		self.dict_actions_prob["actioncount"] = action_count
		self.dict_qvalues[(str(current_state),"actioncount")] = action_count
		self.dict_qvalues[(str(previous_state),"visits")] += 1
		
		if self.not_learn == False:
			# Update the q-values of the previous state using Q-learning
			self.new_qvalue = self.learn(previous_state, decided_action, current_state, reward_obtained)
		
		# If the expert was choosen to plan, compute the news probabilities of actions
		if do_we_plan:
			
			old_time = datetime.datetime.now()
			
			# Run the process of inference
			qvalues = self.infer(current_state)
			
			# Sum the duration of planification with a low pass filter
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
			for a in range(0,8):
				self.dict_qvalues[(current_state,"qvals")] = [0.0]*8
			# reset where reward is found ( as no anymore move )
		else:
			self.not_learn = False

		# Logs
		if self.log == True:
			self.reward_log.write("timecount"+" "+str(action_count)+" "+str(reward_obtained)+" currentTime-nodeStartTime"+" currentTime"+"\n")
			self.states_evolution_log.write("timecount"+" "+str(action_count)+" "+current_state+" "+previous_state+ \
				" currentContactState"+" currentViewState"+" "+str(decided_action)+" currentTime-nodeStartTime"+" currentTime"+"\n")
			self.qvalues_evolution_log.write(",\n"+json.dumps(self.dict_qvalues)) #error key = tuple doesnt work in json
			self.actions_evolution_log.write(",\n"+json.dumps(self.dict_actions_prob))
			self.monitoring_values_log.write(str(action_count)+" "+str(decided_action)+" "+str(plan_time)+" "+str(selection_prob)+" "+str(prefered_action)+"\n")
		# Finish the logging at the end of the simulation (duration or max reward)
		if (action_count == self.duration) or (cumulated_reward == self.max_reward):
			if self.log == True:
				self.qvalues_evolution_log.write('],\n"name" : "Qvalues"\n}') #error key = tuple doesnt work in json
				self.actions_evolution_log.write('],\n"name" : "Actions"\n}')
			# Build the summary file
			if self.summary == True:
				if self.directory_flag == True:
					os.chdir("../")
				prefixe = 'v%d_TBMF_'%(VERSION)
				self.summary_log = open(prefixe+'summary_log.dat', 'a')
				self.summary_log.write(str(self.alpha)+" "+str(self.gamma)+" "+str(self.beta)+" "+str(cumulated_reward)+"\n")

		return decided_action



