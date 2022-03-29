'''
This script permits with the other ones in the folder to simulate an Erwan Renaudo's 
experiment : 
"Renaudo, E. (2016). Des comportements flexibles aux comportements habituels: 
Meta-apprentissage neuro-inspiré pour la robotique autonome (Doctoral dissertation, 
Université Pierre et Marie Curie (Paris 6))". 

With this script, the simulated agent can choose betwen the decision proposed 
by the model-based system or the decision proposed by the the model-free system.
'''

from utility import *
import json
import datetime
import os

VERSION = 1


class MetaController:
	"""
	This class implements a meta-controller that allows the agent to choose between
	several possible decisions according to some crieria
    """

	def __init__(self, agent_model_based, agent_model_free, experiment, map_file, initial_variables, boundaries_exp,
				 beta_MC, criterion, coeff_kappa, options_log):
		"""
		Iinitialise values and models
		"""
		# ---------------------------------------------------------------------------
		self.agent_model_based = agent_model_based
		self.agent_model_free = agent_model_free
		self.experiment = experiment
		current_state = initial_variables["current_state"]
		action_count = initial_variables["action_count"]
		initial_reward = initial_variables["reward"]
		initial_duration = initial_variables["plan_time"]
		replays = initial_variables["replays"]
		self.beta_MC = beta_MC
		self.criterion = criterion
		self.coeff_kappa = coeff_kappa
		self.log = options_log["log"]
		self.epsilon = boundaries_exp["epsilon"]
		# Create a dict that contains the time of planification for each states
		self.dict_duration = dict()
		self.dict_duration["actioncount"] = action_count
		self.dict_duration["values"] = dict()
		# initialise the duration dict
		with open(map_file,'r') as file2:
			self.map = json.load(file2)
		for state in self.map["transitionActions"]:
			s = str(state["state"]).replace(" ", "")
			self.dict_duration["values"][s]={ "duration": 0.0}


		if self.log == True or True:
			mf_special_vals = ""
			mb_special_vals = ""
			self.directory_flag = False
			try:
				os.stat("logs")
			except:
				os.mkdir("logs") 
			os.chdir("logs")

			if self.agent_model_based.module_type == 'value-iteration-shuffle-budget':
				mb_special_vals = "(budget_"+str(self.agent_model_based.module.infer_budget)+")"
				print(mb_special_vals)
			if self.agent_model_free.module_type in [ 'q-learning-replay-budget', 'dyna-q']:
				mf_special_vals = "(replay_budget_" + str(self.agent_model_free.module.replay_budget) + ")"
			if self.agent_model_free.module_type in [ 'dqn', 'dqn-prio'] :
				mf_special_vals = "(minibatch_size_" + str(self.agent_model_free.module.minibatch_size) + ")"
			if self.agent_model_free.module_type in ['prioritized-sweeping-1',  'prioritized-sweeping-4-time-win',
													 'prioritized-sweeping-1-time-win' ,'prioritized-sweeping-2',
													 'prioritized-sweeping-2-time-win' ] :
				mf_special_vals = "(replay_budget_" + str(self.agent_model_free.module.replay_budget) \
								  +"|delta_thresh_"+str(self.agent_model_free.module.delta_threshold)+ ")"
			if self.agent_model_based.module_type in [ 'value-iteration-shuffle-shared-replay-budget-time-win',
													   'value-iteration-shuffle-replay-budget-time-win-a',
													   'value-iteration-shuffle-replay-budget-time-win-b']:
				mb_special_vals = "(replay_budget_"+str(self.agent_model_based.module.replay_budget) \
								  +"|infer_budget_"+str(self.agent_model_based.module.infer_budget)+\
								  "|delta_threshold_"+str(self.agent_model_based.module.delta_threshold)+")"

			if self.agent_model_based.module_type in [ 'value-iteration-shuffle-bi-replay-budget-time-win']:
				mb_special_vals = "(replay_budget_"+str(self.agent_model_based.module.replay_budget) \
								  +"|infer_budget_"+str(self.agent_model_based.module.infer_budget)+\
								  "|bi_budget_"+str(self.agent_model_based.module.bidirectional_budget)+\
								  "|delta_threshold_"+str(self.agent_model_based.module.delta_threshold)+")"

########################################################################################################################
			if self.criterion== 'MF_only':
				self.fileName = str(self.criterion) +"_MF_["+self.agent_model_free.module_type +"]" + mf_special_vals+"_coeff"+str(self.coeff_kappa)+"_exp"+str(self.experiment)+"_log.dat"
				self.MC_log = open(self.fileName, "w")
				# self.MC_log = open(str(self.criterion)+"_MB_["+self.agent_model_based.module_type+"]" + mb_special_vals+"_MF_["+self.agent_model_free.module_type +"]"+"_window"+str(agent_model_based.module.window_size)+"_log.dat", "w")
				print(self.fileName)
				self.MC_log.write(str(action_count)+" "+str(current_state)+" "+str(initial_reward)+" "+str(initial_duration)+" "+str(replays)+" MB 0.5 0.0 0.0 0.0 0\n")

			elif self.criterion== 'MB_only':
				self.fileName = str(self.criterion)+"_MB_["+self.agent_model_based.module_type+"]" + mb_special_vals+"_window"+str(agent_model_based.module.window_size)+"_coeff"+str(self.coeff_kappa)+"_exp"+str(self.experiment)+"_log.dat"
				self.MC_log = open(self.fileName, "w")
				# self.MC_log = open(str(self.criterion)+"_MB_["+self.agent_model_based.module_type+"]" + mb_special_vals+"_MF_["+self.agent_model_free.module_type +"]"+"_window"+str(agent_model_based.module.window_size)+"_log.dat", "w")
				print(self.fileName)
				self.MC_log.write(str(action_count)+" "+str(current_state)+" "+str(initial_reward)+" "+str(initial_duration)+" "+str(replays)+" MB 0.5 0.0 0.0 0.0 0\n")

			else:
				self.fileName = str(self.criterion)+"_MB_["+self.agent_model_based.module_type+"]" + mb_special_vals+"_MF_["+self.agent_model_free.module_type +"]" + mf_special_vals+"_window"+str(agent_model_based.module.window_size)+"_coeff"+str(self.coeff_kappa)+"_exp"+str(self.experiment)+"_log.dat"
				print(self.fileName)
				self.MC_log = open(self.fileName, "w")
				# self.MC_log = open(str(self.criterion)+"_MB_["+self.agent_model_based.module_type+"]" + mb_special_vals+"_MF_["+self.agent_model_free.module_type +"]"+"_window"+str(agent_model_based.module.window_size)+"_log.dat", "w")
				
				self.MC_log.write(str(action_count)+" "+str(current_state)+" "+str(initial_reward)+" "+str(initial_duration)+" "+str(replays)+" MB 0.5 0.0 0.0 0.0 0\n")
		# ---------------------------------------------------------------------------
		os.chdir("..")
		# ---------------------------------------------------------------------------


	def entropy_and_time(self, duration, selection_prob):
		"""
		Choose the best action according to a trade-off betwen the time of planning 
		and the quality of learning (entropy of the ditribution probability of the actions). These parameters
		are normalized.

		--> stochastic expert choice is possible , but not yet enabled
		"""
		# ---------------------------------------------------------------------------
		norm_probs_MF = [prob / sum(selection_prob["MF"]) for prob in selection_prob["MF"]]
		norm_probs_MB = [prob / sum(selection_prob["MB"]) for prob in selection_prob["MB"]]
		# ---------------------------------------------------------------------------
		entropy_probs_MF = shanon_entropy(norm_probs_MF)
		#print("Entropy MF :"+str(entropy_probs_MF))
		entropy_probs_MB = shanon_entropy(norm_probs_MB)
		#print("Entropy MB :"+str(entropy_probs_MB))
		# ---------------------------------------------------------------------------
		max_entropy = shanon_entropy([1/(len(norm_probs_MF))]*len(norm_probs_MF)) # all probas, uniform probability law case
		highest_entropy = max(entropy_probs_MF,entropy_probs_MB)
		mean_entropy = (entropy_probs_MF + entropy_probs_MB) / 2
		# ---------------------------------------------------------------------------
		norm_entropy_MF = entropy_probs_MF / max_entropy
		norm_entropy_MB = entropy_probs_MB / max_entropy
		self.norm_entropy = {"MF": norm_entropy_MF, "MB": norm_entropy_MB}
		#print("Norm entropy MF: "+str(norm_entropy_MF))
		#print("Norm entropy MB: "+str(norm_entropy_MB))
		# ---------------------------------------------------------------------------
		max_duration = max(duration["MF"],duration["MB"])
		if max_duration == 0.0:
			max_duration = 0.000000000001
		norm_duration_MF = (duration["MF"]) / max_duration
		norm_duration_MB = (duration["MB"]) / max_duration
		# ---------------------------------------------------------------------------
		coeff  = math.exp(-entropy_probs_MF * self.coeff_kappa)
		#print("Coeff = exp(-"+str(entropy_probs_MF)+" * "+str(self.coeff_kappa)+") = "+str(coeff))
		# ---------------------------------------------------------------------------
		qval_MF = - (norm_entropy_MF + coeff * norm_duration_MF)
		#print("Qval MF : - ("+str(norm_entropy_MF)+" + "+str(coeff)+" * "+str(norm_duration_MF)+")) = "+str(qval_MF))
		qval_MB = - (norm_entropy_MB + coeff * norm_duration_MB)
		#print("Qval MB : - ("+str(norm_entropy_MB)+" + "+str(coeff)+" * "+str(norm_duration_MB)+")) = "+str(qval_MB))
		qvalues = {"MF": qval_MF, "MB": qval_MB}
		# ---------------------------------------------------------------------------
		# Soft-max function
		#final_actions_prob = softmax_actions_prob(qvalues, self.beta_MC)
		#expert, final_decision = softmax_decision(final_actions_prob, decisions)
		# ---------------------------------------------------------------------------
		# Arg-max function
		final_actions_prob = softmax_actions_prob(qvalues, self.beta_MC)
		if qval_MF > qval_MB:
			expert = "MF"
			who_plan = {"MF": True, "MB": False}
		elif qval_MF < qval_MB:
			expert = "MB"
			who_plan = {"MF": False, "MB": True}
		elif qval_MB == qval_MF:
			randval = np.random.rand()
			if randval >= 0.500000000:
				expert = "MF"
				who_plan = {"MF": True, "MB": False}
			elif randval < 0.50000000:
				expert = "MB"
				who_plan = {"MF": False, "MB": True}
		# ---------------------------------------------------------------------------
		return final_actions_prob, who_plan
		# ---------------------------------------------------------------------------


	def entropy_only(self, selection_prob):
		"""
		Choose the best action according to the value of the entropies of the ditribution 
		probability of the actions. These parametersare normalized.

		---> stochastic expert choice is possible, but not yet enabled
		"""
		# ---------------------------------------------------------------------------
		norm_probs_MF = [prob / sum(selection_prob["MF"]) for prob in selection_prob["MF"]]
		norm_probs_MB = [prob / sum(selection_prob["MB"]) for prob in selection_prob["MB"]]
		# ---------------------------------------------------------------------------
		entropy_probs_MF = shanon_entropy(norm_probs_MF)
		#print("Entropy MF :"+str(entropy_probs_MF))
		entropy_probs_MB = shanon_entropy(norm_probs_MB)
		#print("Entropy MB :"+str(entropy_probs_MB))
		# ---------------------------------------------------------------------------
		max_entropy = shanon_entropy([1/(len(norm_probs_MF))]*len(norm_probs_MF))
		highest_entropy = max(entropy_probs_MF,entropy_probs_MB)
		mean_entropy = (entropy_probs_MF + entropy_probs_MB) / 2
		# ---------------------------------------------------------------------------
		norm_entropy_MF = entropy_probs_MF / max_entropy
		norm_entropy_MB = entropy_probs_MB / max_entropy
		self.norm_entropy = {"MF": norm_entropy_MF, "MB": norm_entropy_MB}
		#print("Norm entropy MF: "+str(norm_entropy_MF))
		#print("Norm entropy MB: "+str(norm_entropy_MB))
		# ---------------------------------------------------------------------------
		qval_MF = - (norm_entropy_MF)
		#print("Qval MF : - "+str(norm_entropy_MF)+") = "+str(qval_MF))
		qval_MB = - (norm_entropy_MB)
		#print("Qval MB : - "+str(norm_entropy_MB)+") = "+str(qval_MB))
		qvalues = {"MF": qval_MF, "MB": qval_MB}
		# ---------------------------------------------------------------------------
		# Soft-max function
		#final_actions_prob = softmax_actions_prob(qvalues, self.beta_MC)
		#expert, final_decision = softmax_decision(final_actions_prob, decisions)
		# ---------------------------------------------------------------------------
		# Arg-max function
		final_actions_prob = softmax_actions_prob(qvalues, self.beta_MC)
		if qval_MF >= qval_MB:
			expert = "MF"
			who_plan = {"MF": True, "MB": False}
		else:
			expert = "MB"
			who_plan = {"MF": False, "MB": True}
		# ---------------------------------------------------------------------------
		return final_actions_prob, who_plan
		# ---------------------------------------------------------------------------


	def only_one(self, selection_prob, expert):
		"""
		Choose the action between those proposed randomly

		--> like entropy_only, but deterministic expert choice only (argmax)
		some normalized entropy is therefore computed but useless
		"""
		# ---------------------------------------------------------------------------
		norm_probs_MF = [prob / sum(selection_prob["MF"]) for prob in selection_prob["MF"]]
		norm_probs_MB = [prob / sum(selection_prob["MB"]) for prob in selection_prob["MB"]]
		entropy_probs_MF = shanon_entropy(norm_probs_MF)
		entropy_probs_MB = shanon_entropy(norm_probs_MB)
		max_entropy = shanon_entropy([1/(len(norm_probs_MF))]*len(norm_probs_MF))
		norm_entropy_MF = entropy_probs_MF / max_entropy
		norm_entropy_MB = entropy_probs_MB / max_entropy
		# ---------------------------------------------------------------------------
		if expert == "MF": 
			who_plan = {"MF": True, "MB": False}
			final_actions_prob = {"MF": 1, "MB": 0}
			self.norm_entropy = {"MF": norm_entropy_MF, "MB": 0.0}
		elif expert == "MB":
			who_plan = {"MF": False, "MB": True}
			final_actions_prob = {"MF": 0, "MB": 1}
			self.norm_entropy = {"MF": 0.0, "MB": norm_entropy_MB}
		# ---------------------------------------------------------------------------
		return final_actions_prob, who_plan
		# ---------------------------------------------------------------------------


	def random(self):
		"""
		Choose the action between those proposed randomly
		"""
		# ---------------------------------------------------------------------------
		#print("Decisions :"+str(decisions))
		randval = np.random.rand()
		if randval <= 0.500000000:
			who_plan = {"MF": True, "MB": False}
			self.norm_entropy = {"MF": 0.0, "MB": 0.0}
		elif randval > 0.50000000:
			who_plan = {"MF": False, "MB": True}
			self.norm_entropy = {"MF": 0.0, "MB": 0.0}
		# ---------------------------------------------------------------------------
		final_actions_prob = {"MF": 0.5, "MB": 0.5}
		# ---------------------------------------------------------------------------
		return final_actions_prob, who_plan
		# ---------------------------------------------------------------------------


	def decide(self, plan_time, selection_prob):
		"""
		Choose the action between those proposed according to the choosen criteria
		"""
		# ---------------------------------------------------------------------------
		if self.criterion == "random": 
			final_actions_prob, who_plan = self.random()
		elif self.criterion == "MF_only":
			final_actions_prob, who_plan = self.only_one(selection_prob, "MF") 
		elif self.criterion == "MB_only":
			final_actions_prob, who_plan = self.only_one(selection_prob, "MB")
		elif self.criterion == "Entropy_and_time":
			final_actions_prob, who_plan = self.entropy_and_time(plan_time, selection_prob)
		elif self.criterion == "Entropy_only":
			final_actions_prob, who_plan = self.entropy_only(selection_prob)

		else:
			sys.exit("This criterion is unknown. Retry with a good one.")
		# ---------------------------------------------------------------------------
		return final_actions_prob, who_plan  # final action prob are probs, who plan makes only one be set to True
		# ---------------------------------------------------------------------------


	def run(self, action_count, reward_obtained, current_state, plan_time, selection_prob):

		"""
		Run the metacontroller

		--> selection probs : probs of actions for each agent (MF, MB ...) , not probs of selection of agent
		"""
		# ---------------------------------------------------------------------------
		old_time = datetime.datetime.now()
		# ---------------------------------------------------------------------------
		#print("------------------------ MC --------------------------------")
		#print("Plan time : "+str(plan_time))
		#print("Probability of actions : "+str(selection_prob))
		#print("Repartition of the prefered actions : "+str(prefered_action))
		# ---------------------------------------------------------------------------
		# Decide betwen the two experts accoring to the choosen criterion
		final_actions_prob, who_plan = self.decide(plan_time, selection_prob)
		# ---------------------------------------------------------------------------
		# Sum the duration of planification with a low pass filter
		current_time = datetime.datetime.now()
		new_plan_time = (current_time - old_time).total_seconds()
		old_plan_time = get_duration(self.dict_duration, current_state)
		filtered_time = low_pass_filter(0.6, old_plan_time, new_plan_time)
		set_duration(self.dict_duration, current_state, filtered_time)#update time
		# ---------------------------------------------------------------------------
		# Logs
			# -----------------------------------------------------------------------
		time = 0.0
		if who_plan["MB"] == True:
			time += plan_time["MB"] # plan time, not the time for replays in case there are replays
		if who_plan["MF"] == True:
			time += plan_time["MF"]
		# -----------------------------------------------------------------------
		if who_plan["MF"] == True:
			if self.log == True or True:# or true because log doenst work for other fonction
				# print(self.fileName)
				self.MC_log.write(str(action_count)+" "+str(current_state)+" "+str(reward_obtained)+" "+str(time)+" MF "
							  +str(final_actions_prob["MF"])+" "+str(self.norm_entropy["MF"])+" "+str(self.norm_entropy["MB"])
							  +" "+str(filtered_time)+ " " + str(self.agent_model_free.module.replay_cycles)+
							  " "+str(self.agent_model_free.module.replay_time)+"\n")
			winner = "MF"

		elif who_plan["MB"] == True:
			if self.log == True or True:# or true because log doenst work for other fonction
				self.MC_log.write(str(action_count)+" "+str(current_state)+" "+str(reward_obtained)+" "+str(time)+" MB "
							  +str(final_actions_prob["MB"])+" "+str(self.norm_entropy["MF"])+" "+str(self.norm_entropy["MB"])+
							  " "+str(filtered_time)+" "+str(self.agent_model_based.module.infer_cycles)
							  +" "+str(self.agent_model_based.module.replay_cycles)
							  + " " + str(self.agent_model_based.module.replay_time)
							  + " " + str(self.agent_model_based.module.bi_replay_cycles)
							  + " " + str(self.agent_model_based.module.bi_replay_time)
							  +"\n")
			winner = "MB"
		# --------------------------------------------------------------------------
		return winner, who_plan
		# ---------------------------------------------------------------------------

