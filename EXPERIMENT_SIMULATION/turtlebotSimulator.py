'''
This script permits with the other in the folder ones to simulate an Erwan Renaudo's 
experiment : 
"Renaudo, E. (2016). Des comportements flexibles aux comportements habituels: 
Meta-apprentissage neuro-inspiré pour la robotique autonome (Doctoral dissertation, 
Université Pierre et Marie Curie (Paris 6))". 
'''

from colorama import Fore, Style
from optparse import OptionParser
from tqdm import *  # easier overview of experiment advancement than prints
from manageEnvironment import *
from ParseMap import *
from metaController import MetaController
from agentModelFree import AgentModelFree
from agentModelBased import AgentModelBased
import sys


def manage_arguments():
	"""
	Manage the arguments of the script
	"""
	usage = "usage: main.py [options] [the number of experiment][the file that containts the structure of the " \
			"environemnt][the file that contains the list of key states] [the file that contains the parameters of each" \
			"expert]"
	parser = OptionParser(usage)
	parser.add_option("-c", "--criterion", action = "store", type = "string", dest = "criterion",
					  help = "This option is the criterion used for the trade-off betwen the two experts",
					  default = "random")
	parser.add_option("-f", "--model_free_type", action = "store", type = "string",
					  dest = "agent_model_free_module_type",
					  help = "This option is the module used by the model free expert/agent", default = "q-learning")
	parser.add_option("-b", "--model_based_type", action = "store", type = "string",
					  dest = "agent_model_based_module_type",
					  help = "This option is the module used by the model based expert/agent",
					  default = "value-iteration-shuffle")
	parser.add_option("-k", "--coeff_kappa", action = "store", type = "float", dest = "coeff_kappa",
					  help = "This option is the coefficient use by the kappa parameter to weight the time",
					  default = 1.0)
	parser.add_option("-r", "--max_reward", action = "store", type = "int", dest = "max_reward",
					  help = "This option is the maximum cumulated reward that the agent will reach before to stop.",
					  default = 10000)
	parser.add_option("-d", "--duration", action = "store", type = "int", dest = "duration",
					  help = "This option is the maximum duration during which the agent will work.",
					  default = 100000)
	parser.add_option("-w", "--window_size", action = "store", type = "int", dest = "window_size",
					  help = "This option is the size of the window of transitions memorized by the agent.",
					  default = 30)
	parser.add_option("-n", "--new_goal", action = "store_true",dest = "change_goal",
					  help = "This option says if the goal will change during the experiment", default = False)
	parser.add_option("-g", "--file_goal", action = "store", type = "string",dest = "file_goal",
					  help = "This option is the file if goal will change during the experiment",
					  default = "Goals.txt")
	parser.add_option("-a", "--add_wall", action = "store_true", dest = "add_wall",
					  help = "This option says if a wall is added during the experiment", default = False)
	parser.add_option("-z", "--file_wall", action = "store",type = "string", dest = "file_wall",
					  help = "This option is the file if a wall is added during the experiment", default = "Walls.txt")
	parser.add_option("-l", "--log", action = "store_true", dest = "log",
					  help =  "This option permit to no log the data.", default = False)
	parser.add_option("-s", "--summary", action = "store_true", dest = "summary",
					  help = "This option permit to make a summary of the data in one file to the grid search.",
					  default = False)
	(options, args) = parser.parse_args()

	if len(args) != 4:
		parser.error("wrong number of arguments")
	else:
		experiment = sys.argv[1]
		map_file = sys.argv[2]
		key_states_file = sys.argv[3]
		parameters_file = sys.argv[4]
	return(experiment, map_file, key_states_file, parameters_file, options)


def parse_parameters(parameters_file):
	"""
	Parse the file that contains the parametersStep
	"""
	with open(parameters_file,'r') as file1:
		for line in file1:
			if line.split(" ")[0] == "MF":
				alpha_MF = float(line.split(" ")[1])
				gamma_MF = float(line.split(" ")[2])
				beta_MF = int(line.split(" ")[3])
				replay_budget_MF = int(line.split(" ")[4])
				delta_threshold_MF = float(line.split(" ")[5])  # For replay priorities
			elif line.split(" ")[0] == "MB":
				gamma_MB = float(line.split(" ")[1])
				beta_MB = int(line.split(" ")[2])
				budget = int(line.split(" ")[3]) # Added budget
				replay_budget_MB = int(line.split(" ")[4]) # Added number of replays
				delta_threshold_MB = float(line.split(" ")[5]) # For replay priorities
				bidirectional_budget_MB = int(line.split(" ")[6]) # Budget for the forward phase of the
			# bidirectional algo
			elif line.split(" ")[0] == "MC":
				beta_MC = int(line.split(" ")[1])

	parameters_agent_MF = {'alpha': alpha_MF, "gamma": gamma_MF, "beta": beta_MF, "replay_budget" : replay_budget_MF,
						   "delta_threshold": delta_threshold_MF}

	parameters_agent_MB = {'alpha': alpha_MF, "gamma": gamma_MB, "beta": beta_MB, "infer_budget": budget,
						   "replay_budget" : replay_budget_MB,
						   "delta_threshold" : delta_threshold_MB, "bidirectional_budget": bidirectional_budget_MB}

	return parameters_agent_MF, parameters_agent_MB, beta_MC


# -------------  THIS FUNCTION SIMULATES ONE CYCLE OF THE TURTLEBOT EXPERIMENT ---------------------
def oneCycle(environement,options, pbar, meta_controller_system, who_plan_global, who_plan_local, agent_model_free,
			 agent_model_based, current_state, cumulated_reward, reward_obtained, previous_state, final_decision,
			 listWall,listGoal,switchGoal,switchWall,path):
	action_count = environement.action_count
	goalhaschange=False
	wallhaschange=False
	# Update potentially the environment
	if options.change_goal == True and action_count == switchGoal:
		swGtemp=10E1000
		for i in listGoal:
			if i["iter"]==switchGoal:
				environement.reward["state"] = i["new_goal"]["state"]
				environement.reward["value"] = i["new_goal"]["value"]
				print("The rewarded state has changed ! Now the state "+str(i["new_goal"]["state"])+
					  " gives the reward.")
			if i["iter"]>switchGoal and i["iter"]<swGtemp:
				swGtemp=i["iter"]
		switchGoal=swGtemp
		goalhaschange=True

	if options.add_wall == True and action_count == switchWall:
		print("wall added!")
		swWtemp=10E1000
		for i in listWall:
			if i["iter"]==switchWall:
				if len(i["list_wall"])==1:
					for duoState in i["list_wall"][0]:
						
						if duoState["direction"]==1:
							environement.addWalluni(environement.listState[ duoState["state1"]],
													environement.listState[duoState["state2"]])
						else:
							environement.addWall(environement.listState[duoState["state1"]],
												 environement.listState[duoState["state2"]])
				else:
					
					if path[0] >= path[1]: #you can change the condition to follow
						n=1
					else:
						n=0
					for duoState in i["list_wall"][n]:
						if duoState["direction"]==1:
							
							environement.addWalluni(environement.listState[duoState["state1"]],
													environement.listState[duoState["state2"]])
						else:
							
							environement.addWall(environement.listState[duoState["state1"]],
												 environement.listState[duoState["state2"]])
							print("wall:",environement.listState[duoState["state1"]].id,
								  environement.listState[duoState["state2"]].id)
						
			if i["iter"]>switchWall and i["iter"]<swWtemp:
				swGtemp=i["iter"]
		switchWall=swWtemp
		wallhaschange=True
		environement.printEnvironementFile()

	# Get the probabilities of selection of the two expert for the current state accoring to the q-values
	if meta_controller_system.criterion == 'MB_only':
		selection_prob_MB = agent_model_based.module.get_actions_prob(current_state)
		selection_prob = {"MF": selection_prob_MB, "MB": selection_prob_MB}
		plan_time_MB = agent_model_based.module.get_plan_time(current_state)
		plan_time = {"MF": plan_time_MB, "MB": plan_time_MB}
		who_plan_local[current_state] = {"MF": False, "MB": True}
		meta_controller_system.run(action_count, reward_obtained, current_state,plan_time, selection_prob)
		decision_MB = agent_model_based.module.run(action_count, cumulated_reward, reward_obtained, previous_state,
											   final_decision, current_state, who_plan_local[current_state]["MB"])
		decision_MF = -1		
		decisions = {"MF": decision_MF, "MB": decision_MB}
		final_decision = decisions["MB"]
		winner = "MB"						   

	elif meta_controller_system.criterion == 'MF_only':
		selection_prob_MF = agent_model_free.module.get_actions_prob(current_state)
		selection_prob = {"MF": selection_prob_MF, "MB":selection_prob_MF}
		plan_time_MF = agent_model_free.module.get_plan_time(current_state)
		plan_time = {"MF": plan_time_MF, "MB": plan_time_MF}
		who_plan_local[current_state] = {"MF": True, "MB": False}
		meta_controller_system.run(action_count, reward_obtained, current_state,plan_time, selection_prob)
		decision_MF = agent_model_free.module.run(action_count, cumulated_reward, reward_obtained, previous_state,
												  final_decision, current_state, who_plan_local[current_state]["MF"])
		decision_MB = -1
		decisions = {"MF": decision_MF, "MB": decision_MB}
		final_decision = decisions["MF"]
		winner="MF"
	else:
		selection_prob_MF = agent_model_free.module.get_actions_prob(current_state)
		selection_prob_MB = agent_model_based.module.get_actions_prob(current_state)

		selection_prob = {"MF": selection_prob_MF, "MB": selection_prob_MB}
		# Get the time of planification of the two expert for the current state according to the previous one
		plan_time_MF = agent_model_free.module.get_plan_time(current_state)
		plan_time_MB = agent_model_based.module.get_plan_time(current_state)
		plan_time = {"MF": plan_time_MF, "MB": plan_time_MB}
		# Choose which expert to inhibit with the MC using a criterion of coordination
		first_visit = False
		if current_state not in who_plan_local.keys():
			first_visit = True
		winner, who_plan_local[current_state] = meta_controller_system.run(action_count, reward_obtained,
																		   current_state,plan_time, selection_prob)
																	   
		if first_visit == True:
			who_plan_local[current_state] = {"MF": True, "MB": True}

		decision_MB = agent_model_based.module.run(action_count, cumulated_reward, reward_obtained, previous_state,
											   final_decision, current_state, who_plan_local[current_state]["MB"])

		decision_MF = agent_model_free.module.run(action_count, cumulated_reward, reward_obtained, previous_state, 
	 										   final_decision, current_state, who_plan_local[current_state]["MF"])

		decisions = {"MF": decision_MF, "MB": decision_MB}

		if winner == "MF":
			final_decision = decisions["MF"]
		elif winner == "MB":
			final_decision = decisions["MB"]	


	# ------- Print info in pbar description (TQDM library) ------------------
	pbar.set_description(
		 "Step: " + str(action_count) + ", Cumulated reward: " + str(cumulated_reward) + ", Winner: " +
		 winner  )


	# The previous state is now the old current state
	previous_state = current_state
	if options.add_wall == True:
		if previous_state == "7":
			path[1] += 1
		elif previous_state == "19":
			path[0] += 1

	# Simulate the effect of the robot's final decision on the environement and find the new current state
	reward_obtained, current_state = update_robot_position( environement, previous_state, final_decision)

	if (action_count == goalhaschange and options.change_goal == True) or (action_count == wallhaschange and
																		   options.add_wall == True):
		reward_obtained = 0
		goalhaschange=False
		wallhaschange=False

	# Update cumulated reward and counter of actions
	cumulated_reward += reward_obtained

	environement.action_count += 1

	return environement, selection_prob, meta_controller_system, who_plan_global, who_plan_local, agent_model_free,\
		   agent_model_based, current_state, cumulated_reward, reward_obtained, previous_state, final_decision,\
		   listWall,listGoal,switchGoal,switchWall,path


def StartExperiment(environement,experiment,options,parameters_agent_MF, parameters_agent_MB, beta_MC,map_file):
	agent_model_free_module_type = options.agent_model_free_module_type
	agent_model_based_module_type = options.agent_model_based_module_type
	print("\n \n ------- TURTLEBOT SIMULATOR EXPERIMENT ------- ", flush=True)
	print("Model free agent choice: " + agent_model_free_module_type, flush=True)
	print("Model based agent choice: " + agent_model_based_module_type + "\n", flush=True)

	boundaries_exp = {"max_reward": options.max_reward, "duration": options.duration,
					  "window_size": options.window_size, "epsilon": 0.01}
	options_log = {"log": options.log, "summary": options.summary}
	initial_variables = {"action_count": 0, "decided_action": 0, "actions_prob": 0.125,
						 "previous_state": str(environement.listDepartState[0]),
						 "current_state": str(environement.listDepartState[0]), \
	"qvalue": 1, "delta": 0.0, "plan_time": 0.0, "reward": 0, "replays": 0}
	action_space = environement.action_space
	state_space = environement.size
	criterion = options.criterion
	coeff_kappa = options.coeff_kappa
	
	listWall = list()
	listGoal = list()
	switchWall = 0
	switchGoal = 0

	if options.add_wall==True:
		switchWall=0
		listWall=ParsingWall(options.file_wall)
		swWtemp=10E1000
		for i in listWall:
			
			if i["iter"]>switchWall and i["iter"]<swWtemp:
				swWtemp=i["iter"]
		switchWall=swWtemp
	
	if options.change_goal==True:
		switchGoal=0
		
		listGoal= ParsingGoal(options.file_goal)
		swGtemp=10E1000
		for i in listGoal:
			
			if i["iter"]>switchGoal and i["iter"]<swGtemp:
				swGtemp=i["iter"]
		switchGoal=swGtemp
		
	agent_model_free = AgentModelFree(agent_model_free_module_type, experiment, map_file, initial_variables,
									  action_space, state_space, boundaries_exp, parameters_agent_MF, options_log)
	agent_model_based = AgentModelBased(agent_model_based_module_type, experiment, map_file, initial_variables,
										action_space, boundaries_exp, parameters_agent_MB, options_log)
	""" --- Those 2 lines below are useful for some algorithms 
	(when MF algorithms need MB model for replay phase for instance)
	"""
	agent_model_free.module.agent_model_based = agent_model_based
	agent_model_based.module.agent_model_free = agent_model_free

	meta_controller_system = MetaController(agent_model_based, agent_model_free,experiment, map_file, initial_variables,
											boundaries_exp, beta_MC, criterion, coeff_kappa, options_log)

	# Initialize parameters and variables for the loop of simulation
	environement.action_count = initial_variables["action_count"] + 1
	final_decision = initial_variables["decided_action"]
	previous_state = initial_variables["previous_state"]
	current_state = initial_variables["current_state"]
	duration = boundaries_exp["duration"]
	reward_obtained = initial_variables["reward"]
	max_reward = boundaries_exp["max_reward"]
	cumulated_reward = 0
	who_plan_global = {"MF": True, "MB": True}
	who_plan_local = {current_state: {"MF": True, "MB": True}} # who does the planning? -->

	# All will be set to False except 1
	path=[0]*2

	for i in range(2):
		path[i]=0

	pbar = tqdm(range(duration))

	for p in pbar:
		if(environement.action_count > duration) or (cumulated_reward > max_reward):
			print(Fore.GREEN + "MAXIMUM REWARD or MAXIMUM EXPERIMENT DURATION REACHED")
			print(Style.RESET_ALL)
			break

		environement, selection_prob, meta_controller_system, who_plan_global, who_plan_local, agent_model_free,\
		agent_model_based, current_state, cumulated_reward, reward_obtained, previous_state, final_decision,\
		listWall,listGoal,switchGoal,switchWall,path = oneCycle(environement,options, pbar, meta_controller_system,
																who_plan_global, who_plan_local, agent_model_free,
																agent_model_based, current_state, cumulated_reward,
																reward_obtained, previous_state, final_decision,
																listWall,listGoal,switchGoal,switchWall,path)

		# check if the agent has not found the reward in 10000 iterations
		if environement.action_count == 10000 and cumulated_reward == 0:
			print(Fore.GREEN + "!!! THE AGENT HAS NOT FOUND THE REWARD IN 10000 ITERATIONS !!!")
			print(Style.RESET_ALL)

			while cumulated_reward == 0:
				environement.action_count = 0
				cumulated_reward = 0
				previous_state = "0"
				final_decision = 0
				current_state = "0"
				who_plan_local = {current_state: {"MF": True, "MB": True}}
				meta_controller_system = MetaController(agent_model_based, agent_model_free, experiment, map_file,
														initial_variables, boundaries_exp, beta_MC, criterion,
														coeff_kappa, options_log)

				# Reset tensorflow graph & replay_cycles, replay_time
				agent_model_free.reset(experiment, map_file, initial_variables, action_space, state_space,
									   boundaries_exp, parameters_agent_MF, options_log)
				agent_model_based.reset()

				for i in range(10000): # as the tqdm bar will not be reset
					environement, selection_prob, meta_controller_system, who_plan_global, who_plan_local,\
					agent_model_free, agent_model_based, current_state, cumulated_reward, reward_obtained,\
					previous_state, final_decision, listWall,listGoal,switchGoal,switchWall,path =\
						oneCycle(environement,options, pbar, meta_controller_system, who_plan_global, who_plan_local,
								 agent_model_free, agent_model_based, current_state, cumulated_reward, reward_obtained,
								 previous_state, final_decision, listWall,listGoal,switchGoal,switchWall,path)

	return cumulated_reward
	
