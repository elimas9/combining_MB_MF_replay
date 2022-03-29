
'''
This script permits with the other ones in the folder to simulate an Erwan Renaudo's 
experiment : 
"Renaudo, E. (2016). Des comportements flexibles aux comportements habituels: 
Meta-apprentissage neuro-inspiré pour la robotique autonome (Doctoral dissertation, 
Université Pierre et Marie Curie (Paris 6))". 

This is the Model Free agent. It makes a connection with a model-free module, ie the chosen learning algorithm.
'''

from MF_modules.QLearning import *
from MF_modules.QLearningReplay import *
from MF_modules.QLearningReplayBudget import *


VERSION = 1


class AgentModelFree:
	"""
	This class implements a model-free agent.
    """

	def __init__(self, agent_model_free_module, experiment, map_file, initial_variables, action_space,
				 state_space, boundaries_exp, parameters_agent_MF, options_log):
		"""
		Initialize values and learning algorithm
		"""
		self.module_type = agent_model_free_module

		# ----- initialization of the model based agent, with the appropriate module --------

		if self.module_type == 'q-learning':
			self.module = QLearning(experiment, map_file, initial_variables, boundaries_exp, parameters_agent_MF,
									options_log)

		elif self.module_type == 'q-learning-replay':
			self.module = QLearningReplay(experiment, map_file, initial_variables, action_space, boundaries_exp,
										  parameters_agent_MF, options_log)

		elif self.module_type == 'q-learning-replay-budget':
			self.module = QLearningReplayBudget(experiment, map_file, initial_variables, action_space, boundaries_exp,
												parameters_agent_MF, options_log)


	def reset(self):
		if self.module_type in ['q-learning-replay', 'q-learning-replay-budget']:
			self.module.replay_cycles = 0
			self.module.replay_time = 0
