'''
This script permits with the other ones in the folder to simulate an Erwan Renaudo's 
experiment : 
"Renaudo, E. (2016). Des comportements flexibles aux comportements habituels: 
Meta-apprentissage neuro-inspiré pour la robotique autonome (Doctoral dissertation, 
Université Pierre et Marie Curie (Paris 6))". 

This is the Model Based agent. It makes a connection with a model-based module, ie the chosen learning algorithm.
'''

from MB_modules.ValueIterationShuffle import *
from MB_modules.ValueIterationShuffleBudget import *

VERSION = 1


class AgentModelBased:
	"""
	This class implements a model-based agent.
	"""

	def __init__(self, agent_model_based_module_type, experiment, map_file, initial_variables, action_space,
				 boundaries_exp, parameters_agent_MB, options_log):
		"""
		Initialise values and learning algorithm.
		"""
		self.module_type = agent_model_based_module_type

		# ----- initialization of the model based agent, with the appropriate module --------

		if self.module_type == 'value-iteration-shuffle':
			self.module = ValueIterationShuffle(experiment, map_file, initial_variables, action_space,
												boundaries_exp, parameters_agent_MB, options_log)

		elif self.module_type == 'value-iteration-shuffle-budget':
			self.module = ValueIterationShuffleBudget(experiment, map_file, initial_variables, action_space,
													  boundaries_exp, parameters_agent_MB, options_log)



	def reset(self):

		self.module.infer_cycles = 0

		if self.module_type in ['value-iteration-shuffle-replay-budget-time-win-a'
								'value-iteration-shuffle-replay-budget-time-win-b',
								"value-iteration-shuffle-bi-replay-budget-time-win" ]:
			self.module.replay_cycles = 0
			self.module.replay_time = 0

			self.module.bi_replay_cycles = 0
			self.module.bi_replay_time = 0



