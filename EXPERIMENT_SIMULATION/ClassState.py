import numpy as np

class State:# a sub-structure of the environeent where the agent can be

	def __init__(self,id,ListAction):
		self.id= id 				# state identifier; it also represents their index in the listState 
		self.ListAction=ListAction  # list that contains the transition between states; every list index represents the action number, every item is a list of transitions
									# a transition is represented by a dictionary from State1 : {prob:x,State_suiv: State2} 
									# : [[{prob:x,State_suiv:sx},...,{prob:x,State_suiv:sy}],...,[{prob:x,State_suiv:sz},...,{prob:x,State_suiv:sa}]] list of list of dict
									#	 |_____________________action1_____________________| ... |_____________________actionN_____________________|							action n

	def chooseState(self,action): # function that chooses the next action for the agent to execute
		tab_act=list()
		packet_etat_suiv = list()
		for i in self.ListAction[action] :# search in the action list the probabilities of reaching a certain state
			nbChance=i["prob"]
			packet_etat_suiv=[i["state"]] * nbChance # multiplying the state number from a list by the probability of reaching that state 
			tab_act=tab_act+packet_etat_suiv # storing the value previously found in a tab 
		arrival = np.random.choice(tab_act) # randomly chosing the next state from the list
		return arrival   
	
		
   
		
		
	
	# def removeTran(self,toState):# removing the transition between the current State (self) and the toState (unilateral)
	# 	for action in range(len(self.ListAction)):
	# 		for transi in self.ListAction[action]:
	# 			if transi["state"]==toState.id:
	# 				self.ListAction[action].remove(transi)
	
	def removeTran(self,toState):# removing the transition between the current State (self) and the toState (unilateral)
		for action in range(len(self.ListAction)):
			for transi in self.ListAction[action]:
				if transi["state"]==toState.id:
					transi["state"]=self.id