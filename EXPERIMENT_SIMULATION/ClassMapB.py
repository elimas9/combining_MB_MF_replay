import numpy as np

class Environement:# class for the environement(map) that the agent will use

	def __init__(self,listState,reward,listDepartState,size,action_space):
		self.listState=listState 			  # listState= [state_0,state_1,state_2,...,state_size-2,state_size-1]
		self.reward=reward 					  # reward={"state":id,"value",n}
		self.listDepartState=listDepartState  # listDepartState[state_x.id,state_y.id,state_z.id]
		self.size=size				  		  #	number of states in the environement 
		self.action_space=action_space		  # number of actions the agent is able to do
		self.setWall=set()            		  # set of added walls during the experiment
		self.action_count=0           		  # counter for the current step in the experiment
		
	
	def chooseDepart(self):# function that returns a random initial state id from the list listDepartState
		depart = np.random.choice(self.listDepartState)
		return depart

	# function that returns a specific initial state id from the list listDepartState
	def chooseDepart_n(self,n): 
		depart = self.listDepartState[n]
		return depart


	def addWalluni(self, state1, state2):# function that adds a one-way wall (conveyor belt) between state1 and state2ats state1 et State2 (removing the transition between state1 and state2 )
		state1.removeTran(state2)
		self.setWall.add((state1.id,state2.id))

	# make ISIR great again
	def addWall(self, state1, state2):# function that adds a two-way wallbetween state1 and state2ats state1 et State2 (removing the transition between state1 and state2 and between state2 and state1 )
		self.addWalluni( state1, state2)
		self.addWalluni( state2, state1)


	def printEnvironementFileName(self,namefile):# function that prints the environement in a file in a json format
		f = open(namefile,"w")
		f.write('{\n\t"transitionActions":[' + "\n")
		for s in range(len(self.listState)):
			state = self.listState[s]
			f.write('\t\t{ "state":'+str(state.id)+',\n\t\t"transitions":['+ "\n")
			for i in range(self.action_space-1)  :
				action = state.ListAction[i]

				for transi in action :
					f.write('\t\t\t{ "action":'+ str(i)+ ', "state":'+ str(transi["state"])+ ', "prob":'+str(transi["prob"]) +'},\n')

			action = state.ListAction[self.action_space-1]
			f.write('\t\t\t{ "action":'+ str(i+1)+ ', "state":'+ str(action[len(action)-1]["state"])+ ', "prob":'+str( action[len(action)-1]["prob"])+'}]\n')
			if(len(self.listState)-1 == s):
				f.write("\t\t}]\n")
			else :
				f.write("\t\t},\n")

		f.write("}\n")	
		f.close()
				
		
	def printEnvironementFile(self, name_environment):# function that prints the environement in a file in a json format
		
		f = open(name_environment + ".txt", "w")
		f.write('{\n\t"transitionActions":[' + "\n")
		for s in range(len(self.listState)):
			state = self.listState[s]
			f.write('\t\t{ "state":'+str(state.id)+',\n\t\t"transitions":['+ "\n")

			for i in range(self.action_space-1)  :
				action = state.ListAction[i]
		
				for transi in action :
					f.write('\t\t\t{ "action":'+ str(i)+ ', "state":'+ str(transi["state"])+ ', "prob":'+str(transi["prob"]) +'},\n')
				
			action = state.ListAction[self.action_space-1]
			f.write('\t\t\t{ "action":'+ str(i+1)+ ', "state":'+ str(action[len(action)-1]["state"])+ ', "prob":'+str( action[len(action)-1]["prob"])+'}]\n')

			if(len(self.listState)-1 == s):
				f.write("\t\t}]\n")
			else :
				f.write("\t\t},\n")

				
		f.write("}\n")	
		f.close()
				

   
