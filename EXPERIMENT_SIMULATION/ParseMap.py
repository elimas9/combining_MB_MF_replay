from ClassState import *
import json

def Buildlist_state(f,action_space):# function that creates listStates from a json-compatible file (as explained in the readme)
	f = open(f, 'r')
	arena = json.load(f)
	nbStates = len(arena["transitionActions"])
	
	listS=[list()] * nbStates#listState
	for state in arena["transitionActions"]:
		id=state["state"]
		listT=[list()] * action_space #listTransition
		for transition in state["transitions"]:
			listT[transition["action"]]=listT[transition["action"]]+[{"state":transition["state"], "prob":transition["prob"]}]
		listS[id]=State(id,listT)
	return listS

def ParsingWall(f):# function that returns a list of walls to be added during the experiment from a json-compatible file (Walls.txt)
	f = open(f, 'r')
	listW=list()
	listW = json.load(f)
	return listW 
	
def ParsingGoal(f):# function that returns a list of rewards to be added during the experiment from a json-compatible file (Goals.txt) 
	f = open(f, 'r')
	listW=list()
	listW = json.load(f)
	return listW 
	