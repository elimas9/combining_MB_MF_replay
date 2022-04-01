from turtlebotSimulator import *

# start program by terminal command as indicated in the readme
def main():
	experiment, map_file, key_states_file, parameters_file, options = manage_arguments() # handles and returns the
	# options chosen by the terminal command

	parameters_agent_MF, parameters_agent_MB, beta_MC = parse_parameters(parameters_file)
	#handles and returns the MF MB et MC parameters from the corresponding text file (generally parameter.txt)

	action_space = 8

	for i in range(int(experiment)):

		environement = initialize_environement_from_file("realisticWorld",key_states_file,action_space)
		StartExperiment(environement,i,options,parameters_agent_MF, parameters_agent_MB, beta_MC,map_file)


if __name__ == "__main__":
	main()