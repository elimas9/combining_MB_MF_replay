import matplotlib.pyplot as plt
import numpy as np
import matplotlib_latex_bridge as mlb
import scipy.spatial as scispa
import operator

nb_iterations = 4000
nb_experiments = 50
nb_experiments2 = 50
nb_vi = 50


def getMeanVarReward(experiences):
	y_all = list()  # cumulated reward
	number_refused = 0
	for expe in experiences:
		y = list()  # cumulated reward list for this experiment
		cumulatedReward = 0
		l = 0  # count of lines
		with open(expe, 'r') as file1:
			for line in file1:
				# Verification to avoid bugs when there is 2 instances for 0
				if int(line.split(" ")[0]) == 0 and l == 1:
					pass
				else:
					reward = int(line.split(" ")[2])
					cumulatedReward += reward
					y.append(cumulatedReward)

					l += 1

				if l == nb_iterations:
					break

		if l == nb_iterations:
			y_all.append(np.array(y))
		else:
			print(expe+ " was not added, problem with the log ("+str(l)+"/"+str(nb_iterations)+" lines only)")
			number_refused += 1

	y_all_array = np.array(y_all)

	yMean = np.mean(y_all_array, axis=0)
	yStatistics = np.percentile(y_all_array[:, -1], [25, 50, 75], axis=0)
	yMedian = np.median(y_all_array, axis=0)
	ySD = np.percentile(y_all_array, [25, 50, 75], axis=0)
	print(" \n NUMBER OF REFUSED DOCS: ", number_refused)

	return yMean[-1], yStatistics, yMedian, ySD


def getMeanVarTime(experiences):
	y_all = list()  # cumulated reward
	for expe in experiences:
		y = list()  # cumulated reward list for this experiment
		cumulatedTime = 0
		l = 0  # count of lines
		with open(expe, 'r') as file1:
			for line in file1:
				# Verification to avoid bugs when there is 2 instances for 0
				if int(line.split(" ")[0]) == 0 and l == 1:
					pass
				else:
					time = float(line.split(" ")[3])
					cumulatedTime += time
					y.append(cumulatedTime)
					l += 1

				if l == nb_iterations:
					break

		if l == nb_iterations:
			y_all.append(np.array(y))
		else:
			print(expe+ " was not added, problem with the log ("+str(l)+"/"+str(nb_iterations)+" lines only)")

	y_all_array = np.array(y_all)

	yMean = np.mean(y_all_array, axis=0)
	yStatistics = np.percentile(y_all_array[:, -1], [25, 50, 75], axis=0)
	yMedian = np.median(y_all_array, axis=0)
	ySD = np.percentile(y_all_array, [25, 50, 75], axis=0)

	return yMean[-1], yStatistics, yMedian, ySD


def compute_dist_optim_point(x, y, opt_point=[0, 336.67]):
	return np.linalg.norm(np.array([x, y]) - np.array(opt_point))


def compute_dist_optim_point_cheb(x, y, opt_point=[0, 336.67]):
	return scispa.distance.chebyshev(np.array(opt_point), np.array([x, y]))


experiences_dec_b5= list()
for i in range(0, nb_vi - 1):
	path = "../EXPERIMENT_SIMULATION/logs/MFreplay/MF_only_MF_[q-learning-replay]_coeff1.0_exp"+str(i)+"_log.dat"
	experiences_dec_b5.append(path)
r_dec_50, stat_dec_50, _, s_dec_50 = getMeanVarReward(experiences_dec_b5)
c_dec_50, c_stat_dec_50, _, c_s_dec_50 = getMeanVarTime(experiences_dec_b5)


experiences_dec_200_100= list()
for i in range(0, nb_vi - 1):
	path = "../EXPERIMENT_SIMULATION/logs/MFreplayb200/MF_only_MF_[q-learning-replay-budget]" \
		   "(replay_budget_200)_coeff1.0_exp"+str(i)+"_log.dat"
	experiences_dec_200_100.append(path)
r_dec_200_100, stat_dec_200_100, _, s_dec_200_100 = getMeanVarReward(experiences_dec_200_100)
c_dec_200_100, c_stat_dec_200_100, _, c_s_dec_200_100  = getMeanVarTime(experiences_dec_200_100)


experiences_dec_et50_50= list()
for i in range(0, nb_vi - 1):
	path = "../EXPERIMENT_SIMULATION/logs/ET_MFreplayb100+MBb100/" \
		   "Entropy_and_time_MB_[value-iteration-shuffle-budget](budget_100)_MF_[q-learning-replay-budget]" \
		   "(replay_budget_100)_window30_coeff1.0_exp"+str(i)+"_log.dat"
	experiences_dec_et50_50.append(path)
r_dec_et_50_50, stat_dec_et_50_50, _, s_dec_et_50_50 = getMeanVarReward(experiences_dec_et50_50)
c_dec_et_50_50, c_stat_dec_et_50_50, _, c_s_dec_et_50_50 = getMeanVarTime(experiences_dec_et50_50)


experiences_ql_3 = list()
for i in range(0, nb_vi - 1):
	path = "../EXPERIMENT_SIMULATION/logs/MF/MF_only_MF_[q-learning]_coeff1.0_exp"+str(i)+"_log.dat"
	experiences_ql_3.append(path)
r_ql_3, stat_ql_3, _, s_ql_3  = getMeanVarReward(experiences_ql_3)
c_ql_3, c_stat_ql_3, _, c_s_ql_3 = getMeanVarTime(experiences_ql_3)


experiences_et = list()
for i in range(0, nb_vi - 1):
	path = "../EXPERIMENT_SIMULATION/logs/ET/Entropy_and_time_MB_[value-iteration-shuffle]_MF_[q-learning]_" \
		   "window30_coeff1.0_exp"+str(i)+"_log.dat"
	experiences_et.append(path)
r_et, stat_et, _, s_et = getMeanVarReward(experiences_et)
c_et, c_stat_et, _, c_s_et = getMeanVarTime(experiences_et)


experiences_et_mb200 = list()
for i in range(0, nb_vi - 1):
	path = "../EXPERIMENT_SIMULATION/logs/ET_MF+MBb200/" \
		   "Entropy_and_time_MB_[value-iteration-shuffle-budget](budget_200)_MF_[q-learning]_" \
		   "window30_coeff1.0_exp"+str(i)+"_log.dat"
	experiences_et_mb200.append(path)
r_et_mb200, stat_et_mb200, _, s_et_mb200 = getMeanVarReward(experiences_et_mb200)
c_et_mb200, c_stat_et_mb200, _, c_s_et_mb200 = getMeanVarTime(experiences_et_mb200)


experiences_mb = list()
for i in range(0, nb_vi - 1):
	path = "../EXPERIMENT_SIMULATION/logs/MB/MB_only_MB_[value-iteration-shuffle]_window30_" \
		   "coeff1.0_exp"+str(i)+"_log.dat"
	experiences_mb.append(path)
r_mb, stat_mb, _, s_mb = getMeanVarReward(experiences_mb)
c_mb, c_stat_mb, _, c_s_mb = getMeanVarTime(experiences_mb)


experiences_mb_b200 = list()
for i in range(0, nb_vi - 1):
	path ="../EXPERIMENT_SIMULATION/logs/MBb200/MB_only_MB_[value-iteration-shuffle-budget](budget_200)_" \
		  "window30_coeff1.0_exp"+str(i)+"_log.dat"
	experiences_mb_b200.append(path)
r_mb_b200, stat_mb_b200, _, s_mb_b200 = getMeanVarReward(experiences_mb_b200)
c_mb_b200, c_stat_mb_b200, _, c_s_mb_b200 = getMeanVarTime(experiences_mb_b200)


print(" \n  -------> Log processing done ")

def plot_pareto_frontier(Xs, Ys, minX=0, maxY=673.33, ax=None):
    '''Pareto frontier selection process'''
    sorted_list = sorted([[Xs[i], Ys[i]] for i in range(len(Xs))], reverse=minX)
    pareto_front = [sorted_list[0]]
    for pair in sorted_list[1:]:
        if maxY:
            if pair[1] >= pareto_front[-1][1]:
                pareto_front.append(pair)
        else:
            if pair[1] <= pareto_front[-1][1]:
                pareto_front.append(pair)

    '''Plotting process'''
    pf_X = [pair[0] for pair in pareto_front]
    pf_Y = [pair[1] for pair in pareto_front]
    plt.plot(pf_X, pf_Y, color='grey', linestyle=(0, (0.1, 2)), dash_capstyle='round', linewidth=2)


# 2D overview plot
mlb.setup_page(textwidth=6.97522, columnwidth=3.36305, fontsize=10)

ms = 7

# fig = mlb.figure_textwidth(height=6.97522*0.8+mlb.get_default_figsize()[1]*0.48)
mlb.figure(width=6.97522*0.8, height=6.97522*0.8)

# detect the max and min values for the performances and the inference cost among the median of the tested algorithms
max_perf = max([stat_ql_3[1], stat_dec_50[1], stat_dec_200_100[1], stat_mb[1], stat_mb_b200[1], stat_et[1],
			   stat_et_mb200[1], stat_dec_et_50_50[1]])
min_perf = min([stat_ql_3[1], stat_dec_50[1], stat_dec_200_100[1], stat_mb[1], stat_mb_b200[1], stat_et[1],
			   stat_et_mb200[1], stat_dec_et_50_50[1]])
max_cost = max([c_stat_ql_3[1], c_stat_dec_50[1], c_stat_dec_200_100[1], c_stat_mb[1], c_stat_mb_b200[1], c_stat_et[1],
			   c_stat_et_mb200[1], c_stat_dec_et_50_50[1]])
min_cost = min([c_stat_ql_3[1], c_stat_dec_50[1], c_stat_dec_200_100[1], c_stat_mb[1], c_stat_mb_b200[1], c_stat_et[1],
			   c_stat_et_mb200[1], c_stat_dec_et_50_50[1]])

# structure all the performances and the costs in 3 dictionaries
all_performances = {"MF": stat_ql_3, "MF - replay budget inf": stat_dec_50,
					"MF - replay budget 200": stat_dec_200_100, "MB - inference budget inf": stat_mb,
					"MB - inference budget 200": stat_mb_b200, "MB - inference budget inf + MF": stat_et,
					"MB - inference budget 200 + MF": stat_et_mb200,
					"MB - inference budget 100 + MF - replay budget 100": stat_dec_et_50_50}
all_costs = {"MF": c_stat_ql_3, "MF - replay budget inf": c_stat_dec_50, "MF - replay budget 200": c_stat_dec_200_100,
			 "MB - inference budget inf": c_stat_mb, "MB - inference budget 200": c_stat_mb_b200,
			 "MB - inference budget inf + MF": c_stat_et, "MB - inference budget 200 + MF": c_stat_et_mb200,
			 "MB - inference budget 100 + MF - replay budget 100": c_stat_dec_et_50_50}

# normalize the performances and the costs
norm_perf = {}
norm_cost = {}
for perf in all_performances.keys():
	norm_perf[perf] = [0] * len(all_performances[perf])
	for idp, st in enumerate(all_performances[perf]):
		norm_perf[perf][idp] = (1 - 0) / (max_perf - min_perf) * (st - max_perf) + 1
for cost in all_costs.keys():
	norm_cost[cost] = [0] * len(all_costs[cost])
	for idc, stc in enumerate(all_costs[cost]):
		norm_cost[cost][idc] = (1 - 0) / (max_cost - min_cost) * (stc - max_cost) + 1

# definition of the optimal point
opt_point = [0, 1]
print(f'opt_point: {opt_point}')

# ranking the performance of the algorithm
cheb_dist = {}
for ke in all_performances.keys():
	cheb_dist[ke] = compute_dist_optim_point_cheb(norm_cost[ke][1], norm_perf[ke][1], opt_point=opt_point)
print(f'cheb_dist: {cheb_dist}')

rank_cheb_dist = dict(sorted(cheb_dist.items(), key=operator.itemgetter(1))).keys()
print(f'rank_cheb_dist: {rank_cheb_dist}')


plt.xlabel("Cumulative inference cost (a.u.)")
plt.ylabel("Cumulative reward (a.u.)")

colors = {"MF": "pink", "MF - replay budget inf": "blueviolet", "MF - replay budget 200": "violet",
		  "MB - inference budget inf": "black", "MB - inference budget 200": "darkblue",
		  "MB - inference budget inf + MF": "gold", "MB - inference budget 200 + MF": "orangered",
		  "MB - inference budget 100 + MF - replay budget 100": "red"}
shapes = {"MF": "s", "MF - replay budget inf": "s", "MF - replay budget 200": "s", "MB - inference budget inf": "v",
		  "MB - inference budget 200": "v", "MB - inference budget inf + MF": "D",
		  "MB - inference budget 200 + MF": "D", "MB - inference budget 100 + MF - replay budget 100": "D"}

plot_pareto_frontier([nc[1] for nc in norm_cost.values()], [np[1] for np in norm_perf.values()])
plt.scatter(0, 1, c='yellow', marker='*', linewidths=1, edgecolor='black', s=50)
plt.annotate('optimal point', (0.03, 1))
texts = {"MF": 1, "MF - replay budget inf": 0.52, "MF - replay budget 200": 0.52, "MB - inference budget inf": 1,
		 "MB - inference budget 200": 0.48, "MB - inference budget inf + MF": 0.53,
		 "MB - inference budget 200 + MF": 0.5, "MB - inference budget 100 + MF - replay budget 100": 0.3}
offset_text = {"MF": (0.03, 0), "MF - replay budget inf": (0.03, 0), "MF - replay budget 200": (-0.1, 0),
			   "MB - inference budget inf": (0.03, 0.04), "MB - inference budget 200": (-0.1, 0.03),
			   "MB - inference budget inf + MF": (0.07, 0), "MB - inference budget 200 + MF": (0.05, 0),
			   "MB - inference budget 100 + MF - replay budget 100": (0.05, 0)}

for algo in norm_cost.keys():
	plt.plot(norm_cost[algo][1], norm_perf[algo][1], shapes[algo], c=colors[algo], markersize=ms, label=algo)
	plt.vlines(norm_cost[algo], norm_perf[algo][0], norm_perf[algo][2], colors=colors[algo])
	plt.hlines(norm_perf[algo], norm_cost[algo][0], norm_cost[algo][2], colors=colors[algo])
	plt.annotate(texts[algo], (norm_cost[algo][1] + offset_text[algo][0], norm_perf[algo][1] + offset_text[algo][1]))

plt1 = plt.gca()
plt1.set_aspect('equal', 'box')

leg = plt.legend()
plt.show()


# cumulative reward plot
mlb.figure_textwidth(widthp=0.48)

x_all = list([i for i in range(nb_iterations)])  # action count list, used for every experiment

plt.plot(x_all, s_ql_3[1, :], c="pink", label="MF")
plt.fill_between(x_all, s_ql_3[0, :], s_ql_3[2, :], color="pink", alpha=0.1)
plt.plot(x_all[-1], stat_ql_3[1],"s", c="pink", markersize=ms, label="MF")

plt.plot(x_all, s_dec_50[1, :], c="blueviolet", label="MF - replay budget inf")
plt.fill_between(x_all, s_dec_50[0, :], s_dec_50[2, :], color="blueviolet", alpha=0.1)
plt.plot(x_all[-1], stat_dec_50[1],"s", c="blueviolet", markersize=ms, label = "MF - replay budget inf ")

plt.plot(x_all, s_dec_200_100[1, :], c="violet", label="MF - replay budget 200")
plt.fill_between(x_all, s_dec_200_100[0, :], s_dec_200_100[2, :], color="violet", alpha=0.1)
plt.plot(x_all[-1], stat_dec_200_100[1],"s", c="violet", markersize=ms, label="MF - replay budget 200 ")

plt.plot(x_all, s_mb[1, :], c="black", label="MB - inference budget inf")
plt.fill_between(x_all, s_mb[0, :], s_mb[2, :], color="black", alpha=0.1)
plt.plot(x_all[-1], stat_mb[1],"v", c="black", markersize=ms, label="MB - inference budget inf")

plt.plot(x_all, s_mb_b200[1, :], c="darkblue", label="MB - inference budget 200")
plt.fill_between(x_all, s_mb_b200[0, :], s_mb_b200[2, :], color="darkblue", alpha=0.1)
plt.plot(x_all[-1], stat_mb_b200[1],"v", c="darkblue", markersize=ms, label="MB - inference budget 200")

plt.plot(x_all, s_et[1, :], c="gold", label="MB - inference budget inf + MF")
plt.fill_between(x_all, s_et[0, :], s_et[2, :], color="gold", alpha=0.1)
plt.plot(x_all[-1], stat_et[1], "D", c="gold", markersize=ms, label="MB - inference budget inf + MF")

plt.plot(x_all, s_et_mb200[1, :], c="orangered", label="MB - inference budget 200 + MF")
plt.fill_between(x_all, s_et_mb200[0, :], s_et_mb200[2, :], color="orangered", alpha=0.1)
plt.plot(x_all[-1], stat_et_mb200[1],"D", c="orangered", markersize=ms, label="MB - inference budget 200 + MF")

plt.plot(x_all, s_dec_et_50_50[1, :], c="red", label="MB - inference budget 100 + MF - replay budget 100")
plt.fill_between(x_all, s_dec_et_50_50[0, :], s_dec_et_50_50[2, :], color="red", alpha=0.1)
plt.plot(x_all[-1], stat_dec_et_50_50[1],"D", c="red", markersize=ms, label = "MB - inference budget 100 + "
																			  "MF - replay budget 100")

plt.xlabel("Number of actions")
plt.axvline(x=1600, c="black", linewidth=2, label="Change of the reward state")
plt.ylabel("Cumulative reward (a.u.)")
plt.xticks(np.arange(0, nb_iterations+1, 500))


# cumulative inference cost plot

mlb.figure_textwidth(widthp=0.48)

plt.plot(x_all, c_s_ql_3[1, :], c="pink", label="MF")
plt.fill_between(x_all, c_s_ql_3[0, :], c_s_ql_3[2, :], color="pink", alpha=0.1)
plt.plot(x_all[-1], c_stat_ql_3[1],"s", c="pink", markersize=ms, label="MF")

plt.plot(x_all, c_s_dec_50[1, :], c="blueviolet", label="MF - replay budget inf")
plt.fill_between(x_all, c_s_dec_50[0, :], c_s_dec_50[2, :], color="blueviolet", alpha=0.1)
plt.plot(x_all[-1], c_stat_dec_50[1],"s", c="blueviolet", markersize=ms, label="MF - replay budget inf ")

plt.plot(x_all, c_s_dec_200_100[1, :], c="violet", label="MF - replay budget 200")
plt.fill_between(x_all, c_s_dec_200_100[0, :], c_s_dec_200_100[2, :], color="violet", alpha=0.1)
plt.plot(x_all[-1], c_stat_dec_200_100[1],"s", c="violet", markersize=ms, label = "MF - replay budget 200 ")

plt.plot(x_all, c_s_mb[1, :], c="black", label="MB - inference budget inf")
plt.fill_between(x_all, c_s_mb[0, :], c_s_mb[2, :], color="black", alpha=0.1)
plt.plot(x_all[-1], c_stat_mb[1],"v", c="black", markersize=ms, label="MB - inference budget inf")

plt.plot(x_all, c_s_mb_b200[1, :], c="darkblue", label="MB - inference budget 200")
plt.fill_between(x_all, c_s_mb_b200[0, :], c_s_mb_b200[2, :], color="darkblue", alpha=0.1)
plt.plot(x_all[-1], c_stat_mb_b200[1],"v", c="darkblue", markersize=ms, label="MB - inference budget 200")

plt.plot(x_all, c_s_et[1, :], c="gold", label="MB - inference budget inf + MF")
plt.fill_between(x_all, c_s_et[0, :], c_s_et[2, :], color="gold", alpha=0.1)
plt.plot(x_all[-1], c_stat_et[1], "D", c="gold", markersize=ms, label="MB - inference budget inf + MF")

plt.plot(x_all, c_s_et_mb200[1, :], c="orangered", label="MB - inference budget 200 + MF")
plt.fill_between(x_all, c_s_et_mb200[0, :], c_s_et_mb200[2, :], color="orangered", alpha=0.1)
plt.plot(x_all[-1], c_stat_et_mb200[1],"D", c="orangered", markersize=ms, label="MB - inference budget 200 + MF")

plt.plot(x_all, c_s_dec_et_50_50[1, :], c="red", label="MB - inference budget 100 + MF - replay budget 50")
plt.fill_between(x_all, c_s_dec_et_50_50[0, :], c_s_dec_et_50_50[2, :], color="red", alpha=0.1)
plt.plot(x_all[-1], c_stat_dec_et_50_50[1],"D", c="red", markersize=ms,
		 label="MB - inference budget 100 + MF - replay budget 100")

plt.xlabel("Number of actions")
plt.axvline(x=1600, c="black", linewidth=2, label="Change of the reward state")
plt.ylabel("Cumulative inference cost (s)")
plt.xticks(np.arange(0, nb_iterations+1, 500))

plt.show()
