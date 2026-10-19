# South of HKUST Campus
# This code is used to plot the throughput and idle rate of different methods
import matplotlib.pyplot as plt

x = [1, 5, 10, 15, 20, 25, 30]
Y_rule_thro = [302.7, 517.8, 1519.62, 1457.68, 1458, 1458, 1458]
Y_llm_thro = [15, 100, 495, 730, 805, 1055, 1160]
Y_agent_thro = [227.03, 410.35, 1040.1, 1343.81, 1335, 1278, 1329.10]

# idle rate
Y_rule_idle = [88.89, 66.67, 8.32,0, 0, 0, 0]
Y_llm_idle = [99.2, 93.33, 70.85, 54.17, 50, 33.33, 27.5]
Y_agent_idle = [84.49, 74.17, 36.67, 0, 0, 0, 0]

# plot
fig, ax1 = plt.subplots(figsize=(8, 6))
ax2 = ax1.twinx()

# color map
COLOR_MAP = {
    'Rule-based': 'blue',  
    'Prompt-based': 'blue', 
    'WirelessAgent': 'blue' 
}


ax1.plot(x, Y_rule_thro, color=COLOR_MAP['Rule-based'], linestyle='-', 
        linewidth=2, marker='o', markersize=9, label='_nolegend_')
ax1.plot(x, Y_agent_thro, color=COLOR_MAP['WirelessAgent'], linestyle='--',
        linewidth=2, marker='^', markersize=9, label='_nolegend_')
ax1.plot(x, Y_llm_thro, color=COLOR_MAP['Prompt-based'], linestyle=':',
        linewidth=2, marker='s', markersize=9, label='_nolegend_')

# color map
COLOR_MAP = {
    'Rule-based': 'black',  
    'Prompt-based': 'black', 
    'WirelessAgent': 'black' 
}

ax2.plot(x, Y_rule_idle, color=COLOR_MAP['Rule-based'], linestyle='-',
        linewidth=2, marker='o', markersize=9, label='Rule-based')
ax2.plot(x, Y_agent_idle, color=COLOR_MAP['WirelessAgent'], linestyle='--',
        linewidth=2, marker='^', markersize=9, label='WirelessAgent')
ax2.plot(x, Y_llm_idle, color=COLOR_MAP['Prompt-based'], linestyle=':',
        linewidth=2, marker='s', markersize=9, label='Prompt-based')


ax1.set_xlabel('Number of Users', fontsize=20)
ax1.set_ylabel('Total Throughput (Mbps)',  color='b', fontsize=20)
ax2.set_ylabel('Bandwidth Idle Rate (%)', fontsize=20)


handles, labels = ax2.get_legend_handles_labels()
legend = ax1.legend(handles, labels, 
                  loc='upper left',
                  bbox_to_anchor=(5/10, 3/10),
                  frameon=True,
                  fontsize=18,
                  ncol=1,
                  borderpad=0.8,
                  handletextpad=0.8)



ax1.add_artist(legend)

plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
ax1.set_xlim(0, 31)
ax1.grid(True, linestyle=':', alpha=0.7)
ax2.set_ylim(0, 100)
plt.tight_layout()

plt.show()