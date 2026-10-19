import matplotlib
import matplotlib.pyplot as plt
import numpy as np

x_all = np.arange(1,26)

# Extended rate arrays to include data for positions 25 and 30 (copying values from position 20)
rate_eMBB_agent = np.array([
    [20, 20, 20, 20, 20, 20],
    [0, 10, 10, 10, 10, 10],
    [0, 20, 7, 7, 7, 7],
    [0, 0, 13, 13, 13, 13],
    [0, 0, 20, 20, 20, 20],
    [0, 0, 20, 20, 20, 20],
])

rate_uRLLC_agent = np.array([
    [5,5,5,3,3,2],
    [5,5,4,3,2,2],
    [1,1,1,1,1,1],
    [5,5,5,4,3,2],
    [0,5,5,5,5,4],
    [0,5,5,5,5,4],
    [0,1,1,1,1,1],
    [0,0,3,3,3,3],
    [0,0,1,1,1,1],
    [0,0,0,1,1,1],
    [0,0,0,1,1,1],
    [0,0,0,1,1,1],
    [0,0,0,1,1,1],
    [0,0,0,0,1,1],
    [0,0,0,0,1,1],
    [0,0,0,0,0,1],
    [0,0,0,0,0,1],
    [0,0,0,0,0,1],
    [0,0,0,0,0,1]
])
#==========================Optimal allocation==========================
rate_eMBB_naive = np.array([
    [20, 20, 15, 12.84, 12.84, 12.84],
    [0, 20, 15, 10.38, 10.38, 10.38],
    [0, 20, 15, 10.38, 10.38, 10.38],
    [0, 0, 15, 11.58, 11.58, 11.58],
    [0, 0, 15, 14.34, 14.34, 14.34],
    [0, 0, 15, 14.34, 14.34, 14.34],
    [0, 0, 0, 16.15, 16.15, 16.15],
])

rate_uRLLC_naive = np.array([
    [5,4.29,3.3,2.31,2,1.58],
    [5,4.29,3.3,2.31,2,1.58],
    [5,4.29,3.3,2.31,2,1.58],
    [5,4.29,3.3,2.31,2,1.58],
    [0,4.29,3.3,2.31,2,1.58],
    [0,4.29,3.3,2.31,2,1.58],
    [0,4.29,3.3,2.31,2,1.58],
    [0,0,3.3,2.31,2,1.58],
    [0,0,3.3,2.31,2,1.58],
    [0,0,0,2.31,2,1.58],
    [0,0,0,2.31,2,1.58],
    [0,0,0,2.31,2,1.58],
    [0,0,0,2.31,2,1.58],
    [0,0,0,0,2,1.58],
    [0,0,0,0,2,1.58],
    [0,0,0,0,0,1.58],
    [0,0,0,0,0,1.58],
    [0,0,0,0,0,1.58],
    [0,0,0,0,0,1.58]
])

#=================================Prompt-based method===================================
# Create data for LLM-based method 
rate_eMBB_llm = np.array([
    [10, 10, 10, 10, 10, 10],
    [0, 10, 10, 10, 10, 10],
    [0, 10, 10, 10, 10, 10],
    [0, 0, 10, 10, 10, 10],
    [0, 0, 0, 0, 10, 10],
    [0, 0, 0, 0, 10, 10],
])

rate_uRLLC_llm = np.array([
    [5,5,5,5,5,5],
    [5,5,5,5,5,5],
    [5,5,5,5,5,5],
    [3,3,3,3,3,3],
    [0,3,3,3,3,3],
    [0,3,3,3,3,3],
    [0,3,3,3,3,3],
    [0,0,3,3,3,3]
])
#rate_eMBB_llm = rate_eMBB_agent.copy()
#rate_uRLLC_llm = rate_uRLLC_agent.copy()

# Create figure with more width
plt.figure(figsize=(12, 4))

# Define x positions for side-by-side bars with more space between
bar_width = 0.75
gap = 1  # Gap between bars
x_positions = np.array([5, 10, 15, 20, 25, 30])  # Added 25 and 30
x_agent = x_positions 
x_llm = x_positions - bar_width - gap/2 
x_naive = x_positions + bar_width + gap/2

# Define colormaps
cmap_eMBB_agent = plt.colormaps['Greens']
cmap_uRLLC_agent = plt.colormaps['Greens']

cmap_eMBB_naive = plt.colormaps['Blues'] #optimal Rdpu
cmap_uRLLC_naive = plt.colormaps['Blues']

cmap_eMBB_llm = plt.colormaps['Greys'] #prompt YlOrBr
cmap_uRLLC_llm = plt.colormaps['Greys']

inner_colors_eMBB_agent = cmap_eMBB_agent([0,20,40,60,80,100,120,140,160,180,200])
inner_colors_uRLLC_agent = cmap_uRLLC_agent([0,10,20,30,40,50,60,70,80,90,100, 110,120,130,140,150,160,170,180,190,200])
inner_colors_eMBB_naive = cmap_eMBB_naive([0,20,40,60,80,100,120,140,160,180,200])
inner_colors_uRLLC_naive = cmap_uRLLC_naive([0,10,20,30,40,50,60,70,80,90,100, 110,120,130,140,150,160,170,180,190,200])
inner_colors_eMBB_llm = cmap_eMBB_llm([0,20,40,60,80,100,120,140,160,180,200])
inner_colors_uRLLC_llm = cmap_uRLLC_llm([0,10,20,30,40,50,60,70,80,90,100, 110,120,130,140,150,160,170,180,190,200])

# Draw LLM-based Method eMBB
for index, row in enumerate(rate_eMBB_llm):
    color_now = inner_colors_eMBB_llm[index] 
    plt.bar(x_llm, row, bottom=rate_eMBB_llm[:index].sum(axis=0),
            width=bar_width, alpha=1, color=color_now, edgecolor='k')

# Draw LLM-based Method uRLLC
for index, row in enumerate(rate_uRLLC_llm):
    color_now = inner_colors_uRLLC_llm[index] 
    plt.bar(x_llm, row, bottom=120-rate_uRLLC_llm[:index+1].sum(axis=0),
            width=bar_width, alpha=1, color=color_now, edgecolor='k')

# Draw Wireless Agent eMBB
for index, row in enumerate(rate_eMBB_agent):
    color_now = inner_colors_eMBB_agent[index] 
    plt.bar(x_agent, row, bottom=rate_eMBB_agent[:index].sum(axis=0),
            width=bar_width, alpha=1, color=color_now, edgecolor='k')

# Draw Wireless Agent uRLLC
for index, row in enumerate(rate_uRLLC_agent):
    color_now = inner_colors_uRLLC_agent[index] 
    plt.bar(x_agent, row, bottom=120-rate_uRLLC_agent[:index+1].sum(axis=0),
            width=bar_width, alpha=1, color=color_now, edgecolor='k')   
    
# Draw Naive Method eMBB
for index, row in enumerate(rate_eMBB_naive):
    color_now = inner_colors_eMBB_naive[index] 
    plt.bar(x_naive, row, bottom=rate_eMBB_naive[:index].sum(axis=0),
            width=bar_width, alpha=1, color=color_now, edgecolor='k')

# Draw Naive Method uRLLC
for index, row in enumerate(rate_uRLLC_naive):
    color_now = inner_colors_uRLLC_naive[index] 
    plt.bar(x_naive, row, bottom=120-rate_uRLLC_naive[:index+1].sum(axis=0),
            width=bar_width, alpha=1, color=color_now, edgecolor='k')   


# Draw idle portions
# Add idle for LLM-based method 
bottom_llm = np.array([10, 30, 40, 40, 60, 60]) # y start
idle_llm = np.array([92, 63, 50, 50, 30, 30])    # gap or idel
x_idle_llm = np.array([5, 10, 15, 20, 25, 30])
plt.bar(x_idle_llm- bar_width - gap/2, idle_llm, bottom=bottom_llm,
        width=bar_width, alpha=1, color='w', edgecolor='k')

bottom_agent = np.array([20, 50, 90]) # agent
idle_agent = np.array([84, 43, 0])
x_idle_agent = np.array([5, 10, 15])
plt.bar(x_idle_agent, idle_agent, bottom=bottom_agent,
        width=bar_width, alpha=1, color='w', edgecolor='k')

bottom_naive = np.array([20, 60, 90]) # LLM
idle_naive = np.array([80, 30, 0])
x_idle_naive = np.array([5, 10, 15])
plt.bar(x_idle_naive + bar_width + gap/2, idle_naive, bottom=bottom_naive,
        width=bar_width, alpha=1, color='w', edgecolor='k')



# Add reference line
plt.plot([3, 32], [90, 90], 'b--', linewidth=1)  # Extended the line

x_arrow = 3.25  # Moved the arrow
plt.annotate('',xy=(x_arrow,0),xytext=(x_arrow,90), arrowprops=dict(arrowstyle = '<->', color='r', linewidth=1))
plt.annotate('',xy=(x_arrow,90),xytext=(x_arrow,121), arrowprops=dict(arrowstyle = '<->', color='r', linewidth=1))
plt.text(1.8, 49, r'eMBB', color='k', fontsize=12)  # Adjusted position
plt.text(1.5, 42, r'(90MHz)', color='k', fontsize=11)  # Adjusted position
plt.text(1.72, 108, r'URLLC', color='k', fontsize=12)  # Adjusted position
plt.text(1.5, 101, r'(30MHz)', color='k', fontsize=11)  # Adjusted position

# Add annotations (including new ones for 25 and 30)
plt.text(5, 125, '30.8%', ha='center', fontsize=11) #Wireless Agent
plt.text(10, 125, '64.2%', ha='center', fontsize=11)
plt.text(15, 125, '100%', ha='center', fontsize=11)
plt.text(20, 125, '100%', ha='center', fontsize=11)
plt.text(25, 125, '100%', ha='center', fontsize=11)  # Added for 25
plt.text(30, 125, '100%', ha='center', fontsize=11)  # Added for 30

plt.text(5 - bar_width - gap/2, 125, '23.3%', ha='center', fontsize=11)  #LLM-based Method 
plt.text(10 - bar_width - gap/2, 125, '47.5%', ha='center', fontsize=11)
plt.text(15 - bar_width - gap/2, 125, '58.3%', ha='center', fontsize=11)
plt.text(20 - bar_width - gap/2, 125, '58.3%', ha='center', fontsize=11)
plt.text(25- bar_width - gap/2, 125, '75%', ha='center', fontsize=11)  # Added for 25
plt.text(30- bar_width - gap/2, 125, '75%', ha='center', fontsize=11)  # Added for 30

plt.text(5 + bar_width + gap/2, 125, '33.3%', ha='center', fontsize=11) #Naive Method
plt.text(10 + bar_width + gap/2, 125, '75%', ha='center', fontsize=11)
plt.text(15 + bar_width + gap/2, 125, '100%', ha='center', fontsize=11)
plt.text(20 + bar_width + gap/2, 125, '100%', ha='center', fontsize=11)
plt.text(25 + bar_width + gap/2, 125, '100%', ha='center', fontsize=11)  # Added for 25
plt.text(30 + bar_width + gap/2, 125, '100%', ha='center', fontsize=11)  # Added for 30

# Add legend
from matplotlib.patches import Patch
legend_elements = [  
    Patch(facecolor=cmap_eMBB_llm(100), edgecolor='k', label='Prompt-based Method'),
    Patch(facecolor=cmap_eMBB_agent(100), edgecolor='k', label='WirelessAgent'),
    Patch(facecolor=cmap_eMBB_naive(100), edgecolor='k', label='Rule-based Method'),
    Patch(facecolor='w', edgecolor='k', label='Idle')
]
plt.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(7.6/10, 3.8/10), fontsize=12)

# Configure axes
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_position(('data', 0))
ax.spines['bottom'].set_linewidth(2)

# Set axis limits and labels
plt.xlim((2.7, 32))  # Extended x-axis limit
plt.xticks(x_positions, ['5', '10', '15', '20', '25', '30'], fontsize=12)  # Added labels for 25 and 30
plt.yticks([])
plt.xlabel('The Number of Users', x=5/10, fontsize=12)
#plt.ylabel('Bandwidth (MHz)', fontsize=12)

# Add arrow at the end of x-axis
ax.plot(32, 0, '>k')  # Moved the arrow
plt.box(True)  # Add frame around plot
plt.tight_layout()
plt.show()