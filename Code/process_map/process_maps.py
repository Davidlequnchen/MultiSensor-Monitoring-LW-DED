import matplotlib.pyplot as plt
import numpy as np

# Define a function to normalize energy density values to a suitable range for marker sizes
def normalize_to_range(values, new_min, new_max):
    min_val = min(values)
    max_val = max(values)
    return [((new_max - new_min) * (value - min_val) / (max_val - min_val)) + new_min for value in values]

# Update the power range and corresponding lines starting from zero
# power_range = np.linspace(0, 4000, 200) # for W 
power_range = np.linspace(0, 4, 200) # for kW
lack_of_fusion_line = power_range * 17  # Adjusting the slope so it starts from zero. 0.37 for W
keyhole_line= power_range* 5.7  # Adjusting the slope and intercept; 1.17 for W


### ---- experiments for modelling  ----### 
# data_points = {
#     # color: #'magenta', "red", "blue", "cyan", 'green', orange 
#     # marker: D, s, X, o, *
#     # 'E6 (Stable Conduction)': {'power': 2100, 'speed': 1500, 'hatch': 1.7, 'layer thickness': 1.4, 'marker': 'o', 'color': 'blue'}, 
#     # 'E7 (Stable Conduction)': {'power': 2500, 'speed': 1500, 'hatch': 1.7, 'layer thickness': 1.4, 'marker': 'o', 'color': 'blue'},
#     # 'E8 (Conduction + Balling)': {'power': 2900, 'speed': 1500, 'hatch': 1.7, 'layer thickness': 1.4, 'marker': '^', 'color': 'orange'},
#     # 'E9 (Conduction + Balling)': {'power': 2300, 'speed': 2100, 'hatch': 1.7, 'layer thickness': 1.4, 'marker': '^', 'color': 'orange'},
#     'E10 (LoF)': {'power': 1500, 'speed': 2100, 'hatch':1.7, 'layer thickness': 1.4, 'marker': 'X', 'color': '#bcbd22'},
#     # 'E13 (Conduction + Balling)': {'power': 2100, 'speed': 2100, 'hatch': 1.7, 'layer thickness': 1.4, 'marker': '^', 'color': 'orange'},
#     'E17 (Conduction + Balling)': {'power': 1800, 'speed': 2100, 'hatch': 1.7, 'layer thickness': 1.4, 'marker': '^', 'color': 'orange'},
#     'E18 (Conduction + Balling)': {'power': 1800, 'speed': 2100, 'hatch': 1.7, 'layer thickness': 1.4, 'marker': 'X', 'color': '#bcbd22'},
#     'E19 (Conduction + Balling)': {'power': 2100, 'speed': 900, 'hatch':1.7, 'layer thickness': 1.4, 'marker': 'X', 'color': '#bcbd22'},
#     # 'E20 (Conduction + Noise Interference)': {'power': 2100, 'speed': 900, 'hatch':1.7, 'layer thickness': 1.4, 'marker': 'D', 'color': '#7f7f7f'},
#     # 'E21 (Stable Conduction)': {'power': 2100, 'speed': 900, 'hatch':1.7, 'layer thickness': 1.4, 'marker': 'o', 'color': 'blue'},
#     'E22 (LoF)': {'power': 1800, 'speed': 2100, 'hatch': 1.7, 'layer thickness': 1.4, 'marker': 'X', 'color': '#bcbd22'},
#     'E23 (Overheating/Keyhole pores)': {'power': 2900, 'speed': 900, 'hatch': 1.7, 'layer thickness': 1.4, 'marker': 'H', 'color': 'red'},
#     'E24 (Overheating/Keyhole pores)': {'power': 2900, 'speed': 480, 'hatch': 1.7, 'layer thickness': 1.4, 'marker': 'H', 'color': 'red'},
#     'E25 (Balling, Non-printable)': {'power': 3800, 'speed': 2100, 'hatch': 1.7, 'layer thickness': 1.4, 'marker': 'h', 'color': '#d62728'},
#     'E14-April (Conduction + Balling)': {'power': 2100, 'speed': 1500, 'hatch': 1.7, 'layer thickness': 0.7, 'marker': 'X', 'color': '#bcbd22'},
#     'E16-April (Conduction + Balling)': {'power': 2100, 'speed': 1500, 'hatch': 1.7, 'layer thickness': 0.5, 'marker': 'X', 'color': '#bcbd22'},
# }

### ---- selected experiments for modelling  ----###
# data_points = {
#     # E1: Originally E10 (LoF)
#     'E1 (LoF)': {'power': 1500, 'speed': 2100, 'hatch': 1.7, 'layer thickness': 1.4, 'marker': 'X', 'color': '#1f77b4'},  # Cold color for LoF
#     # E2: Originally E17 (LoF + Balling), same as E18, E22
#     'E2 (LoF + Balling)': {'power': 1800, 'speed': 2100, 'hatch': 1.7, 'layer thickness': 1.4, 'marker': '^', 'color': '#2ca02c'},  # Green, somewhat cold
#     # E3: Originally E19 (Conduction + Balling)
#     'E3 (Conduction + Balling)': {'power': 2100, 'speed': 900, 'hatch': 1.7, 'layer thickness': 1.4, 'marker': 's', 'color': '#9467bd'},  # Purple, in between
#     # E5: Originally E23 (Overheating/Keyhole pores)
#     'E5 (Overheating/Keyhole pores)': {'power': 2900, 'speed': 900, 'hatch': 1.7, 'layer thickness': 1.4, 'marker': 'H', 'color': '#ff9896'},  # Lighter warm red, lower power in category
#     # E6: Originally E24 (Overheating/Keyhole pores)
#     'E6 (Overheating/Keyhole pores)': {'power': 2900, 'speed': 480, 'hatch': 1.7, 'layer thickness': 1.4, 'marker': 'D', 'color': '#d62728'},  # Darker warm red, same power, different speed
#     # E7: Originally E25 (Balling, Non-printable)
#     'E7 (Balling, Non-printable)': {'power': 3800, 'speed': 2100, 'hatch': 1.7, 'layer thickness': 1.4, 'marker': 'h', 'color': '#8c2d04'},  # Most intense red, indicating highest severity
#     # E8: Originally E14-April (Conduction + Balling), E16-April same
#     'E8 (Conduction + Balling)': {'power': 2100, 'speed': 1500, 'hatch': 1.7, 'layer thickness': 0.7, 'marker': 'p', 'color': '#ffbb78'},  # Light warm color, indicating moderate severity
# }
### ---- After converting the units --> kW, and mm/s ----###
data_points = {
    'E1 (LoF + Balling)': {'power': 1.5, 'speed': 35, 'hatch': 1.7, 'layer thickness': 1.4, 'marker': 'X', 'color': '#1f77b4'},
    'E2 (LoF)': {'power': 1.8, 'speed': 35, 'hatch': 1.7, 'layer thickness': 1.4, 'marker': '^', 'color': '#2ca02c'},
    'E3 (Conduction + Balling)': {'power': 2.1, 'speed': 15, 'hatch': 1.7, 'layer thickness': 1.4, 'marker': 's', 'color': '#9467bd'},
    'E4 (Conduction + Balling)': {'power': 2.1, 'speed': 25, 'hatch': 1.7, 'layer thickness': 0.7, 'marker': 'p', 'color': '#ffbb78'},
    'E5 (Overheating / Keyhole pores)': {'power': 2.9, 'speed': 15, 'hatch': 1.7, 'layer thickness': 1.4, 'marker': 'H', 'color': '#ff9896'},
    'E6 (Overheating / Keyhole pores)': {'power': 2.9, 'speed': 8, 'hatch': 1.7, 'layer thickness': 1.4, 'marker': 'D', 'color': '#d62728'},
    'E7 (Balling, Non-printable)': {'power': 3.8, 'speed': 35, 'hatch': 1.7, 'layer thickness': 1.4, 'marker': 'h', 'color': '#8c2d04'},
}




# Caluclate energy density
for label, info in data_points.items():
    P = info['power']  # Laser power in watts
    v = info['speed'] / 60  # Convert scanning speed to mm/s from mm/min
    h = info['hatch']
    t = info['layer thickness']
    E = P*1000  / (v * h * t)  # Energy density
    info['energy_density'] = E  # Add the calculated energy density to the data points

# Extract energy density values and normalize them to a new range for marker sizes
energy_density_values = [info['energy_density'] for label, info in data_points.items()]
normalized_marker_sizes = normalize_to_range(energy_density_values, new_min=90, new_max=150)  # Marker size range

# Create a new figure
plt.figure(figsize=(11, 5))

plt.grid(True, which='both', linestyle='--', linewidth=1, alpha=0.5)
# Add the updated lines to the plot
plt.plot(power_range, lack_of_fusion_line, 'k-', label='Lack of fusion')  # Dashed line
plt.plot(power_range, keyhole_line, 'k--', label='Keyhole')  # Solid line


for (label, info), marker_size in zip(data_points.items(), normalized_marker_sizes):
    plt.scatter(info['power'], info['speed'], label=label, marker=info['marker'], color=info['color'], s=marker_size)
    # plt.annotate(f"{info['energy_density']:.1f}", 
    #              (info['power'], info['speed']),
    #                textcoords="offset points", 
    #                xytext=(0,10), 
    #                ha='center', fontsize=12)

# Annotations without arrows
# plt.text(200, 2300, 'Lack of fusion', fontsize=14, verticalalignment='center')
# plt.text(2600, 2000, 'Conduction', fontsize=14, verticalalignment='center')
# plt.text(2500, 300, 'Keyhole', fontsize=14, verticalalignment='center')

# Update the axis labels
plt.xlabel('Laser power (kW)', fontsize=18, labelpad=15)
plt.ylabel('Speed (mm/s)', fontsize=18, labelpad=15)

## for kW, mm/min
plt.xticks([0.0, 0.5, 1.0, 1.5, 1.8, 2.1, 2.5, 2.9, 3.5, 3.8], fontsize=14)  # np.linspace(0, 4, 9) # Ticks from 0 to 4 kW
plt.yticks([0, 5, 8, 15, 25, 35, 40, 50, 60], fontsize=14)  # np.linspace(0, 60, 7) Ticks from 0 to 60 mm/s
plt.xlim(0, 4.0)
plt.ylim(0, 60)




### for W, mm/min
# plt.xticks(fontsize=14, rotation=0, ha='center', va='top', rotation_mode='anchor')
# plt.yticks(fontsize=14)
# plt.tick_params(axis='x', which='major', pad=8)
# plt.tick_params(axis='y', which='major', pad=8)
# # Update the plot limits
# plt.xlim(0, 4000)
# plt.ylim(0, 2500)

# Place the legend outside the plot area, to the right
plt.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=12, labelspacing=0.9, handleheight=1)
plt.tight_layout()

# Show the plot
plt.show()
