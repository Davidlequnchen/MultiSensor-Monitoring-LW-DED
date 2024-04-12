import matplotlib.pyplot as plt
import numpy as np

# Define a function to normalize energy density values to a suitable range for marker sizes
def normalize_to_range(values, new_min, new_max):
    min_val = min(values)
    max_val = max(values)
    return [((new_max - new_min) * (value - min_val) / (max_val - min_val)) + new_min for value in values]

# Update the power range and corresponding lines starting from zero
power_range = np.linspace(0, 4000, 200)
lack_of_fusion_line = power_range * 0.37  # Adjusting the slope so it starts from zero
keyhole_line= power_range* 1.17  # Adjusting the slope and intercept


### ----(1) Each points are unique color-marker combination  ----### 
# data_points = {
#     'E6 (Conduction)': {'power': 2100, 'speed': 1500, 'hatch': 1.7, 'layer thickness': 1.4, 'marker': 'o', 'color': '#1f77b4'},
#     'E7 (Conduction)': {'power': 2500, 'speed': 1500, 'hatch': 1.7, 'layer thickness': 1.4, 'marker': '^', 'color': '#ff7f0e'},
#     'E8 (Conduction)': {'power': 2900, 'speed': 1500, 'hatch': 1.7, 'layer thickness': 1.4, 'marker': 's', 'color': '#2ca02c'},
#     'E9 (Conduction)': {'power': 2300, 'speed': 2100, 'hatch': 1.7, 'layer thickness': 1.4, 'marker': 'P', 'color': '#d62728'},
#     'E10 (LoF)': {'power': 1500, 'speed': 2100, 'hatch': 1.7, 'layer thickness': 1.4, 'marker': '*', 'color': '#9467bd'},
#     'E13 (Conduction)': {'power': 2100, 'speed': 2100, 'hatch': 1.7, 'layer thickness': 1.4, 'marker': 'X', 'color': '#8c564b'},
#     'E21 (Conduction)': {'power': 2100, 'speed': 900, 'hatch': 1.7, 'layer thickness': 1.4, 'marker': 'D', 'color': '#e377c2'},
#     'E22 (LoF)': {'power': 1800, 'speed': 2100, 'hatch': 1.7, 'layer thickness': 1.4, 'marker': '<', 'color': '#7f7f7f'},
#     'E23 (Keyhole)': {'power': 2900, 'speed': 900, 'hatch': 1.7, 'layer thickness': 1.4, 'marker': '>', 'color': '#bcbd22'},
#     'E24 (Keyhole)': {'power': 2900, 'speed': 480, 'hatch': 1.7, 'layer thickness': 1.4, 'marker': 'H', 'color': '#17becf'},
#     'E25 (Conduction)': {'power': 3800, 'speed': 2100, 'hatch': 1.7, 'layer thickness': 1.4, 'marker': 'h', 'color': '#1a55FF'}
# }

### ----(2) Selected experiments for modelling  ----### 
data_points = {
    # color: #'magenta', "red", "blue", "cyan", 'green', orange 
    # marker: D, s, X, o, *
    'E6 (Stable Conduction)': {'power': 2100, 'speed': 1500, 'hatch':1.7, 'layer thickness': 1.4, 'marker': 'o', 'color': 'blue'}, 
    'E7 (Stable Conduction)': {'power': 2500, 'speed': 1500, 'hatch':1.7, 'layer thickness': 1.4, 'marker': 'o', 'color': 'blue'},
    'E8 (Conduction + Balling)': {'power': 2900, 'speed': 1500, 'hatch':1.7, 'layer thickness': 1.4, 'marker': '^', 'color': 'orange'},
    # 'E9 (Conduction + Balling)': {'power': 2300, 'speed': 2100, 'hatch':1.7, 'layer thickness': 1.4, 'marker': '^', 'color': 'orange'},
    'E10 (LoF)': {'power': 1500, 'speed': 2100, 'hatch':1.7, 'layer thickness': 1.4, 'marker': 'X', 'color': '#bcbd22'},
    'E13 (Conduction + Balling)': {'power': 2100, 'speed': 2100, 'hatch':1.7, 'layer thickness': 1.4, 'marker': '^', 'color': 'orange'},
    'E20 (Conduction + Noise Interference)': {'power': 2100, 'speed': 900, 'hatch':1.7, 'layer thickness': 1.4, 'marker': 'D', 'color': '#7f7f7f'},
    'E21 (Stable Conduction)': {'power': 2100, 'speed': 900, 'hatch':1.7, 'layer thickness': 1.4, 'marker': 'o', 'color': 'blue'},
    # 'E22 (LoF + noise)': {'power': 1800, 'speed': 2100, 'hatch':1.7, 'layer thickness': 1.4, 'marker': 'X', 'color': '#bcbd22'},
    'E23 (Keyhole)': {'power': 2900, 'speed': 900, 'hatch':1.7, 'layer thickness': 1.4, 'marker': 'H', 'color': 'red'},
    'E24 (Keyhole)': {'power': 2900, 'speed': 480, 'hatch':1.7, 'layer thickness': 1.4, 'marker': 'H', 'color': 'red'},
    'E25 (Conduction + Balling + Nozzle Damage)': {'power': 3800, 'speed': 2100, 'hatch':1.7, 'layer thickness': 1.4, 'marker': 'h', 'color': '#d62728'}
}


# Caluclate energy density
for label, info in data_points.items():
    P = info['power']  # Laser power in watts
    v = info['speed'] / 60  # Convert scanning speed to mm/s from mm/min
    h = info['hatch']
    t = info['layer thickness']
    E = P / (v * h * t)  # Energy density
    info['energy_density'] = E  # Add the calculated energy density to the data points

# Extract energy density values and normalize them to a new range for marker sizes
energy_density_values = [info['energy_density'] for label, info in data_points.items()]
normalized_marker_sizes = normalize_to_range(energy_density_values, new_min=90, new_max=150)  # Marker size range

# Create a new figure
plt.figure(figsize=(10, 6))

# Add the updated lines to the plot
plt.plot(power_range, lack_of_fusion_line, 'k-', label='Lack of fusion')  # Dashed line
plt.plot(power_range, keyhole_line, 'k--', label='Keyhole')  # Solid line


for (label, info), marker_size in zip(data_points.items(), normalized_marker_sizes):
    plt.scatter(info['power'], info['speed'], label=label, marker=info['marker'], color=info['color'], s=marker_size)
    # Annotate the energy density
    plt.annotate(f"{info['energy_density']:.1f}", 
                 (info['power'], info['speed']), 
                 textcoords="offset points", 
                 xytext=(0,10), 
                 ha='center', fontsize=9)

# Annotations without arrows
plt.text(200, 2300, 'Lack of fusion', fontsize=14, verticalalignment='center')
plt.text(2600, 2000, 'Conduction', fontsize=14, verticalalignment='center')
plt.text(2500, 300, 'Keyhole', fontsize=14, verticalalignment='center')

# Update the axis labels
plt.xlabel('Laser power (W)', fontsize=14)
plt.ylabel('Scanning speed (mm/min)', fontsize=14)

plt.xticks(fontsize=12, rotation=0, ha='center', va='top', rotation_mode='anchor')
plt.yticks(fontsize=12)
plt.tick_params(axis='x', which='major', pad=8)
plt.tick_params(axis='y', which='major', pad=8)

# Update the plot limits
plt.xlim(0, 4000)
plt.ylim(0, 2500)

# Place the legend outside the plot area, to the right
plt.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=10)

# Adjust the layout to make room for the legend
plt.tight_layout()

# Show the plot
plt.show()
