import matplotlib.pyplot as plt
import numpy as np

# Define a function to normalize energy density values to a suitable range for marker sizes
def normalize_to_range(values, new_min, new_max):
    min_val = min(values)
    max_val = max(values)
    return [((new_max - new_min) * (value - min_val) / (max_val - min_val)) + new_min for value in values]

# Update the power range and corresponding lines starting from zero
power_range = np.linspace(0, 3500, 100)
lack_of_fusion_line = power_range * 0.5  # Adjusting the slope so it starts from zero
conduction_keyhole_transition = power_range* 1.2  # Adjusting the slope and intercept

data_points = {
    'E6 (Conduction)': {'power': 2100, 'speed': 1500, 'hatch':1.7, 'layer thickness': 1.4, 'marker': 'D', 'color': 'magenta'},
    'E7 (Conduction)': {'power': 2500, 'speed': 1500, 'hatch':1.7, 'layer thickness': 1.4, 'marker': 's', 'color': 'orange'},
    'E8 (Keyhole?)': {'power': 2900, 'speed': 1500, 'hatch':1.7, 'layer thickness': 1.4, 'marker': '*', 'color': 'green'},
    'E9 (Conduction)': {'power': 2300, 'speed': 2100, 'hatch':1.7, 'layer thickness': 1.4, 'marker': 's', 'color': 'red'},
    'E10 (LoF)': {'power': 1500, 'speed': 2100, 'hatch':1.7, 'layer thickness': 1.4, 'marker': 'X', 'color': 'blue'},
    'E13 (Conduction)': {'power': 2100, 'speed': 2100, 'hatch':1.7, 'layer thickness': 1.4, 'marker': 'o', 'color': 'cyan'}
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
normalized_marker_sizes = normalize_to_range(energy_density_values, new_min=50, new_max=300)  # Marker size range

# Create a new figure
plt.figure(figsize=(10, 6))

# Add the updated lines to the plot
plt.plot(power_range, lack_of_fusion_line, 'k--', label='Lack of fusion')  # Dashed line
plt.plot(power_range, conduction_keyhole_transition, 'k-', label='Keyhole')  # Solid line


for (label, info), marker_size in zip(data_points.items(), normalized_marker_sizes):
    plt.scatter(info['power'], info['speed'], label=label, marker=info['marker'], color=info['color'], s=marker_size)
    # Annotate the energy density
    plt.annotate(f"{info['energy_density']:.2f}", 
                 (info['power'], info['speed']), 
                 textcoords="offset points", 
                 xytext=(0,10), 
                 ha='center', fontsize=9)

# Annotations without arrows
plt.text(200, 1700, 'Lack of fusion', fontsize=14, verticalalignment='center')
plt.text(2000, 1750, 'Conduction', fontsize=14, verticalalignment='center')
plt.text(2500, 300, 'Keyhole', fontsize=14, verticalalignment='center')

# Update the axis labels
plt.xlabel('Laser power (W)', fontsize=14)
plt.ylabel('Scanning speed (mm/min)', fontsize=14)

plt.xticks(fontsize=12, rotation=0, ha='center', va='top', rotation_mode='anchor')
plt.yticks(fontsize=12)
plt.tick_params(axis='x', which='major', pad=8)
plt.tick_params(axis='y', which='major', pad=8)

# Update the plot limits
plt.xlim(0, 3500)
plt.ylim(0, 2500)

# Place the legend outside the plot area, to the right
plt.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=12)

# Adjust the layout to make room for the legend
plt.tight_layout()

# Show the plot
plt.show()
