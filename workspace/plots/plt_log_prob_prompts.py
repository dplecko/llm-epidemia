import pandas as pd
import numpy as np
from plotnine import *

# --- 1. Mock Data Setup (Replicating structure and applying new constraints) ---

# Define the models to be used for the rows, maintaining the user's requested order
models_subset = ["llama3_8b_instruct", "phi4"]

# Helper function to map internal model names to display names (for consistency)
def model_name(series):
    name_map = {
        "llama3_8b_instruct": "Llama 3 8B",
        "llama3_70b_instruct": "Llama 3 70B",
        "mistral_7b_instruct": "Mistral 7B",
        "phi4": "Phi4", # Renaming "phi4" to "Phi-3 Mini" for display
        "gemma3_27b_instruct": "Gemma 2 27B",
        "deepseek_7b_chat": "Deepseek 7B",
    }
    return series.map(name_map)

# Define the remaining variables
dimensionality_types = ["Low-Dimensional", "High-Dimensional"]

# Initialize empty list to store mock data for the 4 facets (2 Models x 2 Dimensionalities)
mock_data = []

N_POINTS = 100 # New constant for the number of points per subplot

# Generate mock data for the 4 facet combinations (100 points each)
np.random.seed(42) # for reproducibility

for dim in dimensionality_types:
    for model in models_subset: 
        # Set baseline characteristics
        if model == "llama3_8b_instruct":
            base_log_prob = -3.2 # Slightly less efficient 
            base_score = 70
            log_prob_range = 0.6
            score_range = 15
        else: # phi4 (Phi-3 Mini)
            base_log_prob = -2.7 # More efficient
            base_score = 80
            log_prob_range = 0.5
            score_range = 12

        # Adjustment based on Dimensionality (High-Dimensional tasks are usually harder and less efficient)
        if dim == "High-Dimensional":
            base_score -= 15 
            base_log_prob -= 0.3
        
        for i in range(N_POINTS): # Loop 100 times
            # Add random noise for a scatter effect
            log_prob = base_log_prob + np.random.uniform(-log_prob_range/2, log_prob_range/2)
            score = base_score + np.random.uniform(-score_range/2, score_range/2)
            score = max(0, min(100, score)) # Cap score between 0 and 100

            mock_data.append({
                "model_raw": model,
                "Log Prob Per Token": log_prob,
                "Score": score,
                "Dimensionality": dim,
            })

df_scatter = pd.DataFrame(mock_data)

# Apply display names and categorical types
df_scatter["Model"] = model_name(df_scatter["model_raw"])
df_scatter["Dimensionality"] = pd.Categorical(df_scatter["Dimensionality"], categories=dimensionality_types, ordered=True)
# Setting the explicit order for the rows (Row 1: Llama 3 8B, Row 2: Phi4)
df_scatter["Model"] = pd.Categorical(df_scatter["Model"], categories=["Llama 3 8B", "Phi4"], ordered=True)

# --- Dynamic Axis Calculation for Zooming (Bounding Box) ---
Y_BUFFER = 5
X_BUFFER = 0.1

y_min = max(0, df_scatter['Score'].min() - Y_BUFFER)
y_max = min(100, df_scatter['Score'].max() + Y_BUFFER)

x_min = df_scatter['Log Prob Per Token'].min() - X_BUFFER
x_max = df_scatter['Log Prob Per Token'].max() + X_BUFFER

# --- 2. Plot Generation (Applying the requested style) ---

plt_scatter = (
    # Removed 'color="Prompting"' from aes mapping
    ggplot(df_scatter, aes(x="Log Prob Per Token", y="Score")) 
    
    # Scatter plot, using fixed color and smaller size for 100 points
    + geom_point(size=2, alpha=0.6, color="#4c72b0") 

    # Faceting for 2x2 plot: Model (rows) ~ Dimensionality (columns)
    + facet_grid("Model ~ Dimensionality") 
    
    # Labels
    + labs(
        x="Log Probability Per Token", 
        y="Score",
    )
    
    # Visual Styling from the original plot
    + theme_bw() 
    # Replaced coord_cartesian with scale_y_continuous for dynamic zoom
    + scale_y_continuous(limits=(y_min, y_max))

    + theme(
        # Set background to white
        panel_background=element_rect(fill="white"),
        plot_background=element_rect(fill="white"),
        # Improve Facet Strip appearance (as in original figure)
        strip_background=element_rect(fill="#e0e0e0", color="black"),
        strip_text=element_text(color="black", size=10, fontweight="bold"),
        
        # Explicitly hide legend
        legend_position="none",
        
        # General text styling
        # axis_title_x=element_text(fontweight="bold"),
        # axis_title_y=element_text(fontweight="bold"),
        # plot_title=element_text(size=12, fontweight="bold")
    )
    # Applied dynamic limits and re-calculated breaks for the x-axis
    + scale_x_continuous(limits=(x_min, x_max), breaks=np.linspace(x_min, x_max, 5).round(2))
)

# Display the plot object (assuming the environment handles plotnine rendering)
# print(plt_scatter)

# Save the plot
plt_scatter.save("model_scatter_2x2_revised.png", dpi=300, width=11, height=7)