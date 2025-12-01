import pandas as pd
import numpy as np
from plotnine import *
import pandas as pd
import json
import os
sys.path.insert(0, os.path.dirname(__file__))

from workspace.utils.helpers import load_dts
from workspace.utils.hd_helpers import fit_lgbm, promptify, gen_prob_lvls
from workspace.common import *
from tqdm import tqdm


def map_hd_task_lengths(task_spec, prob=False):
    # Step 1: determine if query is marginal or conditional
    if "v_cond" in task_spec:
        ttyp = "hd"
    elif len(task_spec["variables"]) == 1:
        ttyp = "marginal"
    elif len(task_spec["variables"]) == 2:
        ttyp = "conditional"
    
    data = load_dts(task_spec, None)

    if ttyp == "hd":
        cond_vars = task_spec["v_cond"]
        out_var = task_spec["v_out"]
        
        # get all combinations
        cond_df = data[cond_vars].drop_duplicates().reset_index(drop=True)
        cond_df["llm_pred"] = np.nan
        dataset_name = task_spec['dataset'].split('/')[-1].split('.')[0]
        prompts = [
            promptify(out_var, cond_vars, row, dataset_name, prob=prob)  # type: ignore[name‑defined]
            for _, row in cond_df.iterrows()
        ]
        if isinstance(prompts, list):
            return len(prompts)
        else:
            return 1


def load_highdim_logprobs(file):
    task_hd_prompt_lenghts = []
    for task_hd in tqdm(task_specs_hd):
        task_hd_prompt_lenghts.append(map_hd_task_lengths(task_hd, prob=False))
        
    with open(file, "r") as f:
        res = json.load(f)
    # average prompts per task
    values = list(res.values())
    
    avg_values = []
    start = 0
    for el in task_hd_prompt_lenghts:
        current_vals = []
        for i in range(start, start + el):
            current_vals.append(values[i])
        start += el
        avg_values.append(np.mean(current_vals))
    
    return avg_values


def load_lowdim_logprobs(file):
    with open(file, "r") as f:
        res = json.load(f)
    return list(res.values())


df = pd.read_parquet("pacho_kralj.parquet")

llama_lowdim_logprobs = load_lowdim_logprobs("log_probs_llama3_8b_instruct_lowdim.json")
llama_lowdim_scores = list(df[(df["model"] == "llama3_8b_instruct") & (df["dim"] == 1)]["score"])

phi4_lowdim_logprobs = load_lowdim_logprobs("log_probs_phi4_lowdim.json")
phi4_lowdim_scores = list(df[(df["model"] == "phi4") & (df["dim"] == 1)]["score"])

llama_highdim_logprobs = load_highdim_logprobs("log_probs_llama3_8b_instruct_highdim.json")
llama_highdim_scores = list(df[(df["model"] == "llama3_8b_instruct") & (df["dim"] > 1)]["score"])

phi4_highdim_logprobs = load_highdim_logprobs("log_probs_phi4_highdim.json")
phi4_highdim_scores = list(df[(df["model"] == "phi4") & (df["dim"] > 1)]["score"])


def plot2x2():

    # Combine the four data arrays into the final required DataFrame structure
    data_frames = []

    data_frames.append(pd.DataFrame({
        "Log Prob Per Token": llama_lowdim_logprobs,
        "Score": llama_lowdim_scores,
        "Model": "Llama 3 8B",
        "Dimensionality": "Low-Dimensional"
    }))

    data_frames.append(pd.DataFrame({
        "Log Prob Per Token": phi4_lowdim_logprobs,
        "Score": phi4_lowdim_scores,
        "Model": "Phi4",
        "Dimensionality": "Low-Dimensional"
    }))

    data_frames.append(pd.DataFrame({
        "Log Prob Per Token": llama_highdim_logprobs,
        "Score": llama_highdim_scores,
        "Model": "Llama 3 8B",
        "Dimensionality": "High-Dimensional"
    }))

    data_frames.append(pd.DataFrame({
        "Log Prob Per Token": phi4_highdim_logprobs,
        "Score": phi4_highdim_scores,
        "Model": "Phi4",
        "Dimensionality": "High-Dimensional"
    }))

    df_scatter = pd.concat(data_frames, ignore_index=True)

    # Define and order categories for faceting
    dimensionality_types = ["Low-Dimensional", "High-Dimensional"]
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
        # Applied dynamic limits for the y-axis
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
            
            # General text styling (axes titles kept bold, plot title removed as per your last code update)
            axis_title_x=element_text(fontweight="bold"),
            axis_title_y=element_text(fontweight="bold"),
        )
        # Applied dynamic limits and re-calculated breaks for the x-axis
        # Note: If you encounter a drawing error, try removing the 'breaks' argument.
        + scale_x_continuous(limits=(x_min, x_max), breaks=np.linspace(x_min, x_max, 5).round(2))
    )

    # Display the plot object (assuming the environment handles plotnine rendering)
    print(plt_scatter)

    # Save the plot
    plt_scatter.save("model_scatter_2x2_revised.png", dpi=300, width=11, height=7)
    
    
def plot1x2():
    # Combine the four data arrays into the final required DataFrame structure
    data_frames = []

    data_frames.append(pd.DataFrame({
        "Log Prob Per Token": llama_lowdim_logprobs,
        "Score": llama_lowdim_scores,
        "Model": "Llama 3 8B",
        "Dimensionality": "Low-Dimensional"
    }))

    data_frames.append(pd.DataFrame({
        "Log Prob Per Token": phi4_lowdim_logprobs,
        "Score": phi4_lowdim_scores,
        "Model": "Phi4",
        "Dimensionality": "Low-Dimensional"
    }))

    data_frames.append(pd.DataFrame({
        "Log Prob Per Token": llama_highdim_logprobs,
        "Score": llama_highdim_scores,
        "Model": "Llama 3 8B",
        "Dimensionality": "High-Dimensional"
    }))

    data_frames.append(pd.DataFrame({
        "Log Prob Per Token": phi4_highdim_logprobs,
        "Score": phi4_highdim_scores,
        "Model": "Phi4",
        "Dimensionality": "High-Dimensional"
    }))

    df_scatter = pd.concat(data_frames, ignore_index=True)

    # Define and order categories for faceting
    dimensionality_types = ["Low-Dimensional", "High-Dimensional"]
    df_scatter["Dimensionality"] = pd.Categorical(df_scatter["Dimensionality"], categories=dimensionality_types, ordered=True)
    # Setting the explicit order for the rows (Row 1: Llama 3 8B, Row 2: Phi4)
    df_scatter["Model"] = pd.Categorical(df_scatter["Model"], categories=["Llama 3 8B", "Phi4"], ordered=True)

    # Prepare a new column for the color aesthetic (same as Dimensionality, but required for the legend title)
    df_scatter["Dimension_Type"] = df_scatter["Dimensionality"]

    # --- Dynamic Axis Calculation for Zooming (Bounding Box) ---
    Y_BUFFER = 5
    X_BUFFER = 0.1

    y_min = max(0, df_scatter['Score'].min() - Y_BUFFER)
    y_max = min(100, df_scatter['Score'].max() + Y_BUFFER)

    x_min = df_scatter['Log Prob Per Token'].min() - X_BUFFER
    x_max = df_scatter['Log Prob Per Token'].max() + X_BUFFER

    # --- 2. Plot Generation (Applying the requested style) ---

    plt_scatter = (
        # Map color to the new Dimension_Type column
        ggplot(df_scatter, aes(x="Log Prob Per Token", y="Score", color="Dimension_Type")) 
        
        # Scatter plot, using color from aesthetic, size 2, and alpha 0.6
        + geom_point(size=2, alpha=0.6) 

        # Faceting for 1 row, 2 columns: Model (columns)
        + facet_grid(". ~ Model") 
        
        # Labels
        + labs(
            x="Log Probability Per Token", 
            y="Score",
            color="Dimensionality" # Set the legend title
        )
        
        # Visual Styling from the original plot
        + theme_bw() 
        # Applied dynamic limits for the y-axis
        + scale_y_continuous(limits=(y_min, y_max))

        + theme(
            # Set background to white
            panel_background=element_rect(fill="white"),
            plot_background=element_rect(fill="white"),
            # Improve Facet Strip appearance (as in original figure)
            strip_background=element_rect(fill="#e0e0e0", color="black"),
            strip_text=element_text(color="black", size=10, fontweight="bold"),
            
            # Explicitly position the legend inside (as per original requirements)
            legend_position="inside",
            # MOVED LEGEND TO UPPER RIGHT CORNER
            legend_position_inside=(0.98, 0.98), 
            legend_background=element_rect(color="black", fill="white"),
            legend_margin=5,
            legend_title=element_text(fontweight="bold"),
            
            # General text styling (axes titles kept bold)
            axis_title_x=element_text(fontweight="bold"),
            axis_title_y=element_text(fontweight="bold"),
        )
        # Applied dynamic limits and breaks for the x-axis
        + scale_x_continuous(limits=(x_min, x_max), breaks=np.linspace(x_min, x_max, 5).round(2))
        
        # Use the custom colors to match the user's bar chart palette
        + scale_color_manual(values={"Low-Dimensional": "#551A8B", "High-Dimensional": "#FADA51"})
    )

    # Display the plot object (assuming the environment handles plotnine rendering)
    print(plt_scatter)

    # Save the plot
    plt_scatter.save("model_scatter_1x2_revised.png", dpi=300, width=11, height=7)
    
plot1x2()
plot2x2()