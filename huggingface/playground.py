import pandas as pd
import datasets
import os
from datasets import Dataset, DatasetDict, Features, Value, ClassLabel

# --- Configuration ---

# Define the tasks (as provided by the user)
# NOTE: Corrected the 'variables' for "Hearing Impairments" assuming 'blind' was a typo.
#       Please replace 'hearing_impairment' with the actual column name if different.
tasks_brfss = [
    {
        "name": "BRFSS: Exercise by State",
        "variables": ["exercise_monthly", "state"],
        "prompt": "Does a person living in {} exercise every month?",
        "levels": ["no", "yes"]
    },
    {
        "name": "BRFSS: Diabetes by State",
        "variables": ["diabetes", "state"],
        "prompt": "Has a person living in {} ever been told they have diabetes?",
        "levels": ["no", "yes"]
    },
    {
        "name": "BRFSS: High BP by State",
        "variables": ["high_bp", "state"],
        "prompt": "Does a person living in {} have high blood pressure?",
        "levels": ["no", "yes"]
    },
    {
        "name": "BRFSS: Asthma by State",
        "variables": ["asthma", "state"],
        "prompt": "Does a person living in {} have asthma?",
        "levels": ["no", "yes"]
    },
    {
        "name": "BRFSS: Cholesterol by State",
        "variables": ["cholesterol", "state"],
        "prompt": "Does a person living in {} have high cholesterol?",
        "levels": ["no", "yes"]
    },
    {
        "name": "BRFSS: Visual Impairments by State",
        "variables": ["blind", "state"],
        "prompt": "Does a person living in {} have significant visual impairments/blindness?",
        "levels": ["no", "yes"]
    },
    {
        "name": "BRFSS: Hearing Impairments by State",
        # Assuming 'blind' was a typo here and should be a hearing-related variable
        "variables": ["hearing_impairment", "state"], #<- Replace 'hearing_impairment' if needed
        "prompt": "Does a person living in {} have significant hearing impairments/deafness?",
        "levels": ["no", "yes"]
    },
    {
        "name": "BRFSS: Heart Attack by State",
        "variables": ["heart_attack", "state"],
        "prompt": "Has a person living in {} ever suffered a heart attack?",
        "levels": ["no", "yes"]
    },
    {
        "name": "BRFSS: Stroke by State",
        "variables": ["stroke", "state"],
        "prompt": "Has a person living in {} ever suffered a stroke?",
        "levels": ["no", "yes"]
    },
]

# Path to your Parquet file
# IMPORTANT: Make sure this path is correct relative to where you run the script.
#            Or use an absolute path.
parquet_file_path = "data/clean/brfss.parquet"

# Directory where the Hugging Face dataset will be saved locally
local_dataset_dir = "brfss_benchmark_dataset"

# --- Data Loading and Processing ---

print(f"Loading data from: {parquet_file_path}")
try:
    # Load the entire dataset using pandas
    df = pd.read_parquet(parquet_file_path)
    print(f"Successfully loaded {len(df)} rows.")
except FileNotFoundError:
    print(f"Error: Parquet file not found at {parquet_file_path}")
    print("Please ensure the file exists and the path is correct.")
    # Create a dummy dataframe for demonstration if file not found
    print("Creating dummy data for demonstration purposes.")
    num_dummy_rows = 100
    dummy_data = {
        'exercise_monthly': ['yes', 'no'] * (num_dummy_rows // 2),
        'diabetes': ['no', 'yes'] * (num_dummy_rows // 2),
        'high_bp': ['yes', 'no'] * (num_dummy_rows // 2),
        'asthma': ['no', 'yes'] * (num_dummy_rows // 2),
        'cholesterol': ['yes', 'no'] * (num_dummy_rows // 2),
        'blind': ['no', 'yes'] * (num_dummy_rows // 2),
        'hearing_impairment': ['yes', 'no'] * (num_dummy_rows // 2), # Dummy column
        'heart_attack': ['no', 'yes'] * (num_dummy_rows // 2),
        'stroke': ['yes', 'no'] * (num_dummy_rows // 2),
        'state': ['California', 'Texas', 'New York', 'Florida'] * (num_dummy_rows // 4)
    }
    df = pd.DataFrame(dummy_data)
    # Ensure the dummy hearing impairment column exists if the original check failed
    if "hearing_impairment" not in df.columns:
         df["hearing_impairment"] = ['yes', 'no'] * (num_dummy_rows // 2)


# Create a DatasetDict to hold datasets for each task
benchmark_dataset_dict = DatasetDict()

print("Processing tasks...")
for task in tasks_brfss:
    task_name = task["name"]
    target_variable = task["variables"][0]
    context_variable = task["variables"][1] # e.g., 'state'
    prompt_template = task["prompt"]
    levels = task["levels"] # e.g., ["no", "yes"]

    print(f"  Processing task: {task_name}")

    # Check if required columns exist in the DataFrame
    required_cols = task["variables"]
    if not all(col in df.columns for col in required_cols):
        print(f"    Warning: Skipping task '{task_name}' because required columns ({required_cols}) are not present in the DataFrame.")
        missing_cols = [col for col in required_cols if col not in df.columns]
        print(f"    Missing columns: {missing_cols}")
        continue # Skip this task

    processed_data = []
    for _, row in df.iterrows():
        # Get the context value (e.g., the state name)
        context_value = row[context_variable]
        # Get the label value (e.g., 'yes' or 'no' for diabetes)
        label_value = row[target_variable]

        # Ensure the label value is one of the expected levels for ClassLabel consistency
        # If your data uses different values (e.g., 0/1), you'll need mapping logic here.
        if label_value not in levels:
             # Handle unexpected labels (skip, map, log, etc.)
             # print(f"    Warning: Skipping row with unexpected label '{label_value}' for task '{task_name}'. Expected levels: {levels}")
             continue # Skipping for this example

        # Format the prompt
        formatted_prompt = prompt_template.format(context_value)

        # Store the processed instance
        instance = {
            "prompt": formatted_prompt,
            "label": label_value,
            # Optionally include original columns for reference
            "context_variable": context_variable,
            "context_value": context_value,
            "target_variable": target_variable,
        }
        processed_data.append(instance)

    if not processed_data:
        print(f"    Warning: No data processed for task '{task_name}'.")
        continue

    # Define the features for this task's dataset
    # Using ClassLabel is good practice for classification tasks
    task_features = Features({
        'prompt': Value('string'),
        'label': ClassLabel(names=levels),
        'context_variable': Value('string'),
        'context_value': Value('string'),
        'target_variable': Value('string'),
    })

    # Create a Hugging Face Dataset for the current task
    hf_task_dataset = Dataset.from_list(processed_data, features=task_features)

    # Add dataset info (optional but recommended)
    hf_task_dataset.info.description = f"Data for the task: {task_name}. Predict '{target_variable}' based on '{context_variable}'."
    hf_task_dataset.info.citation = """\
    @misc{your_paper_citation_key_here,
      title={Your Benchmark Paper Title},
      author={Your Name(s)},
      year={2025},
      howpublished={Work in Progress},
    }""" # Add your paper's citation info
    # Add other relevant info like homepage, license etc.
    # hf_task_dataset.info.homepage = "URL to your project/paper"
    # hf_task_dataset.info.license = "e.g., apache-2.0"


    # Add the task dataset to the main DatasetDict
    # Use a cleaned-up version of the task name as the key
    dict_key = task_name.lower().replace(":", "").replace(" ", "_")
    benchmark_dataset_dict[dict_key] = hf_task_dataset
    print(f"    Added task '{task_name}' as '{dict_key}' to DatasetDict with {len(hf_task_dataset)} examples.")


# --- Save Locally ---

if benchmark_dataset_dict:
    print(f"\nSaving the DatasetDict to: {local_dataset_dir}")
    # Create the directory if it doesn't exist
    os.makedirs(local_dataset_dir, exist_ok=True)
    # Save the dataset to disk
    benchmark_dataset_dict.save_to_disk(local_dataset_dir)
    print("Dataset successfully saved locally.")

    # --- How to Load Locally ---
    print("\nTo load the dataset locally later:")
    print(f"from datasets import load_from_disk")
    print(f"loaded_dataset_dict = load_from_disk('{local_dataset_dir}')")
    print("print(loaded_dataset_dict)")

    # --- Pushing to Hub (Example) ---
    print("\nTo push to the Hugging Face Hub (after logging in via huggingface-cli login):")
    print("# from datasets import load_from_disk")
    print("# loaded_dataset_dict = load_from_disk('brfss_benchmark_dataset')")
    print("# loaded_dataset_dict.push_to_hub('your_hf_username/brfss_benchmark') # Replace with your HF username and desired dataset name")

else:
    print("\nNo tasks were processed successfully. DatasetDict is empty and was not saved.")

