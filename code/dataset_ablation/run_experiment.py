import os
import shutil
import json
from websocietysimulator import Simulator
from websocietysimulator.llm import OpenAILLM
from agent import RobustSimulationAgent
from data_processor import (
    load_yelp_data, load_amazon_data, load_goodreads_data, generate_dataset
)

# === CONFIGURATION ===
# Use environment variables for sensitive data and paths
API_KEY = os.getenv("OPENAI_API_KEY")
if not API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable not set. Please set it before running.")

RAW_DATA_PATH = os.getenv("RAW_DATA_PATH", "./raw_dataset")
TEMP_DATA_DIR = "./temp_sim_data"
TASKS_DIR_BASE = "./track1/{domain}/tasks"
GT_DIR_BASE = "./track1/{domain}/groundtruth"

def run_simulation_for_config(target_domain, sources, loaded_data):
    """Runs a single simulation experiment."""
    print(f"\n>>> Running Experiment: Target={target_domain}, Sources={sources}")
    
    # 1. Generate Data
    yelp_in = loaded_data['yelp'] if 'yelp' in sources else None
    amazon_in = loaded_data['amazon'] if 'amazon' in sources else None
    goodreads_in = loaded_data['goodreads'] if 'goodreads' in sources else None
    
    if os.path.exists(TEMP_DATA_DIR):
        shutil.rmtree(TEMP_DATA_DIR)
    
    generate_dataset(TEMP_DATA_DIR, yelp_in, amazon_in, goodreads_in)

    task_dir = TASKS_DIR_BASE.format(domain=target_domain)
    gt_dir = GT_DIR_BASE.format(domain=target_domain)


    # 3. Setup Simulator
    simulator = Simulator(data_dir=TEMP_DATA_DIR, device="gpu", cache=False)
    simulator.set_task_and_groundtruth(task_dir=task_dir, groundtruth_dir=gt_dir)
    
    # 4. Set Agent & LLM
    simulator.set_agent(RobustSimulationAgent)
    simulator.set_llm(OpenAILLM(api_key=API_KEY, model="gpt-3.5-turbo"))
    
    # 5. Run & Evaluate
    simulator.run_simulation(number_of_tasks=None, enable_threading=True, max_workers=10)
    
    # Don't try to extract specific metrics to avoid parsing errors. 
    # Just return whatever the simulator gives us.
    results = simulator.evaluate()
    
    print(f"Raw Result for {target_domain} w/ {sources}: {results}")
    return results

def main():
    print("Loading all raw datasets into memory...")
    yelp_data = load_yelp_data(RAW_DATA_PATH)
    amazon_data = load_amazon_data(RAW_DATA_PATH)
    goodreads_data = load_goodreads_data(RAW_DATA_PATH)
    
    loaded_data = {
        'yelp': yelp_data,
        'amazon': amazon_data,
        'goodreads': goodreads_data
    }

    # Define Experiments
    experiments = {
        'yelp': [
            ['yelp'], 
            ['yelp', 'amazon'], 
            ['yelp', 'amazon', 'goodreads']
        ],
        'amazon': [
            ['amazon'],
            ['amazon', 'yelp'],
            ['amazon', 'yelp', 'goodreads']
        ],
        'goodreads': [
            ['goodreads'],
            ['goodreads', 'amazon'],
            ['goodreads', 'amazon', 'yelp']
        ]
    }

    final_report = {}

    for target, source_configs in experiments.items():
        print(f"--- Processing Target: {target.upper()} ---")
        
        for sources in source_configs:
            # Get raw results
            raw_result = run_simulation_for_config(target, sources, loaded_data)
            
            # Create a string key for the JSON (lists aren't valid keys)
            key = f"{target}_{'_'.join(sources)}"
            final_report[key] = {
                "target": target,
                "sources": sources,
                "metrics": raw_result
            }

    # Save full report to JSON file
    output_file = "cross_domain_results.json"
    print(f"\n\n=== SAVING RESULTS TO {output_file} ===")
    
    try:
        with open(output_file, 'w') as f:
            json.dump(final_report, f, indent=4)
        print("Save successful.")
    except Exception as e:
        print(f"Error saving JSON: {e}")
        # Fallback print if file save fails
        print(json.dumps(final_report, indent=4))

if __name__ == "__main__":
    main()