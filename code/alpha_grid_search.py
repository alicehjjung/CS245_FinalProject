from websocietysimulator import Simulator
from websocietysimulator.agent import SimulationAgent
import json 
from websocietysimulator.llm import LLMBase, OpenAILLM
from websocietysimulator.agent.modules.planning_modules import PlanningBase 
from websocietysimulator.agent.modules.reasoning_modules import ReasoningBase
import logging
import re
import numpy as np
import os

# --- CONFIGURATION ---
# Use environment variables for sensitive data
API_KEY = os.getenv("OPENAI_API_KEY")
if not API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable not set. Please set it before running.")

DATA_DIR = os.getenv("DATA_DIR", "./processed_data")
OUTPUT_FILE = "alpha_grid_search_results.json"

# Tasks to test
TASK_SETS = ["goodreads", "yelp", "amazon"] 
# Alphas to test (0.0 = User Bias only, 1.0 = Item Quality only)
ALPHAS_TO_TEST = [0.1, 0.3, 0.5, 0.7, 0.9] 

class HeuristicPlanning(PlanningBase):
    def __call__(self, task_description):
        return [{'description': 'Process Data', 'reasoning instruction': 'None', 'tool use instruction': 'None'}]

class GuidedReasoning(ReasoningBase):
    def __init__(self, profile_type_prompt, llm):
        super().__init__(profile_type_prompt=profile_type_prompt, memory=None, llm=llm)
        
    def __call__(self, prompt_input: str):
        messages = [{"role": "user", "content": prompt_input}]
        # Very low temperature for consistency
        return self.llm(messages=messages, temperature=1, max_tokens=None)

class RobustSimulationAgent(SimulationAgent):
    # CLASS VARIABLE: This allows us to modify alpha dynamically without passing it to __init__
    ALPHA = 0.5 

    def __init__(self, llm: LLMBase):
        super().__init__(llm=llm)
        self.planning = HeuristicPlanning(llm=self.llm)
        self.reasoning = GuidedReasoning(profile_type_prompt='', llm=self.llm)
        
    def calculate_stars_heuristic(self, user_reviews, item_reviews):
        """
        Pure math logic to predict the star rating using the class-level ALPHA.
        """
        # 1. Analyze User Bias
        if user_reviews:
            user_ratings = [float(r['stars']) for r in user_reviews if r.get('stars') is not None]
            user_avg = np.mean(user_ratings)
        else:
            user_avg = 3.0 

        # 2. Analyze Item Quality
        if item_reviews:
            item_ratings = [float(r['stars']) for r in item_reviews if r.get('stars') is not None]
            item_avg = np.mean(item_ratings)
        else:
            item_avg = user_avg 

        # 3. Weighted Prediction using the dynamic ALPHA
        # Accessing class variable explicitly
        current_alpha = RobustSimulationAgent.ALPHA 
        
        if item_reviews and user_reviews:
            predicted_stars = (item_avg * current_alpha) + (user_avg * (1 - current_alpha))
        else:
            predicted_stars = item_avg if item_reviews else user_avg

        return float(round(predicted_stars))

    def workflow(self):
        try:
            user_id = self.task['user_id']
            item_id = self.task['item_id']
            
            user_info = self.interaction_tool.get_user(user_id=user_id)
            item_info = self.interaction_tool.get_item(item_id=item_id)
            user_reviews = self.interaction_tool.get_reviews(user_id=user_id)
            item_reviews = self.interaction_tool.get_reviews(item_id=item_id)

            # Calculate stars using the current alpha
            target_stars = self.calculate_stars_heuristic(user_reviews, item_reviews)
            
            # Prepare prompts to ensure review text matches the star rating
            style_examples = ""
            if user_reviews:
                matching_reviews = [r for r in user_reviews if round(float(r['stars'])) == target_stars]
                if not matching_reviews:
                    matching_reviews = user_reviews
                style_examples = "\n".join([f"- {r['text'][:300]}..." for r in matching_reviews[:3]])

            item_context = ""
            if item_reviews:
                item_context = f"Other users mentioned: {item_reviews[0]['text'][:200]}"

            prompt = f"""
            Task: Write a short review for a product/business.
            TARGET RATING: {target_stars} stars.
            User Profile: {user_info}
            User's Writing Style Examples: {style_examples}
            Item Info: {item_info}
            {item_context}
            Format:
            stars: [Number]
            review: [Your text]
            """
            result = self.reasoning(prompt)

            # Parsing logic
            final_stars = target_stars
            final_review = "Standard service."
            star_match = re.search(r"RATING:\s*([\d.]+)", result, re.IGNORECASE)
            review_match = re.search(r"REVIEW:\s*(.*)", result, re.IGNORECASE | re.DOTALL)
            
            if star_match:
                try: 
                    final_stars = float(star_match.group(1)) 
                except: pass
            
            if review_match:
                final_review = review_match.group(1).strip()
            elif not result.lower().startswith("rating:"):
                final_review = result.strip()

            return {"stars": final_stars, "review": final_review}

        except Exception as e:
            # print(f"Error: {e}")
            return {"stars": 3.0, "review": "Standard service."}

def run_grid_search():
    overall_results = {}

    for task_set in TASK_SETS:
        print(f"\n{'='*20}\nSTARTING DATASET: {task_set}\n{'='*20}")
        overall_results[task_set] = {}
        
        # Initialize paths
        task_dir = f"./track1/{task_set}/tasks"
        groundtruth_dir = f"./track1/{task_set}/groundtruth"
        
        # Check if paths exist to avoid crashing
        if not os.path.exists(task_dir):
            print(f"Skipping {task_set}: Directory not found at {task_dir}")
            continue

        for alpha in ALPHAS_TO_TEST:
            print(f"--- Testing Alpha: {alpha} ---")
            
            # 1. Update the Class Variable
            RobustSimulationAgent.ALPHA = alpha
            
            # 2. Initialize Simulator
            # We re-init for every loop to ensure clean state
            simulator = Simulator(data_dir=DATA_DIR, device="gpu", cache=False)
            simulator.set_task_and_groundtruth(task_dir=task_dir, groundtruth_dir=groundtruth_dir)
            simulator.set_agent(RobustSimulationAgent)
            simulator.set_llm(OpenAILLM(api_key=API_KEY, model="gpt-3.5-turbo"))

            # 3. Run Simulation
            # Note: Set number_of_tasks to a smaller number (e.g., 20) for debugging, or None for full run
            simulator.run_simulation(number_of_tasks=None, enable_threading=True, max_workers=10)
            
            # 4. Evaluate
            try:
                metrics = simulator.evaluate()
                print(f"Result for {task_set} (alpha={alpha}): {metrics}")
                
                # Store results
                overall_results[task_set][str(alpha)] = metrics
            except Exception as e:
                print(f"Evaluation failed for {task_set} alpha={alpha}: {e}")
                overall_results[task_set][str(alpha)] = "FAILED"

            # Save intermediate results
            with open(OUTPUT_FILE, 'w') as f:
                json.dump(overall_results, f, indent=4)

    return overall_results

if __name__ == "__main__":
    results = run_grid_search()
    
    print("\n\nGRID SEARCH COMPLETE.")
    print(f"Results saved to {OUTPUT_FILE}")