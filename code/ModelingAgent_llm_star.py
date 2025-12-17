from websocietysimulator import Simulator
from websocietysimulator.agent import SimulationAgent
import json 
import os
from websocietysimulator.llm import LLMBase, OpenAILLM
from websocietysimulator.agent.modules.planning_modules import PlanningBase 
from websocietysimulator.agent.modules.reasoning_modules import ReasoningBase
import logging
import re
import numpy as np

class HeuristicPlanning(PlanningBase):
    def __call__(self, task_description):
        return [{'description': 'Process Data', 'reasoning instruction': 'None', 'tool use instruction': 'None'}]

class GuidedReasoning(ReasoningBase):
    def __init__(self, profile_type_prompt, llm):
        super().__init__(profile_type_prompt=profile_type_prompt, memory=None, llm=llm)
        
    def __call__(self, prompt_input: str):
        messages = [{"role": "user", "content": prompt_input}]
        # Temperature slightly higher than the strict Robust agent (0.1 -> 0.3)
        # to allow the LLM to express its own 'opinion' for the experiment.
        return self.llm(messages=messages, temperature=0.3, max_tokens=600)

class HybridSimulationAgent(SimulationAgent):
    """
    A Hybrid Agent that uses the Robust architecture (no external pickle files)
    but implements the weighted ensemble strategy:
    Final_Stars = 0.6 * Heuristic_Math + 0.4 * LLM_Output
    """
    def __init__(self, llm: LLMBase):
        super().__init__(llm=llm)
        self.planning = HeuristicPlanning(llm=self.llm)
        self.reasoning = GuidedReasoning(profile_type_prompt='', llm=self.llm)
        
    def calculate_stars_heuristic(self, user_reviews, item_reviews):
        """
        Calculates the 'Bias_Pred' using statistical averages.
        """
        # 1. Analyze User Bias (How grumpy is this user?)
        if user_reviews:
            user_ratings = [float(r['stars']) for r in user_reviews if r.get('stars') is not None]
            user_avg = np.mean(user_ratings)
        else:
            user_avg = 3.0 # Default fallback

        # 2. Analyze Item Quality (How good is the item?)
        if item_reviews:
            item_ratings = [float(r['stars']) for r in item_reviews if r.get('stars') is not None]
            item_avg = np.mean(item_ratings)
        else:
            item_avg = user_avg 

        # 3. Weighted Prediction (The "Bias_Pred")
        alpha = 0.7 
        if item_reviews and user_reviews:
            predicted_stars = (item_avg * alpha) + (user_avg * (1 - alpha))
        else:
            predicted_stars = item_avg if item_reviews else user_avg

        return float(round(predicted_stars))

    def workflow(self):
        try:
            # --- STEP 1: GATHER RAW DATA ---
            user_id = self.task['user_id']
            item_id = self.task['item_id']
            
            user_info = self.interaction_tool.get_user(user_id=user_id)
            item_info = self.interaction_tool.get_item(item_id=item_id)
            
            user_reviews = self.interaction_tool.get_reviews(user_id=user_id)
            item_reviews = self.interaction_tool.get_reviews(item_id=item_id)

            # --- STEP 2: CALCULATE HEURISTIC STARS (Bias_Pred) ---
            target_stars = self.calculate_stars_heuristic(user_reviews, item_reviews)
            
            # --- STEP 3: PREPARE PROMPT CONTENT ---
            # Smart Filtering: Find reviews where user gave similar ratings to target
            style_examples = ""
            if user_reviews:
                matching_reviews = [r for r in user_reviews if round(float(r['stars'])) == target_stars]
                if not matching_reviews:
                    matching_reviews = user_reviews 
                style_examples = "\n".join([f"- {r['text'][:300]}..." for r in matching_reviews[:3]])

            # Item context
            item_context = ""
            if item_reviews:
                item_context = f"Other users mentioned: {item_reviews[0]['text'][:200]}"

            # --- STEP 4: GENERATE TEXT (The Experiment) ---
            # We present the target_stars as a "Statistical Prediction" but ask the LLM to verify it.
            prompt = f"""
            You are simulating a specific user on Amazon/Yelp.
            
            User Profile: {user_info}
            User's Past Review Style:
            {style_examples}

            Item Info: {item_info}
            {item_context}

            Statistical Model Prediction: {target_stars} stars.

            Task:
            1. Consider the Item Info and User Profile.
            2. Decide if the Statistical Prediction ({target_stars}) fits this user's likely reaction to this item.
            3. You can agree with the prediction or deviate if the item description strongly conflicts with the user's preferences.
            4. Write a review text consistent with the final rating.

            Output Format:
            stars: [Your Rating, e.g., 1.0, 2.0, 3.0, 4.0, 5.0]
            review: [Your review text]
            """
            result = self.reasoning(prompt)

            # --- STEP 5: PARSE LLM OUTPUT ---
            llm_stars = target_stars # Fallback
            final_review = "Standard service."

            # Regex parsing for robustness
            star_match = re.search(r"stars:\s*([\d.]+)", result, re.IGNORECASE)
            review_match = re.search(r"review:\s*(.*)", result, re.IGNORECASE | re.DOTALL)

            if star_match:
                try:
                    llm_stars = float(star_match.group(1))
                except ValueError:
                    pass 

            if review_match:
                final_review = review_match.group(1).strip()
            else:
                # Fallback cleanup
                lines = result.split('\n')
                for line in lines:
                    if 'review:' not in line.lower() and 'stars:' not in line.lower() and len(line) > 10:
                        final_review = line
                        break

            # --- STEP 6: APPLY WEIGHTED FORMULA ---
            # Formula: 0.6 * Bias_Pred (target_stars) + 0.4 * LLM_Output (llm_stars)
            
            weighted_score = (0.6 * target_stars) + (0.4 * llm_stars)
            
            # Clamp and Round
            final_stars = min(5.0, max(1.0, weighted_score))
            final_stars = float(round(final_stars))

            return {
                "stars": final_stars,
                "review": final_review 
            }

        except Exception as e:
            print(f"Error in workflow: {e}")
            # Fallback to pure heuristic if LLM fails
            try:
                safe_stars = self.calculate_stars_heuristic(
                    self.interaction_tool.get_reviews(user_id=self.task['user_id']),
                    self.interaction_tool.get_reviews(item_id=self.task['item_id'])
                )
            except:
                safe_stars = 3.0
            return {"stars": safe_stars, "review": "Standard experience."}

if __name__ == "__main__":
    # Configuration
    task_set = "goodreads"  # "goodreads", "yelp", or "amazon"
    data_dir = os.getenv("DATA_DIR", "./processed_data")
    api_key = os.getenv("OPENAI_API_KEY")
    
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set. Please set it before running.")
    
    simulator = Simulator(data_dir=data_dir, device="gpu", cache=False)
    simulator.set_task_and_groundtruth(task_dir=f"./track1/{task_set}/tasks", groundtruth_dir=f"./track1/{task_set}/groundtruth")

    simulator.set_agent(HybridSimulationAgent)
    simulator.set_llm(OpenAILLM(api_key=api_key, model="gpt-3.5-turbo"))

    # Run Simulation
    print("Running Hybrid Experiment (0.6 Heuristic + 0.4 LLM)...")
    outputs = simulator.run_simulation(number_of_tasks=None, enable_threading=True, max_workers=10)
    
    # Evaluate
    evaluation_results = simulator.evaluate()       
    filename = f'./evaluation_results_track1_{task_set}_llm_star.json'
    with open(filename, 'w') as f:
        json.dump(evaluation_results, f, indent=4)
    
    print(f"Experiment Complete. Results saved to {filename}")