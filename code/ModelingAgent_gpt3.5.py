from websocietysimulator import Simulator
from websocietysimulator.agent import SimulationAgent
import json 
import os
from websocietysimulator.llm import LLMBase, InfinigenceLLM, OpenAILLM
from websocietysimulator.agent.modules.planning_modules import PlanningBase 
from websocietysimulator.agent.modules.reasoning_modules import ReasoningBase
from websocietysimulator.agent.modules.memory_modules import MemoryDILU
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
        # Lower temperature to keep it obedient to the forced star rating
        return self.llm(messages=messages, temperature=0.1, max_tokens=600)

class RobustSimulationAgent(SimulationAgent):
    def __init__(self, llm: LLMBase):
        super().__init__(llm=llm)
        self.planning = HeuristicPlanning(llm=self.llm)
        self.reasoning = GuidedReasoning(profile_type_prompt='', llm=self.llm)
        
    def calculate_stars_heuristic(self, user_reviews, item_reviews):
        """
        Pure math logic to predict the star rating.
        This beats LLM reasoning for numerical accuracy.
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
            item_avg = user_avg # If item is new, rely on user's habit

        # 3. Weighted Prediction
        alpha = 0.7 # Weight for item quality
        if item_reviews and user_reviews:
            predicted_stars = (item_avg * alpha) + (user_avg * (1 - alpha))
        else:
            predicted_stars = item_avg if item_reviews else user_avg

        # Round to nearest 0.5 or 1.0 (Yelp/Amazon usually integers)
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

            # --- STEP 2: CALCULATE STARS (NO LLM) ---
            target_stars = self.calculate_stars_heuristic(user_reviews, item_reviews)
            
            # --- STEP 3: PREPARE PROMPT CONTENT ---
            # Get 3 examples of this user's writing to copy style
            style_examples = ""
            if user_reviews:
                # Try to find reviews where they gave the SAME rating we just predicted
                # This ensures the tone matches the rating.
                matching_reviews = [r for r in user_reviews if round(float(r['stars'])) == target_stars]
                if not matching_reviews:
                    matching_reviews = user_reviews # Fallback to any reviews
                
                style_examples = "\n".join([f"- {r['text'][:300]}..." for r in matching_reviews[:3]])

            # Get some item details to mention
            item_context = ""
            if item_reviews:
                # Summarize top 2 keywords from other reviews (naive approach: just grab text)
                item_context = f"Other users mentioned: {item_reviews[0]['text'][:200]}"

            # --- STEP 4: GENERATE TEXT (GUIDED) ---
            prompt = f"""
            Task: Write a short review for a product/business as if you are the user described below.

            TARGET RATING: {target_stars} stars. (Modify if really necessary).

            User Profile: {user_info}
            User's Writing Style Examples:
            {style_examples}

            Item Info: {item_info}
            {item_context}

            Instructions:
            1. Write a review that justifies {target_stars} stars.
            2. Mimic the sentence length and tone of the User Examples.
            3. Mention specific details about the item.
            4. Output Format:
            stars: [Number]
            review: [Your generated text]
            """
            result = self.reasoning(prompt)

            # --- STEP 5: PARSE ---
            final_stars = target_stars
            final_review = "Standard service."

            star_match = re.search(r"RATING:\s*([\d.]+)", result, re.IGNORECASE)
            review_match = re.search(r"REVIEW:\s*(.*)", result, re.IGNORECASE | re.DOTALL)

            if star_match:
                try:
                    final_stars = float(star_match.group(1))
                except ValueError:
                    pass 

            if review_match:
                final_review = review_match.group(1).strip()
            else:
                if not result.lower().startswith("rating:"):
                    final_review = result.strip()

            if not final_review:
                final_review = "Standard service."

            # DO NOT TRUNCATE HERE if you want to allow full expression,
            # but rely on the Prompt to keep it short.
            return {
                "stars": final_stars,
                "review": final_review 
            }

        except Exception as e:
            print(f"Error: {e}")
            return {"stars": 3.0, "review": "Standard service."}

if __name__ == "__main__":
    # Configuration
    task_set = "amazon"  # "goodreads", "yelp", or "amazon"
    data_dir = os.getenv("DATA_DIR", "./processed_data")
    api_key = os.getenv("OPENAI_API_KEY")
    
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set. Please set it before running.")
    
    simulator = Simulator(data_dir=data_dir, device="gpu", cache=False)
    simulator.set_task_and_groundtruth(task_dir=f"./track1/{task_set}/tasks", groundtruth_dir=f"./track1/{task_set}/groundtruth")

    # Set the agent and LLM
    simulator.set_agent(RobustSimulationAgent)
    simulator.set_llm(OpenAILLM(api_key=api_key, model="gpt-3.5-turbo"))

    # Run the simulation
    # If you don't set the number of tasks, the simulator will run all tasks.
    outputs = simulator.run_simulation(number_of_tasks=None, enable_threading=True, max_workers=10)
    # print(outputs)
    
    # Evaluate the agent
    evaluation_results = simulator.evaluate()       
    with open(f'./evaluation_results_track1_{task_set}_gpt3.5.json', 'w') as f:
        json.dump(evaluation_results, f, indent=4)

    # Get evaluation history
    evaluation_history = simulator.get_evaluation_history()
