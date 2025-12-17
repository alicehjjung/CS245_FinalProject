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
        # --- SYSTEM PROMPT UPDATE ---
        # We now ALLOW the LLM to output labels because we need to parse the new star rating
        system_prompt = (
            "You are a ghostwriter simulating a specific human user's reviews. "
            "You are NOT an AI assistant. You are the user. "
            "Be biased according to the profile. "
            "Follow the requested output format strictly."
        )
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt_input}
        ]

        # Increased max_tokens to 2000 to ensure it doesn't get cut off while thinking
        return self.llm(messages=messages, temperature=1, max_tokens=2000)

class RobustSimulationAgent(SimulationAgent):
    def __init__(self, llm: LLMBase):
        super().__init__(llm=llm)
        self.planning = HeuristicPlanning(llm=self.llm)
        self.reasoning = GuidedReasoning(profile_type_prompt='', llm=self.llm)
        
    def calculate_stars_heuristic(self, user_reviews, item_reviews):
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

        # 3. Weighted Prediction
        if item_reviews and user_reviews:
            predicted_stars = (item_avg * 0.7) + (user_avg * 0.3)
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

            # --- STEP 2: CALCULATE SUGGESTED STARS ---
            suggested_stars = self.calculate_stars_heuristic(user_reviews, item_reviews)
            
            # --- STEP 3: PREPARE PROMPT CONTENT ---
            # 3a. Calculate Strict Length Constraints
            avg_len = 30 
            if user_reviews:
                lens = [len(r['text'].split()) for r in user_reviews if r.get('text')]
                if lens:
                    avg_len = int(np.mean(lens))
            
            # Create a tight window (+/- 25%) to force score alignment
            min_words = max(3, int(avg_len * 0.75))
            max_words = int(avg_len * 1.25) + 5
            
            # 3b. Get Context
            style_examples = ""
            if user_reviews:
                matching_reviews = [r for r in user_reviews if round(float(r['stars'])) == suggested_stars]
                if not matching_reviews:
                    matching_reviews = user_reviews 
                style_examples = "\n".join([f"- ({len(r['text'].split())} words) {r['text'][:250]}..." for r in matching_reviews[:3]])

            context_list = []
            if item_reviews:
                for r in item_reviews[:3]:
                    clean_text = r['text'].replace('\n', ' ')[:150]
                    context_list.append(f"- {clean_text}...")
            
            item_context_str = "\n".join(context_list) if context_list else "No other reviews available."

            # --- STEP 4: GENERATE TEXT (With Length Enforcement) ---
            prompt = f"""
            Task: Write a short review for a product/business as if you are the user described below.

            TARGET RATING: {suggested_stars} stars. (Modify if really necessary).
            Target Word Count: {avg_len} words.
            Acceptable Range: {min_words} to {max_words} words.

            User Profile: {user_info}
            User's Writing Style Examples:
            {style_examples}

            Item Info: {item_info}
            Context from other users:
            {item_context_str}

            Instructions:
            1. Write a review that justifies {suggested_stars} stars.
            2. Mimic the sentence length and tone of the User Examples.
            3. Mention specific details about the item.
            4. Output Format:
            RATING: [Number]
            REVIEW: [Text]

            """
            
            result = self.reasoning(prompt) 

            # --- STEP 5: PARSE ---
            final_stars = suggested_stars
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
            print(f"CRITICAL ERROR in workflow: {e}")
            return {"stars": 3.0, "review": "Standard service."}        

if __name__ == "__main__":
    # Configuration
    task_set = "goodreads"  # "goodreads", "yelp", or "amazon"
    data_dir = os.getenv("DATA_DIR", "./processed_data")
    api_key = os.getenv("OPENAI_API_KEY")
    
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set. Please set it before running.")
    
    simulator = Simulator(data_dir=data_dir, device="gpu", cache=False)
    simulator.set_task_and_groundtruth(task_dir=f"./track1/{task_set}/tasks", groundtruth_dir=f"./track1/{task_set}/groundtruth")

    # Set the agent and LLM
    simulator.set_agent(RobustSimulationAgent)
    simulator.set_llm(OpenAILLM(api_key=api_key, model="gpt-5"))

    # Run the simulation
    # If you don't set the number of tasks, the simulator will run all tasks.
    outputs = simulator.run_simulation(number_of_tasks=None, enable_threading=True, max_workers=20)
    # print(outputs)
    
    # Evaluate the agent
    evaluation_results = simulator.evaluate()       
    with open(f'./evaluation_results_track1_{task_set}_gpt5.json', 'w') as f:
        json.dump(evaluation_results, f, indent=4)

    # Get evaluation history
    evaluation_history = simulator.get_evaluation_history()
