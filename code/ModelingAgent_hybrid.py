from websocietysimulator import Simulator
from websocietysimulator.agent import SimulationAgent
import os
from websocietysimulator.llm import LLMBase, OpenAILLM
from websocietysimulator.agent.modules.planning_modules import PlanningBase 
from websocietysimulator.agent.modules.reasoning_modules import ReasoningBase
import json
import re
import numpy as np
import logging

class HeuristicPlanning(PlanningBase):
    def __call__(self, task_description):
        return [{'description': 'Process Data', 'reasoning instruction': 'None', 'tool use instruction': 'None'}]

class GuidedReasoning(ReasoningBase):
    def __init__(self, profile_type_prompt, llm):
        super().__init__(profile_type_prompt=profile_type_prompt, memory=None, llm=llm)
        
    def __call__(self, prompt_input: str):
        # BACK TO BASICS: No complex System Prompt. 
        # Let the few-shot examples in the user prompt do the work.
        messages = [{"role": "user", "content": prompt_input}]
        
        # Temperature 0.1 is still good for consistency
        return self.llm(messages=messages, temperature=0.1, max_tokens=600)

class RobustSimulationAgent(SimulationAgent):
    def __init__(self, llm: LLMBase):
        super().__init__(llm=llm)
        self.planning = HeuristicPlanning(llm=self.llm)
        self.reasoning = GuidedReasoning(profile_type_prompt='', llm=self.llm)

    def calculate_stars_heuristic(self, user_reviews, item_reviews):
        # Keep this logic - it is statistically solid.
        if user_reviews:
            user_ratings = [float(r['stars']) for r in user_reviews if r.get('stars') is not None]
            user_avg = np.mean(user_ratings)
        else:
            user_avg = 3.0 

        if item_reviews:
            item_ratings = [float(r['stars']) for r in item_reviews if r.get('stars') is not None]
            item_avg = np.mean(item_ratings)
        else:
            item_avg = user_avg 

        if item_reviews and user_reviews:
            predicted_stars = (item_avg * 0.7) + (user_avg * 0.3)
        else:
            predicted_stars = item_avg if item_reviews else user_avg

        return float(round(predicted_stars))

    def detect_domain(self, item_info):
        if "authors" in item_info:
            return "goodreads"
        if "address" in item_info or "city" in item_info:
            return "yelp"
        return "amazon"

    # --- SIMPLIFIED DOMAIN PROMPTS ---
    # We removed the "Ghostwriter" logic and strict word counts.
    # We kept the "Domain Context" (e.g. asking for "plot" vs "service") but made it optional.
    
    def build_prompt_goodreads(self, user_info, style_examples, item_info, item_context, target_stars):
        return f"""
TASK: Write a review for this book as if you are the user described below.

TARGET RATING: {target_stars} stars.

Book Info:
Title: {item_info.get("name")}
Authors: {item_info.get("authors")}
Description: {item_info.get("description", "")[:300]}...

User Profile:
{user_info}

Writing Style Examples (Mimic these exactly):
{style_examples}

Other Readers Mentioned:
{item_context}

Instructions:
1. Write a review that justifies {target_stars} stars.
2. Mimic the sentence length, tone, and vocabulary of the Style Examples.
3. If the user usually discusses characters or plot, do so. If they are brief, be brief.
4. Output Format:
RATING: {target_stars}
REVIEW: [Your review text]
"""

    def build_prompt_yelp(self, user_info, style_examples, item_info, item_context, target_stars):
        return f"""
TASK: Write a review for this restaurant as if you are the user described below.

TARGET RATING: {target_stars} stars.

Restaurant Info:
Name: {item_info.get("name")}
Categories: {item_info.get("categories")}
Address: {item_info.get("address")}

User Profile:
{user_info}

Writing Style Examples (Mimic these exactly):
{style_examples}

Popular Opinions:
{item_context}

Instructions:
1. Write a review that justifies {target_stars} stars.
2. Mimic the sentence length, tone, and vocabulary of the Style Examples.
3. If the user usually discusses service or food, do so. If they are brief, be brief.
4. Output Format:
RATING: {target_stars}
REVIEW: [Your review text]
"""

    def build_prompt_amazon(self, user_info, style_examples, item_info, item_context, target_stars):
        return f"""
TASK: Write a review for this product as if you are the user described below.

TARGET RATING: {target_stars} stars.

Product Info:
Name: {item_info.get("name")}
Category: {item_info.get("categories")}
Attributes: {item_info.get("attributes")}

User Profile:
{user_info}

Writing Style Examples (Mimic these exactly):
{style_examples}

Other Buyers Mentioned:
{item_context}

Instructions:
1. Write a review that justifies {target_stars} stars.
2. Mimic the sentence length, tone, and vocabulary of the Style Examples.
3. If the user usually discusses quality or shipping, do so. If they are brief, be brief.
4. Output Format:
RATING: {target_stars}
REVIEW: [Your review text]
"""

    def workflow(self):
        try:
            # --- STEP 1: GATHER RAW DATA ---
            user_id = self.task['user_id']
            item_id = self.task['item_id']
            
            user_info = self.interaction_tool.get_user(user_id=user_id)
            item_info = self.interaction_tool.get_item(item_id=item_id)
            
            user_reviews = self.interaction_tool.get_reviews(user_id=user_id)
            item_reviews = self.interaction_tool.get_reviews(item_id=item_id)

            # --- STEP 2: CALCULATE STARS ---
            target_stars = self.calculate_stars_heuristic(user_reviews, item_reviews)
            
            # --- STEP 3: PREPARE CONTEXT ---
            # Style Examples: REMOVED the strict "Word Count" labels to let the model "feel" the length naturally.
            style_examples = ""
            if user_reviews:
                matching_reviews = [r for r in user_reviews if round(float(r['stars'])) == target_stars]
                if not matching_reviews:
                    matching_reviews = user_reviews 
                style_examples = "\n".join([f"- {r['text'][:300]}..." for r in matching_reviews[:3]])

            # Item Context
            context_list = []
            if item_reviews:
                for r in item_reviews[:1]: # Less context = less distraction
                    clean_text = r['text'].replace('\n', ' ')[:200]
                    context_list.append(f"- {clean_text}...")
            item_context = "\n".join(context_list) if context_list else "No other reviews available."

            # --- STEP 4: SELECT PROMPT BY DOMAIN ---
            domain = self.detect_domain(item_info)
            
            if domain == "goodreads":
                prompt = self.build_prompt_goodreads(
                    user_info, style_examples, item_info, item_context, target_stars
                )
            elif domain == "yelp":
                prompt = self.build_prompt_yelp(
                    user_info, style_examples, item_info, item_context, target_stars
                )
            else:
                prompt = self.build_prompt_amazon(
                    user_info, style_examples, item_info, item_context, target_stars
                )

            # --- STEP 5: GENERATE TEXT ---
            result = self.reasoning(prompt)

            # --- STEP 6: ROBUST PARSING (Always keep this) ---
            final_stars = target_stars
            final_review = "Standard service."

            star_match = re.search(r"(?:RATING|STARS):\s*([\d.]+)", result, re.IGNORECASE)
            review_match = re.search(r"(?:REVIEW):\s*(.*)", result, re.IGNORECASE | re.DOTALL)

            if star_match:
                try:
                    final_stars = float(star_match.group(1))
                except ValueError:
                    pass 

            if review_match:
                final_review = review_match.group(1).strip()
            else:
                clean_text = re.sub(r"(?:RATING|STARS):\s*[\d.]+\s*", "", result, flags=re.IGNORECASE)
                final_review = clean_text.strip()

            if not final_review:
                final_review = "Standard service."

            return {
                "stars": final_stars,
                "review": final_review
            }

        except Exception as e:
            print(f"CRITICAL ERROR in workflow: {e}")
            return {"stars": 3.0, "review": "Standard service."}

if __name__ == "__main__":
    # Configuration
    task_set = "yelp"  # "goodreads", "yelp", or "amazon"
    data_dir = os.getenv("DATA_DIR", "./processed_data")
    api_key = os.getenv("OPENAI_API_KEY")
    
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set. Please set it before running.")
    
    simulator = Simulator(data_dir=data_dir, device="gpu", cache=False)
    
    simulator.set_task_and_groundtruth(
        task_dir=f"./track1/{task_set}/tasks", 
        groundtruth_dir=f"./track1/{task_set}/groundtruth"
    )

    simulator.set_agent(RobustSimulationAgent)
    simulator.set_llm(OpenAILLM(api_key=api_key, model="gpt-3.5-turbo"))

    outputs = simulator.run_simulation(number_of_tasks=None, enable_threading=True, max_workers=20)
    
    evaluation_results = simulator.evaluate()       
    print(evaluation_results)
    
    with open(f'./evaluation_results_track1_{task_set}_hybrid.json', 'w') as f:
        json.dump(evaluation_results, f, indent=4)