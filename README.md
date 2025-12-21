# 245 Project - Agent Society Challenge

Complete implementation of user behavior simulation agents for the Agent Society Challenge.

**Project Report**: [PDF](https://drive.google.com/file/d/1pxWwJre1lVOu_b4b2KOZHqVrlZ24Y-Vp/view?usp=sharing)

## 📁 Folder Structure

```
245_project/
├── code/                               # All Python source code
│   ├── ModelingAgent_baseline.py       # Pure LLM-driven baseline
│   ├── ModelingAgent_gpt3.5.py         # Heuristic + LLM (RECOMMENDED)
│   ├── ModelingAgent_gpt5.py           # Enhanced with strict constraints
│   ├── ModelingAgent_hybrid.py         # Domain-aware prompt tuning
│   ├── ModelingAgent_llm_star.py       # Ensemble method (60% heuristic + 40% LLM)
│   ├── alpha_grid_search.py            # Hyperparameter optimization
│   └── dataset_ablation/               # Cross-domain evaluation
│       ├── agent.py
│       ├── data_processor.py
│       └── run_experiment.py
├── logs/                               # Experiment results & evaluation outputs
│   ├── evaluation_results_track1_*.json
│   ├── alpha_grid_search_results.json
│   └── cross_domain_results.json
├── track1/ & track2/                   # Track-specific evaluation data
├── .env.example                        # Environment variable template
├── .gitignore                          # Git ignore rules
└── README.md                           # This file
```

---

## 🚀 Quick Start

### 1. Environment Setup

```bash
# Copy environment template
cp .env.example .env

# Edit .env with your settings
nano .env
```

**Required Variables:**
- `OPENAI_API_KEY` - Your OpenAI API key (get from https://platform.openai.com/account/api-keys)

**Optional Variables:**
- `DATA_DIR` - Path to processed_data (default: `./processed_data`)
- `RAW_DATA_PATH` - Path to raw_dataset (default: `../raw_dataset`)
- `DEVICE` - GPU device: `cpu`, `cuda`, or `mps` (default: `cpu`)

### 2. Install Dependencies

```bash
pip install -r ../requirements.txt
```

### 3. Run an Agent

```bash
# Try the recommended hybrid agent
python code/ModelingAgent_hybrid.py

# Or any other agent
python code/ModelingAgent_gpt3.5.py
```

---

## 📚 Agent Documentation

### Overview

All agents follow the same workflow:
1. **Fetch Context Data** — Retrieve user profile, item details, historical reviews via `InteractionTool`
2. **Predict Star Rating** — Estimate rating (heuristic or LLM-based)
3. **Generate Review Text** — Use LLM to compose review matching user's style
4. **Return Output** — `{"stars": float, "review": str}`

---

### 1. ModelingAgent_baseline.py

**Approach:** Pure LLM-driven (no heuristics)

**Key Features:**
- Uses `MemoryDILU` module to retrieve similar reviews
- Constructs detailed prompt with user profile, item details, style examples
- Relies entirely on LLM for both star and review generation
- Parses output for `stars:` and `review:` lines

**Best For:**
- Baseline comparisons
- Understanding raw LLM capability

**Limitations:**
- No statistical grounding (unrealistic ratings possible)
- May ignore user preferences if prompt unclear

---

### 2. ModelingAgent_gpt3.5.py ⭐ RECOMMENDED

**Approach:** Heuristic star prediction + LLM text generation

**Key Features:**
- **Heuristic Star Calculation:**
  ```
  predicted_stars = (item_avg * 0.7) + (user_avg * 0.3)
  ```
  - Computes user bias (average rating) and item quality
  - Weights item quality (70%) more than user bias (30%)
  - Rounds to 1.0–5.0 scale

- **LLM for Text Only:**
  - Provides pre-calculated target rating to guide LLM
  - LLM generates review consistent with rating
  - Temperature 0.1 for consistency

**Why It Works:**
- Star ratings are statistically predictable from historical averages
- LLM excels at generating natural text, not predictions
- Separates concerns: math handles numbers, LLM handles language

**Best For:**
- Production use with GPT-3.5-Turbo
- Quick inference with reasonable accuracy

---

### 3. ModelingAgent_gpt5.py

**Approach:** Enhanced version of GPT3.5 with stricter constraints

**Key Features:**
- Same heuristic star calculation as GPT3.5
- **Strict word-count enforcement:**
  - Calculates average review length from user's history
  - Enforces ±25% window around average
  - Example: 30-word avg → 22–42 words required
- **System prompt:**
  - Frames as "ghostwriter" with strict persona
  - Forces LLM to stay in character
- Higher temperature (1.0) and max_tokens (2000) for creative freedom
- More robust parsing with fallback cleanup

**Why It Works:**
- User writing length is highly consistent
- Length enforcement creates authentic reviews
- System prompt prevents LLM from "breaking character"

**Best For:**
- Datasets requiring style consistency (Goodreads)
- Higher fidelity simulations

---

### 4. ModelingAgent_hybrid.py

**Approach:** Domain-aware prompt tuning + flexible parsing

**Key Features:**
- **Domain Detection:**
  - Identifies Goodreads, Yelp, or Amazon from item fields
  - Uses domain-specific prompts (plot for books, service for restaurants)
- **Simplified Prompts:**
  - Removes strict "ghostwriter" framing
  - No word-count enforcement (lets LLM choose naturally)
  - Focuses on style examples and domain context
- **Flexible Parsing:**
  - Case-insensitive rating extraction
  - Falls back to first line if format breaks
- Same heuristic star calculation as GPT3.5
- Temperature 0.1 for consistency

**Why It Works:**
- Domain-specific prompts improve relevance
- Simpler prompts more robust across LLM versions
- Flexible parsing handles unexpected formats

**Best For:**
- Multi-domain simulations (Amazon + Yelp + Goodreads)
- Balancing quality and robustness

---

### 5. ModelingAgent_llm_star.py

**Approach:** Ensemble method (60% heuristic + 40% LLM)

**Key Features:**
- **Two Star Predictions:**
  1. Heuristic (0.7×item + 0.3×user)
  2. LLM verification/suggestion
- **Weighted Ensemble:**
  ```
  final_stars = 0.6 × heuristic + 0.4 × llm_stars
  ```
- LLM is consultative ("Do you agree with X stars?")
- Temperature 0.3 for slight disagreement capability
- Falls back to heuristic if LLM fails

**Why It Works:**
- Combines interpretable math with learned patterns
- LLM catches edge cases (luxury items, etc.)
- Weighted formula prevents LLM errors from dominating

**Best For:**
- Research/experimentation with ensembles
- Datasets where LLM intuition adds value

---

### 6. alpha_grid_search.py

**Purpose:** Hyperparameter optimization for star prediction

**Key Features:**
- **Grid Search Over Alpha Values:**
  - Tests `predicted_stars = (item_avg × α) + (user_avg × (1-α))`
  - Default: `[0.1, 0.3, 0.5, 0.7, 0.9]`
  - α=0.1: Heavy user bias, α=0.9: Heavy item quality
- **Dynamic Class Variable:**
  - Uses `RobustSimulationAgent.ALPHA` without reinitializing
  - Tests across all datasets (Goodreads, Yelp, Amazon)
- **Resumable:**
  - Saves intermediate results after each alpha
  - Can resume if interrupted

**Output:**
```json
{
  "goodreads": {
    "0.1": {metrics...},
    "0.3": {metrics...},
    ...
  },
  "yelp": {...}
}
```

**Usage:**
```bash
python code/alpha_grid_search.py
# Results: logs/alpha_grid_search_results.json
```

---

### 7. dataset_ablation/

**Purpose:** Cross-domain evaluation (train on one domain, test on another)

**Structure:**
```
dataset_ablation/
├── agent.py                  # RobustSimulationAgent
├── data_processor.py         # Data loading & combining
├── run_experiment.py         # Main orchestration
└── temp_sim_data/            # Processed datasets
```

**Example Experiments:**
```
Yelp Target:
  1. Yelp only
  2. Yelp + Amazon
  3. Yelp + Amazon + Goodreads

Amazon Target:
  1. Amazon only
  2. Amazon + Yelp
  ...
```

**Usage:**
```bash
cd code/dataset_ablation
python run_experiment.py
# Results: ../../logs/cross_domain_results.json
```

---

## 📊 Agent Comparison Table

| Agent | Star Method | Text Method | Domain Support | Robustness | Speed | Use Case |
|-------|-------------|------------|-----------------|-----------|-------|----------|
| **Baseline** | LLM | LLM | Generic | Medium | Slow | Reference |
| **GPT3.5** | Heuristic | LLM | Generic | High | Fast | **Recommended** |
| **GPT5** | Heuristic | LLM (strict) | Generic | High | Slow | Research |
| **Hybrid** | Heuristic | LLM (domain-aware) | All 3 | Very High | Fast | Research |
| **LLM_Star** | Ensemble | LLM | Generic | High | Slow | Research |
| **alpha_grid_search** | Heuristic (tuned) | N/A | All 3 | Very High | N/A | Hyperparameter Tuning |
| **dataset_ablation** | Heuristic | N/A | Cross-domain | N/A | N/A | Ablation Studies |

---

## 🔧 Common Patterns

### Data Retrieval
```python
user_info = self.interaction_tool.get_user(user_id=user_id)
item_info = self.interaction_tool.get_item(item_id=item_id)
user_reviews = self.interaction_tool.get_reviews(user_id=user_id)
item_reviews = self.interaction_tool.get_reviews(item_id=item_id)
```

### LLM Invocation
```python
messages = [{"role": "user", "content": prompt}]
result = self.llm(messages=messages, temperature=0.1, max_tokens=600)
```

### Output Format
```python
return {
    "stars": 4.5,        # 1.0 to 5.0
    "review": "..."      # 2–4 sentences typically
}
```

---

## 📖 Example: Running the Full Simulator

```python
from websocietysimulator import Simulator
from code.ModelingAgent_hybrid import RobustSimulationAgent
from websocietysimulator.llm import OpenAILLM
import os

# Get configuration from .env
api_key = os.getenv("OPENAI_API_KEY")
data_dir = os.getenv("DATA_DIR", "./processed_data")
device = os.getenv("DEVICE", "cpu")

if not api_key:
    raise ValueError("OPENAI_API_KEY environment variable not set")

# Initialize simulator
simulator = Simulator(
    data_dir=data_dir,
    device=device,
    cache=True
)

# Load tasks
simulator.set_task_and_groundtruth(
    task_dir="./track1/yelp/tasks",
    groundtruth_dir="./track1/yelp/groundtruth"
)

# Set agent and LLM
simulator.set_agent(RobustSimulationAgent)
simulator.set_llm(OpenAILLM(api_key=api_key, model="gpt-3.5-turbo"))

# Run
outputs = simulator.run_simulation(
    number_of_tasks=None,
    enable_threading=True,
    max_workers=20
)

# Evaluate
results = simulator.evaluate()
print(results)
```

---

## 🔧 Customization

### Modify Heuristic Weight
```python
# In any agent - change from 0.7/0.3 to your preferred ratio
predicted_stars = (item_avg * 0.7) + (user_avg * 0.3)
#                      ^               ^
#                 Change these values
```

### Adjust LLM Temperature
```python
temperature=0.1  # Conservative (deterministic)
temperature=0.5  # Balanced
temperature=1.0  # Creative (more variance)
```

### Use Different LLM
```python
# GPT-4
simulator.set_llm(OpenAILLM(api_key=api_key, model="gpt-4"))

# Qwen (via Infinigence)
simulator.set_llm(InfinigenceLLM(api_key=api_key, model="qwen2.5-72b-instruct"))
```

---


## 📝 Output Formats

### Agent Output
```json
{
    "stars": 4.5,
    "review": "Excellent service and great atmosphere. Will definitely return!"
}
```

### Evaluation Results (in `logs/`)
```json
{
    "type": "simulation",
    "metrics": {
        "rmse": 1.2,
        "mae": 0.9,
        "correlation": 0.78
    }
}
```
