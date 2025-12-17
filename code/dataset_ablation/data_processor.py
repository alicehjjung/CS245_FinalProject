import json
import logging
import uuid
import pandas as pd
from tqdm import tqdm
import os

# Configure logging to show progress
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

def load_data(file_path):
    """Load JSON data into a Pandas DataFrame."""
    data = []
    if not os.path.exists(file_path):
        logging.warning(f"File not found: {file_path}")
        return pd.DataFrame()
        
    with open(file_path, 'r') as file:
        for line in tqdm(file, desc=f"Loading {os.path.basename(file_path)}", unit=" lines"):
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return pd.DataFrame(data)

def load_yelp_data(input_dir):
    """Load Yelp data specifically."""
    logging.info("Loading Yelp data...")
    b_df = load_data(os.path.join(input_dir, 'yelp_academic_dataset_business.json'))
    u_df = load_data(os.path.join(input_dir, 'yelp_academic_dataset_user.json'))
    r_df = load_data(os.path.join(input_dir, 'yelp_academic_dataset_review.json'))
    
    # Filter for top cities to keep dataset manageable
    top_cities = ['Philadelphia', 'Tampa', 'Tucson']
    if not b_df.empty:
        b_df = b_df[b_df['city'].isin(top_cities)]
        r_df = r_df[r_df['business_id'].isin(b_df['business_id'])]
        u_df = u_df[u_df['user_id'].isin(r_df['user_id'])]
        
    return b_df, r_df, u_df

def load_amazon_data(input_dir):
    """Load Amazon data."""
    logging.info("Loading Amazon data...")
    # Adjust filenames as needed based on your actual files
    rating_files = ['Industrial_and_Scientific.csv', 'Musical_Instruments.csv'] # Add others if needed
    review_files = ['Industrial_and_Scientific.jsonl', 'Musical_Instruments.jsonl']
    meta_files = ['meta_Industrial_and_Scientific.jsonl', 'meta_Musical_Instruments.jsonl']
    
    # Load Ratings
    dfs = []
    for f in rating_files:
        path = os.path.join(input_dir, f)
        if os.path.exists(path):
            dfs.append(pd.read_csv(path))
    rating_df = pd.concat(dfs) if dfs else pd.DataFrame(columns=['user_id', 'parent_asin'])
    
    if rating_df.empty: return pd.DataFrame(), pd.DataFrame()

    users = rating_df['user_id'].unique()
    items = rating_df['parent_asin'].unique()

    # Load Reviews
    rev_dfs = []
    for f in review_files:
        path = os.path.join(input_dir, f)
        if os.path.exists(path):
            df = load_data(path)
            # Filter
            df = df[df['user_id'].isin(users) & df['parent_asin'].isin(items)]
            rev_dfs.append(df)
    reviews_df = pd.concat(rev_dfs) if rev_dfs else pd.DataFrame()

    # Load Meta
    meta_dfs = []
    for f in meta_files:
        path = os.path.join(input_dir, f)
        if os.path.exists(path):
            df = load_data(path)
            df = df[df['parent_asin'].isin(items)]
            meta_dfs.append(df)
    meta_df = pd.concat(meta_dfs) if meta_dfs else pd.DataFrame()
    
    return reviews_df, meta_df

def load_goodreads_data(input_dir):
    """Load Goodreads data."""
    logging.info("Loading Goodreads data...")
    book_files = ['goodreads_books_children.json'] # Add others
    review_files = ['goodreads_reviews_children.json'] # Add others
    
    b_dfs = [load_data(os.path.join(input_dir, f)) for f in book_files if os.path.exists(os.path.join(input_dir, f))]
    r_dfs = [load_data(os.path.join(input_dir, f)) for f in review_files if os.path.exists(os.path.join(input_dir, f))]
    
    books = pd.concat(b_dfs) if b_dfs else pd.DataFrame()
    reviews = pd.concat(r_dfs) if r_dfs else pd.DataFrame()
    
    return books, reviews

def generate_dataset(output_dir, yelp_data=None, amazon_data=None, goodreads_data=None):
    """
    Generates unified item.json, review.json, user.json based on provided inputs.
    Any input set to None will be excluded.
    
    Args:
        yelp_data: (business_df, review_df, user_df)
        amazon_data: (review_df, meta_df)
        goodreads_data: (book_df, review_df)
    """
    os.makedirs(output_dir, exist_ok=True)
    
    all_items = []
    all_reviews = []
    all_users = []

    # --- PROCESS YELP ---
    if yelp_data:
        b_df, r_df, u_df = yelp_data
        if not b_df.empty:
            # Items
            temp_b = b_df.rename(columns={'business_id': 'item_id'})
            temp_b['source'] = 'yelp'
            temp_b['type'] = 'business'
            all_items.extend(temp_b.to_dict('records'))
            
            # Reviews
            temp_r = r_df.rename(columns={'business_id': 'item_id'})
            temp_r['source'] = 'yelp'
            temp_r['type'] = 'business'
            all_reviews.extend(temp_r.to_dict('records'))
            
            # Users
            temp_u = u_df.copy()
            temp_u['source'] = 'yelp'
            all_users.extend(temp_u.to_dict('records'))

    # --- PROCESS AMAZON ---
    if amazon_data:
        r_df, m_df = amazon_data
        if not m_df.empty:
            # Items
            temp_m = m_df.rename(columns={'parent_asin': 'item_id'})
            temp_m['source'] = 'amazon'
            temp_m['type'] = 'product'
            all_items.extend(temp_m.to_dict('records'))
            
            # Reviews
            if not r_df.empty:
                temp_r = r_df.rename(columns={'parent_asin': 'item_id', 'rating': 'stars'})
                temp_r['source'] = 'amazon'
                temp_r['type'] = 'product'
                temp_r['review_id'] = [str(uuid.uuid4()) for _ in range(len(temp_r))]
                all_reviews.extend(temp_r.to_dict('records'))
                
                # Users (Amazon doesn't have a user profile file usually, so we make dummy ones)
                unique_users = r_df['user_id'].unique()
                for uid in unique_users:
                    all_users.append({'user_id': uid, 'source': 'amazon'})

    # --- PROCESS GOODREADS ---
    if goodreads_data:
        b_df, r_df = goodreads_data
        if not b_df.empty:
            # Items
            temp_b = b_df.rename(columns={'book_id': 'item_id'})
            temp_b['source'] = 'goodreads'
            temp_b['type'] = 'book'
            all_items.extend(temp_b.to_dict('records'))
            
            # Reviews
            if not r_df.empty:
                temp_r = r_df.rename(columns={'book_id': 'item_id', 'rating': 'stars', 'review_text': 'text'})
                temp_r['source'] = 'goodreads'
                temp_r['type'] = 'book'
                all_reviews.extend(temp_r.to_dict('records'))
                
                # Users
                unique_users = r_df['user_id'].unique()
                for uid in unique_users:
                    all_users.append({'user_id': uid, 'source': 'goodreads'})

    # --- SAVE ---
    logging.info(f"Saving merged data to {output_dir}")
    pd.DataFrame(all_items).to_json(os.path.join(output_dir, 'item.json'), orient='records', lines=True)
    pd.DataFrame(all_reviews).to_json(os.path.join(output_dir, 'review.json'), orient='records', lines=True)
    pd.DataFrame(all_users).to_json(os.path.join(output_dir, 'user.json'), orient='records', lines=True)