import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from supabase import create_client

url = "https://bbkwerllrsqlezrzxqqf.supabase.co"
key = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImJia3dlcmxscnNxbGV6cnp4cXFmIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NTYyODM2NzEsImV4cCI6MjA3MTg1OTY3MX0.-s6W-R_fg0JnE_-CUqtA8i6SSjSIlFaqbVb3k6R85Kg"
supabase = create_client(url, key)
def fetch_all_rows(table_name, batch_size=1000):
    """Fetch all rows from a Supabase table with pagination"""
    all_data = []
    offset = 0
    
    while True:
        response = supabase.table(table_name).select("*").range(offset, offset + batch_size - 1).execute()
        batch_data = response.data
        
        if not batch_data:
            break
            
        all_data.extend(batch_data)
        
        if len(batch_data) < batch_size:
            break
            
        offset += batch_size
    
    return pd.DataFrame(all_data)

dim_clients = fetch_all_rows("dim_clients")

print(dim_clients.head())
