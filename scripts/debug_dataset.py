import json 
from collections import Counter 

with open("data/processed/unified_dataset.json") as f:
     data = json.load(f) 

sources = Counter(item["source"] for item in data) 
print("Samples per source:")
for src, count in sources.items():
     print(f" {src}: {count}") 
     
print(f"\nTotal: {len(data)}")