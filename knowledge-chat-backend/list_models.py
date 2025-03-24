#!/usr/bin/env python3
from src.dbt_tools import DbtTools
import re

# Initialize the tools
tools = DbtTools('https://github.com/dbt-labs/dbt-cloud-snowflake-demo-template.git', '', '')
repo_path = tools.initialize()

print(f"Repo path: {repo_path}")
print("Looking for sales-related models...\n")

# List all models
print("ALL MODELS:")
for model_name, path in tools.file_scanner.model_files.items():
    print(f"- {model_name}: {path}")

# Search for sales-related terms
search_terms = [
    "sales", "order", "item", "discount", "tax", "gross", "net", "amount"
]

print("\nSEARCHING FOR SALES-RELATED MODELS AND COLUMNS:")
for model_name, path in tools.file_scanner.model_files.items():
    try:
        full_path = f"{repo_path}/{path}"
        with open(full_path, 'r') as f:
            content = f.read().lower()
            
        # Check if any term exists in the content
        matches = []
        for term in search_terms:
            if term in content:
                matches.append(term)
        
        if matches:
            print(f"\nModel: {model_name}")
            print(f"File: {path}")
            print(f"Contains terms: {', '.join(matches)}")
            
            # Look for columns with calculation patterns
            calc_patterns = [
                r'(\w+)\s*=\s*.*?(?:amount|sales|discount|tax)',  # Variable assignments
                r'as\s+(\w+_(?:amount|sales|discount|tax)\w*)',   # Column aliases with amount/sales/etc.
                r'as\s+([\'"]?\w+_(?:amount|sales|discount|tax)\w*[\'"]?)',  # With quotes
                r'(\w+_(?:amount|sales|discount|tax)\w*)',        # Just column names
                r'(\w+_(?:gross|net)\w*)'                         # gross or net columns
            ]
            
            for pattern in calc_patterns:
                column_matches = re.findall(pattern, content, re.IGNORECASE)
                if column_matches:
                    unique_columns = list(set(column_matches))
                    print(f"Potential columns: {', '.join(unique_columns)}")
    except Exception as e:
        print(f"Error processing {model_name}: {str(e)}")

# Search for specific columns
print("\n\nSEARCHING FOR YOUR SPECIFIC COLUMNS:")
specific_columns = [
    "gross_item_sales_amount", 
    "item_discount_amount",
    "item_tax_amount", 
    "net_item_sales_amount"
]

for column in specific_columns:
    print(f"\nLooking for '{column}'...")
    results = tools.search_content(column)
    if results['status'] == 'success':
        print(f"  Found in {len(results['results'])} models")
        for result in results['results']:
            print(f"  - Model: {result['model_name']}")
            print(f"    File: {result['file_path']}")
            # Show context - just the relevant part
            context = result['match_context']
            # Try to find the most relevant part (near the column name)
            lines = context.split('\n')
            relevant_lines = []
            for line in lines:
                if column.lower() in line.lower() or any(term in line.lower() for term in ["select", "case", "when", "sum", "as "]):
                    relevant_lines.append(line.strip())
            if relevant_lines:
                print(f"    Context:")
                for line in relevant_lines[:5]:  # Show up to 5 relevant lines
                    print(f"      {line}")
    else:
        print(f"  Not found directly.")
        
        # Try a broader search by parts of the name
        parts = column.split('_')
        broader_results = False
        
        for i in range(len(parts)):
            if len(parts[i]) > 3 and parts[i] not in ['item', 'amount']:
                print(f"  Searching more broadly for '{parts[i]}'...")
                results = tools.search_content(parts[i])
                if results['status'] == 'success':
                    broader_results = True
                    print(f"  Found term '{parts[i]}' in {len(results['results'])} models")
                    for result in results['results']:
                        print(f"  - Model: {result['model_name']}")
                        print(f"    File: {result['file_path']}")
        
        if not broader_results:
            print("  No matches found even with broader search.") 