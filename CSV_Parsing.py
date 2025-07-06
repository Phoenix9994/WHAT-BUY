#csv | all_stocks_with_loadmore
#Goal for this 
"""
I want the Symbol, The Company Name and The Price
We will only return these if the price is within our range

Symbol: String
Name: String- in string we want to remove (Symbol+âˆ’)
Price: Double (we want to remove USD- or not include that anyway since its an assumption)

Given the first 600 Stocks of today companies that meet your budget are

"""

def clean_company_name(row):
    name = row['company_name']
    symbol = row['symbol']
    
    # Remove the weird symbol
    name = name.replace('âˆ’', '')
    
    # Remove the repeated symbol from the start if it exists
    if name.startswith(symbol):
        name = name[len(symbol):]
        
    # Strip whitespace
    return name.strip()


import pandas as pd

# Get User Input
min_val = float(input("Min Value: "))
print("to")
max_val = float(input("Max Value: "))

# Ensure min < max
if max_val < min_val:
    min_val, max_val = max_val, min_val

# Load CSV
df = pd.read_csv('all_stocks_with_loadmore.csv')

# Clean the data column
df["price"] = df["price"].str.replace("USD", "", regex=False)
df["price"] = df["price"].str.replace(" ", "", regex=False)
df["price"] = df["price"].str.replace(",", "", regex=False)
df["price"] = df["price"].astype(float)
df['company_name'] = df.apply(clean_company_name, axis=1)
df['company_name'] = df['company_name'].str.replace('−', '', regex=False).str.strip()

# Filter by price range
filtered_df = df[(df['price'] >= min_val) & (df['price'] <= max_val)]

# Sort the filtered results by price
filtered_df = filtered_df.sort_values(by="price")

selected_columns = filtered_df[['symbol', 'company_name', 'price', 'change']].to_string(index=False)

# Store results in a dictionary using symbol as the key
stock_dict = {}

#Filtered Price Dictionary, low to high
for _, row in filtered_df.iterrows():
    stock_dict[row['symbol']] = {
        'company_name': row['company_name'],
        'price': row['price'],
        'change': row['change']
    }

#Now to use it else where
#will use the stock data
import json
with open('filtered_stocks.json', 'w') as f:
    json.dump(stock_dict, f, indent=2)


# Example: print the first 5 items
#for symbol, data in list(stock_dict.items())[:5]:
  #  print(f"{symbol}: {data['name']} - ${data['price']:.2f} ({data['change']})")

# Print the filtered, sorted results
#print(selected_columns)
