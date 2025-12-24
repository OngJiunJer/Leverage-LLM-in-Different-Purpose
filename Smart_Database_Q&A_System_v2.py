#-------------------------------#
# Step 1: Define the Database Schema (Text for LLM)
#-------------------------------#
# Write down all the definition of each column
SCHEMA = """
Table: best_selling_game_consoles

Columns:
- Console_Name (STRING): Name of the game console
- Type (STRING): Home console, handheld console, etc.
- Company (STRING): Manufacturer
- Released_Year (INTEGER): First release year
- Discontinuation_Year (INTEGER): Discontinued year (nullable)
- Units_Sold (DECIMAL): Units sold worldwide (millions)
- Remarks (STRING): Additional notes
"""

# Rule 
RULE = """
You are an expert SQL assistant.
Rules:
- Use ONLY the tables and columns provided
- Do NOT invent tables or columns
- Use SQLite-compatible SQL
- Do NOT include explanations
"""

# User Question
user_question = """Generate an SQL query to calculate the average Units_Sold for consoles by each company."""

#-------------------------------#
# Step 2: Combine SCHEMA & Rules & User Question to Prompt
#-------------------------------#
def built_prompt(user_question):
    prompt = f"""
{RULE}

Schema:
{SCHEMA}

User Question:
{user_question}
"""
    return prompt

question = built_prompt(user_question)
print(question)

#-------------------------------#
# Step 3: Generate SQL Query
#-------------------------------#
# Install on terminal
# pip install google-genai

from google import genai
from google.genai import types

# Set API key
client = genai.Client(
    api_key="AIzaSyAwaJiEdHIMh7NSjKnpfb88ibt0epTBC-w",
    http_options=types.HttpOptions(api_version='v1beta')
)

def generate_sql(prompt):
    try:
        response = client.models.generate_content(
            model="gemini-3-flash-preview",
            contents=prompt
        )
        return response.text.strip()  # new field for generated text
    except Exception as e:
        return f"Error: {e}"

# Example usage
sql_query = generate_sql(question)
print(sql_query)

# Check the SQL Quey and rewrite it if needed
check_sql_query = "SELECT Company, AVG(Units_Sold) FROM best_selling_game_consoles GROUP BY Company;"

#-------------------------------#
# Step 4: Run SQL Query (CSV File Version)
#-------------------------------#
import sqlite3
import pandas as pd

# Ensure Column Name Correct Format
df = pd.read_csv("best_selling_game_consoles.csv")
print(df.columns)
df.columns = [c.replace(" ", "_") for c in df.columns]  # convert spaces to underscores
df.rename(columns={"Units_sold_(million)": "Units_Sold"}, inplace=True)


# Create connection and import csv file data
conn = sqlite3.connect("game_data.db")
df.to_sql("best_selling_game_consoles", conn, if_exists="replace", index=False)

# Execute SQL Query
cursor = conn.cursor()
cursor.execute(check_sql_query)
results = cursor.fetchall()
print(results)

# Save & Close
conn.commit()
conn.close()






























