import sqlite3
conn = sqlite3.connect("b3_market_data.sqlite")
cursor = conn.cursor()

print("Dropping existing corporate_actions tables...")
cursor.execute("DROP TABLE IF EXISTS corporate_actions")
cursor.execute("DROP TABLE IF EXISTS stock_actions")
cursor.execute("DROP TABLE IF EXISTS detected_splits")

print("Recreating schemas...")
from b3_pipeline.storage import SCHEMA_CORPORATE_ACTIONS, SCHEMA_STOCK_ACTIONS, SCHEMA_DETECTED_SPLITS
cursor.execute(SCHEMA_CORPORATE_ACTIONS)
cursor.execute(SCHEMA_STOCK_ACTIONS)
cursor.execute(SCHEMA_DETECTED_SPLITS)

conn.commit()
print("Done!")
conn.close()
