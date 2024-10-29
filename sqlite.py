import sqlite3

# Connect to SQLite (it creates an in-memory database for checking version)
connection = sqlite3.connect(":memory:")

# Get SQLite version
sqlite_version = connection.execute("SELECT sqlite_version();").fetchone()[0]
print("SQLite version:", sqlite_version)

# Close the connection
connection.close()