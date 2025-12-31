# ==========================
# Standard Library Imports
# ==========================
import sqlite3

# ==========================
# Third-Party Libraries
# ==========================
from langgraph.checkpoint.sqlite import SqliteSaver

# ==========================
# SQLite Checkpoint Store
# ==========================
conn = sqlite3.connect(database="chatbot.db", check_same_thread=False)
checkpointer = SqliteSaver(conn=conn)