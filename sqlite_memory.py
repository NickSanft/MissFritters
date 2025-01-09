import sqlite3
import jsonpickle
from typing import List
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import BaseMessage

# Initialize SQLite connection
DB_NAME = "chat_history.db"

def get_db_connection():
    conn = sqlite3.connect(DB_NAME)
    return conn

# Function to initialize the database
def init_db():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS message_history (
            user_id TEXT,
            conversation_id TEXT,
            messages TEXT,
            PRIMARY KEY (user_id, conversation_id)
        )
    ''')
    conn.commit()
    conn.close()

# In-memory message history implementation (this will still use SQLite now)
class SQLiteHistory(BaseChatMessageHistory):
    user_id: str
    conversation_id: str

    def __init__(self, user_id: str, conversation_id: str):
        self.user_id = user_id
        self.conversation_id = conversation_id

    @property
    def messages(self) -> List[BaseMessage]:
        return self.get_user_messages()

    def get_user_messages(self):
        # Fetch the messages from the SQLite database
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute('''
            SELECT messages FROM message_history
            WHERE user_id = ? AND conversation_id = ?
        ''', (self.user_id, self.conversation_id))
        result = cursor.fetchone()
        conn.close()

        if result:
            print("Current history: {}".format(result[0]))
            return jsonpickle.loads(result[0]) # Serialize the messages
        return []

    def add_messages(self, messages: List[BaseMessage]) -> None:
        for message in messages:
            message.pretty_print()
        existing_messages = self.get_user_messages()
        print("user_id: {} conversation_id: {}".format(self.user_id, self.conversation_id))
        print("Existing messages: {}".format(existing_messages))
        print("New messages: {}".format(messages))
        existing_messages.extend(messages)
        # Serialize messages using jsonpickle
        serialized_messages = jsonpickle.dumps(existing_messages)

        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute('''
            INSERT OR REPLACE INTO message_history (user_id, conversation_id, messages)
            VALUES (?, ?, ?)
        ''', (self.user_id, self.conversation_id, serialized_messages))
        conn.commit()
        conn.close()

    def clear(self) -> None:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute('''
            DELETE FROM message_history WHERE user_id = ? AND conversation_id = ?
        ''', (self.user_id, self.conversation_id))
        conn.commit()
        conn.close()

# Function to get session history (now using SQLite)
def get_session_history_persistent_db_memory(user_id: str, conversation_id: str) -> SQLiteHistory:
    history = SQLiteHistory(user_id=user_id, conversation_id=conversation_id)
    return history

# Initialize the database (this should be run once to set up the database)
init_db()