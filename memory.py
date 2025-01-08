from typing import List
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import BaseMessage, AIMessage
from pydantic import BaseModel, Field

store = {}

# In-memory message history implementation
class InMemoryHistory(BaseChatMessageHistory, BaseModel):
    messages: List[BaseMessage] = Field(default_factory=list)

    def add_messages(self, messages: List[BaseMessage]) -> None:
        for message in messages:
            if isinstance(message, AIMessage) and message.tool_calls:
                first_result = message.tool_calls[0]
                if first_result and first_result['name'] == "respond_to_user":
                    message.content = first_result['args']['content']
            self.messages.append(message)

    def clear(self) -> None:
        self.messages.clear()

# Function to get session history
def get_session_history(user_id: str, conversation_id: str) -> BaseChatMessageHistory:
    if (user_id, conversation_id) not in store:
        store[(user_id, conversation_id)] = InMemoryHistory()
    return store[(user_id, conversation_id)]