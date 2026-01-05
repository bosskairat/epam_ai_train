class ConversationContext:
    
    def __init__(self):
        self.history = []


    def add(self, role: str, content: str):
        self.history.append({"role": role, "content": content})


    def last_user_message(self):
        for msg in reversed(self.history):
            if msg["role"] == "user":
                return msg["content"]
        return ""
    

    def clear(self):
        self.history = []