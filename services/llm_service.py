import os
from openai import OpenAI

class LLMService:
    def __init__(self):
        self.api_key = os.getenv("GROQ_API_KEY")
        
        if not self.api_key:
            print("Warning: GROQ_API_KEY not set")
            self.client = None
        else:
            self.client = OpenAI(
                    base_url="https://api.groq.com/openai/v1",
                    api_key=self.api_key
                )
    
    async def get_answer(self, query: str) -> str:
        try:
            
            response = self.client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful AI assistant. Provide clear, concise, and informative answers to user queries, structure it as a direct short paragraph."
                    },
                    {
                        "role": "user",
                        "content": query
                    }
                ]
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            print(f"Error getting LLM answer: {e}")    
