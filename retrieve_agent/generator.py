"""
Generator Module
================
Handles the GENERATION part of RAG:
- Takes user query, retrieved context, and optional conversation history
- Uses Azure OpenAI GPT model to generate accurate answers
- Applies prompt engineering and fallback logic

Author: HR RAG Bot Team
Date: October 2025
"""

from openai import AzureOpenAI
from utils.setting import (
    AZURE_OPENAI_CHAT_ENDPOINT,
    AZURE_OPENAI_CHAT_API_KEY,
    CHAT_MODEL_DEPLOYMENT,
    AZURE_OPENAI_API_VERSION
)

class GPTGenerator:
    """Generates answers using Azure OpenAI GPT model based on context and conversation history."""

    def __init__(self):
        print(" Initializing GPT Generator...")
        self.chat_client = AzureOpenAI(
            azure_endpoint=AZURE_OPENAI_CHAT_ENDPOINT,
            api_key=AZURE_OPENAI_CHAT_API_KEY,
            api_version=AZURE_OPENAI_API_VERSION
        )
        print(" GPT Generator initialized successfully!\n")

    def generate_answer(self, query: str, context: str, history: str = "") -> str:
        """
        Generate an answer using GPT model based on context and optional conversation history.
        - If answer exists: descriptive response (2–3 sentences)
        - If partial info: summarize available details
        - If nothing relevant: say 'I don't have that information in the available documents.'
        """
        if not context or "No relevant documents" in context:
            return "I don't have that information in the available documents."

        # Improved system prompt
        system_prompt = (
            "You are an HR assistant. Use BOTH the conversation history and provided context to answer. "
            "Do not repeat sources or mention file names. "
            "Give a clear, accurate response in 2–3 sentences. "
            "If nothing relevant is found, say: 'I don't have that information in the available documents.' "
            "Never guess or add information not in the context."
        )

        # Combine history and context
        user_prompt = (
            f"Conversation History:\n{history}\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {query}\nAnswer:"
        )

        try:
            response = self.chat_client.chat.completions.create(
                model=CHAT_MODEL_DEPLOYMENT,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.0,  # Keep answers factual
                max_tokens=500
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"Error generating answer: {e}"
 