import os
import asyncio
from dotenv import load_dotenv

from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient

load_dotenv()
from autogen_ext.models.openai import AzureOpenAIChatCompletionClient

# Initialize Azure OpenAI model client
model_client = AzureOpenAIChatCompletionClient(
    model=os.getenv("CHAT_MODEL_DEPLOYMENT"),  # The model name (e.g., "gpt-4o")
    azure_deployment=os.getenv("CHAT_MODEL_DEPLOYMENT"),  # Your deployment name
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),  # e.g., "2024-06-01"
    azure_endpoint=os.getenv("AZURE_OPENAI_CHAT_ENDPOINT"),  # Your Azure endpoint
    api_key=os.getenv("AZURE_OPENAI_CHAT_API_KEY")
)


# Async function to run the GPT Generator Agent
async def generate_answer_agent(query: str, context: str, history: str = "") -> str:
    """Generate an answer using Azure OpenAI GPT model via Autogen agent."""
    try:
        # Initialize Azure OpenAI model client
        model_client = AzureOpenAIChatCompletionClient(
            model=os.getenv("CHAT_MODEL_DEPLOYMENT"),
            azure_deployment=os.getenv("CHAT_MODEL_DEPLOYMENT"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
            azure_endpoint=os.getenv("AZURE_OPENAI_CHAT_ENDPOINT"),
            api_key=os.getenv("AZURE_OPENAI_CHAT_API_KEY")
        )

        print("Initializing GPT Generator Agent...")

        # Enhanced system message with greeting handling
        system_message = (
            "You are a friendly HR assistant helping employees with their policy questions.\n\n"
            "CONVERSATION HANDLING:\n"
            "1. GREETINGS & SMALL TALK:\n"
            "   - If user says greetings (hi, hello, hey, good morning, etc.), respond warmly: "
            "'Hello! I'm your HR assistant. How can I help you with policy-related questions today?'\n"
            "   - If user says thanks/thank you, respond: 'You're welcome! Feel free to ask if you have other questions.'\n"
            "   - If user says goodbye/bye, respond: 'Goodbye! Have a great day.'\n"
            "   - For these casual messages, DO NOT use or mention the provided context.\n\n"
            "2. POLICY QUESTIONS:\n"
            "   - Use BOTH conversation history and provided context to answer accurately\n"
            "   - Provide clear, concise responses in 2-3 well-structured sentences\n"
            "   - Never mention file names, page numbers, or sources\n"
            "   - If information is not in the context, say: 'I don't have that information in the available documents.'\n"
            "   - Never guess or add information not present in the context\n\n"
            "3. TONE: Professional, helpful, and grammatically correct at all times."
        )

        # Create the agent
        agent = AssistantAgent(
            name="gpt_generator_agent",
            model_client=model_client,
            system_message=system_message,
        )

        # Construct task prompt (always include context, let agent decide how to use it)
        task_prompt = (
            f"Conversation History:\n{history}\n\n"
            f"Available Context:\n{context}\n\n"
            f"User Message: {query}\n\n"
            f"Respond appropriately based on the type of message (greeting vs policy question):"
        )

        # Run the agent
        response = await agent.run(task=task_prompt)
        await model_client.close()

        # Extract the response content
        if hasattr(response, 'messages') and response.messages:
            answer = response.messages[-1].content.strip()
        elif hasattr(response, 'content'):
            answer = response.content.strip()
        else:
            answer = str(response)

        return answer

    except Exception as e:
        return f"Error generating answer with agent: {str(e)}"
