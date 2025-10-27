from flask import Blueprint, request, jsonify
import asyncio

from retrieval.retrieve_doc import DocumentRetriever
from retrieval.generator_agent import generate_answer_agent  # Import the async function

api = Blueprint('api', __name__)

# Initialize RAG components
retriever = DocumentRetriever()

# Simple in-memory conversation history
conversation_history = []

@api.route('/answer', methods=['POST'])
def answer_question():
    try:
        data = request.get_json(force=True)
        query = data.get("query", "").strip()

        if not query:
            return jsonify({"error": "Query is required"}), 400

        # Retrieve top 3 relevant documents
        results = retriever.retrieve_documents(query, top_k=3)

        if not results:
            return jsonify({
                "answer": "I don't have that information in the available documents.",
                "sources": []
            })

        # Combine content for context
        context = "\n\n".join([doc.get("content", "") for doc in results])

        # Add previous conversation history to context
        history_text = ""
        if conversation_history:
            history_text = "\n".join([
                f"Q: {item['query']}\nA: {item['answer']}"
                for item in conversation_history[-5:]
            ])

        # Generate answer using async agent
        answer = asyncio.run(generate_answer_agent(query, context, history_text))

        # Handle short answers
        if len(answer.split()) < 10:
            relevant_sentences = [
                s for s in context.split(".")
                if any(word in s.lower() for word in query.lower().split())
            ]
            answer = f"Here’s what I found: {relevant_sentences[0].strip()}." if relevant_sentences else \
                     "I don't have that information in the available documents."

        # Deduplicate sources by (source, page)
        sources = []
        seen = set()
        for doc in results:
            source = doc.get("source", "Unknown")
            page = doc.get("page", "N/A")
            score = doc.get("similarity_score", 0.0)
            key = (source, page)
            if key not in seen:
                sources.append(f"{source} (Page {page}, Score: {score:.4f})")
                seen.add(key)

        # Save this turn in memory
        conversation_history.append({"query": query, "answer": answer})

        return jsonify({"answer": answer, "sources": sources})

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": "Internal Server Error"}), 500
