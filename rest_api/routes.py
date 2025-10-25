# from flask import Blueprint, request, jsonify
# from retrieve_agent.retrieve_doc import DocumentRetriever
# from retrieve_agent.generator import GPTGenerator

# api = Blueprint('api', __name__)

# retriever = DocumentRetriever()
# generator = GPTGenerator()

# @api.route('/answer', methods=['POST'])
# def answer_question():
#     try:
#         data = request.get_json(force=True)
#         query = data.get("query", "").strip()

#         if not query:
#             return jsonify({"error": "Query is required"}), 400

#         # Retrieve top 3 relevant documents
#         results = retriever.retrieve_documents(query, top_k=3)

#         if not results:
#             return jsonify({
#                 "answer": "I don't have that information in the available documents.",
#                 "sources": []
#             })

#         # Combine content for context
#         context = "\n\n".join([doc.get("content", "") for doc in results])
#         answer = generator.generate_answer(query, context)

#         # Handle short answers by extracting relevant sentences
#         if len(answer.split()) < 10:
#             relevant_sentences = [
#                 s for s in context.split(".")
#                 if any(word in s.lower() for word in query.lower().split())
#             ]
#             answer = f"Here’s what I found: {relevant_sentences[0].strip()}." if relevant_sentences else \
#                      "I don't have that information in the available documents."

#         # Deduplicate sources by (source, page)
#         sources = []
#         seen = set()
#         for doc in results:
#             source = doc.get("source", "Unknown")
#             page = doc.get("page", "N/A")
#             score = doc.get("similarity_score", 0.0)
#             key = (source, page)
#             if key not in seen:
#                 sources.append(f"{source} (Page {page}, Score: {score:.4f})")
#                 seen.add(key)

#         return jsonify({"answer": answer, "sources": sources})

#     except Exception as e:
#         print(f"Error: {e}")
#         return jsonify({"error": "Internal Server Error"}), 500

from flask import Blueprint, request, jsonify
from retrieve_agent.retrieve_doc import DocumentRetriever
from retrieve_agent.generator import GPTGenerator

api = Blueprint('api', __name__)

# Initialize RAG components
retriever = DocumentRetriever()
generator = GPTGenerator()

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
        if conversation_history:
            history_text = "\n".join([f"Q: {item['query']}\nA: {item['answer']}" for item in conversation_history[-5:]])  # last 5 turns
            context = history_text + "\n\n" + context

        # Generate answer
        answer = generator.generate_answer(query, context)

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