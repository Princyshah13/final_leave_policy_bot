from flask import Blueprint, request, jsonify
from retrieve_agent.retrieve_doc import DocumentRetriever
from retrieve_agent.generator import GPTGenerator

api = Blueprint('api', __name__)

# Initialize RAG components
retriever = DocumentRetriever()
generator = GPTGenerator()

@api.route('/answer', methods=['POST'])
def answer_question():
    try:
        data = request.get_json(force=True)
        query = data.get("query", "").strip()

        if not query:
            return jsonify({"error": "Query is required"}), 400

        results = retriever.retrieve_documents(query, top_k=3)

        if not results:
            return jsonify({
                "answer": "I don't have that information in the available documents.",
                "sources": []
            })

        context = "\n\n".join([doc.get("content", "") for doc in results])
        answer = generator.generate_answer(query, context)

        if len(answer.split()) < 10:
            relevant_sentences = [
                s for s in context.split(".")
                if any(word in s.lower() for word in query.lower().split())
            ]
            answer = f"Here’s what I found: {relevant_sentences[0].strip()}." if relevant_sentences else \
                     "I don't have that information in the available documents."

        sources = list({doc.get("source", "Unknown") for doc in results})

        return jsonify({"answer": answer, "sources": sources})

    except Exception as e:
        return jsonify({"error": f"Internal Server Error: {str(e)}"}), 500