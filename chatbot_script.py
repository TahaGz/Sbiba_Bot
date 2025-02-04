from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaLLM
from langchain_ollama import OllamaEmbeddings
from langdetect import detect
import pickle
import faiss
import numpy as np

# Load FAISS index & metadata
embedding_model = OllamaEmbeddings(model="nomic-embed-text")
index = faiss.read_index("faiss_index")

# Load stored answers (metadata)
with open("faiss_metadata.pkl", "rb") as f:
    stored_answers = pickle.load(f)  # Only answers, no questions

# Load Ollama Mistral LLM
llm = OllamaLLM(model="mistral")




# **RAG Pipeline: Retrieve Only Context**
def retrieve_context(user_query, threshold=2):
    """Converts user query into an embedding, searches FAISS, and retrieves relevant context only."""
    query_embedding = embedding_model.embed_query(user_query)

    top_k = 2 if len(user_query) < 50 else 3  # Adjust dynamically based on query length
    distances, indices = index.search(np.array([query_embedding]), k=top_k)

    # Filter results based on threshold
    relevant_texts = []
    for dist, idx in zip(distances[0], indices[0]):
        if dist < threshold:  # Consider only results within threshold
            relevant_texts.append(stored_answers[idx])
    

    # print("\n Retrieved Context:", relevant_texts)  # Debugging print

    # Merge multiple retrieved documents into a single context block
    return "\n".join(relevant_texts)




# **Generate Response Using RAG**
def get_response(user_query):
    lang = detect(user_query)

    # Retrieve relevant context using FAISS
    context = retrieve_context(user_query)
    print("This is the context: ", context)
    # print(context) #For debugging
    # Define prompts
    prompt_fr = f"""
Vous êtes un guide expert de Sbiba, Kasserine. Répondez uniquement en fonction du contexte fourni.

**Directives :**
- Répondez de manière concise (**40 mots maximum**).
- Si aucune information pertinente n’est disponible, seulement répondez :  
  *"Désolé, je n’ai pas d’information à ce sujet."*
- Évitez toute spéculation ; basez votre réponse uniquement sur des faits.

---
### Contexte :
{context}

---
### Question de l'utilisateur :
{user_query}

---
### Réponse (**40 mots maximum**) :
"""


    prompt_en = f"""
You are an expert guide for Sbiba, Kasserine. Answer based strictly on the provided context.

**Guidelines:**
- Respond concisely (Max **40 words**).
- If no relevant information is found, only say:  
  *"Sorry, I don't have information on that."*
- Avoid speculation; base your answer on facts only.

---
### Context:
{context}

---
### User's Question:
{user_query}

---
### Response (Max 40 words):
"""

    # Generate response using Ollama Mistral
    prompt = prompt_en if lang == 'en' else prompt_fr
    response = llm.invoke(prompt)

    return {"answer": response}


#Test the chatbot: French or English
user_query = "What is Sbiba famous for?"
test_response = get_response(user_query)
print("\n Chat Bot Response:", test_response["answer"])







#Evaluate FAISS: Recall@k :
# def evaluate_retrieval(user_query, expected_answer, top_k=3):
#     """Check if FAISS retrieves the correct answer"""
#     retrieved_texts = retrieve_context(user_query)  # Get retrieved context
    
#     # Check if expected answer appears in retrieved results
#     match = any(expected_answer.lower() in text.lower() for text in retrieved_texts)
    
#     return {"query": user_query, "retrieved_context": retrieved_texts, "correct_retrieval": match}

# #Run evaluate_retrieval:
# test_case = evaluate_retrieval("Where is Sbiba?", "Sbiba is in Tunisia")
# print(test_case)

# #BLUE score: LLM response quality
# from nltk.translate.bleu_score import sentence_bleu

# def evaluate_llm_response(user_query, expected_answer):
#     """Compares LLM response to expected answer using BLEU Score"""
#     model_response = get_response(user_query)["answer"]
    
#     # Tokenize expected and generated responses
#     reference = expected_answer.lower().split()
#     candidate = model_response.lower().split()
    
#     bleu_score = sentence_bleu([reference], candidate)
    
#     return {"query": user_query, "model_response": model_response, "BLEU_score": bleu_score}

# test_case = evaluate_llm_response("Where is Sbiba?", "Sbiba is in Tunisia")
# print(test_case)


# import time

# def measure_latency(user_query):
#     """Measures how long it takes to get a response"""
#     start_time = time.time()
#     _ = get_response(user_query)
#     end_time = time.time()
    
#     return {"query": user_query, "response_time_sec": end_time - start_time}

# # Test latency
# latency_test = measure_latency("Tell me about the history of Sbiba")
# print(latency_test)
