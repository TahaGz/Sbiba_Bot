# Chatbot with FAISS and Ollama Mistral

## 📌 Documentation
This document explains the logic and components of the chatbot that leverages **FAISS (Facebook AI Similarity Search)** and **Ollama Mistral LLM** to perform **retrieval-augmented generation (RAG)**. The chatbot retrieves relevant information from a **vector store** and generates a fact-based response.

---

## 📦 Dependencies
To run this chatbot, you need to install the following dependencies:

```bash
pip install langchain_community langchain_ollama faiss-cpu numpy langdetect pickle5 nltk
```

### 🔹 Required Libraries
| Library | Purpose |
|---------|---------|
| `langchain_community` | Provides FAISS vector store functionality |
| `langchain_ollama` | Embedding model and LLM for generating responses |
| `faiss-cpu` | Efficient similarity search for vector embeddings |
| `numpy` | Handles embedding array transformations |
| `pickle5` | Stores metadata (answers) efficiently |
| `langdetect` | Detects the language of user queries |
| `nltk` | Used for **BLEU score** evaluation of LLM responses |

---

## 🧩 Components and Concepts
The chatbot follows a **retrieval-augmented generation (RAG)** pipeline, which consists of the following steps:

### **1. FAISS Vector Store**
- **Concept:** A similarity search system that allows efficient retrieval of stored embeddings.
- **Implementation:** The chatbot loads a pre-built FAISS index (`faiss_index`) and metadata (`faiss_metadata.pkl`).
- **Why FAISS?** FAISS enables fast nearest neighbor search, making it ideal for retrieving similar text-based embeddings.

### **2. Ollama Embeddings**
- **Concept:** Converts text (answers) into vector embeddings using `nomic-embed-text`.
- **Implementation:**
  ```python
  embedding_model = OllamaEmbeddings(model="nomic-embed-text")
  ```
- **Why embeddings?** They capture semantic meaning, allowing **context-aware** searches rather than keyword-based retrieval.

### **3. Context Retrieval (Retrieve-Only RAG)**
- **Concept:** Finds the most relevant answer embeddings to the user query.
- **Implementation:** Uses FAISS to retrieve top-k closest embeddings based on **L2 distance**.
- **Threshold Mechanism:** Ensures that only relevant results within a similarity threshold are included.

### **4. Ollama Mistral LLM (Language Model)**
- **Concept:** Uses the Mistral LLM to generate responses based on retrieved information.
- **Implementation:**
  ```python
  llm = OllamaLLM(model="mistral")
  ```
- **Why Mistral?** It is optimized for fast and accurate text generation, making it suitable for conversational AI.

### **5. Language Detection**
- **Concept:** Determines whether the query is in English or French.
- **Implementation:**
  ```python
  lang = detect(user_query)
  ```
- **Why Language Detection?** Ensures responses are provided in the correct language.

### **6. Prompt Engineering**
- **Concept:** Uses structured prompts to ensure responses are concise and fact-based.
- **Implementation:** Different prompts for English and French to guide the LLM.

### **7. Evaluation Methods**
- **Recall@K:** Checks whether FAISS retrieves the expected answer.
- **BLEU Score:** Measures how similar the LLM-generated response is to the expected answer.
- **Latency Measurement:** Tests response time to ensure efficiency.

---

## 🔄 Alternatives
### **1. Alternative Vector Stores**
| Vector Store | Pros | Cons |
|-------------|------|------|
| FAISS | Fast, scalable | Requires memory optimization for large datasets |
| Pinecone | Cloud-based, scalable | Paid service, requires API key |
| ChromaDB | Simple, easy-to-use | May not be as optimized for large-scale retrieval |
| Weaviate | Supports hybrid search | More setup required |

### **2. Alternative Embedding Models**
| Model | Pros | Cons |
|-------------|------|------|
| `nomic-embed-text` | Fast, efficient | Less customizable than OpenAI embeddings |
| `OpenAI embeddings` | High accuracy, large model | Requires API key, paid |
| `SentenceTransformers` | Good for sentence-based embeddings | Slightly slower |

### **3. Alternative LLMs**
| LLM | Pros | Cons |
|-------------|------|------|
| Mistral (Ollama) | Fast, lightweight | May lack domain-specific knowledge |
| OpenAI GPT-4 | Highly accurate | Paid service |
| LLaMA 2 | Open-source | Requires more resources |
| Claude (Anthropic) | Good at reasoning | Requires API key |

---

## 🎯 Conclusion
This chatbot leverages **FAISS + Ollama Mistral** to create an efficient **retrieval-augmented generation (RAG)** pipeline. It retrieves stored FAQ answers, ensuring fast and context-aware responses.

### 🚀 **Next Steps**
- Improve retrieval quality by **fine-tuning embeddings**.
- Implement **query expansion** to improve recall.
- Add support for **multimodal retrieval** (e.g., image + text search).

---

🔗 **References**
- FAISS: [https://faiss.ai/](https://faiss.ai/)
- LangChain Docs: [https://python.langchain.com/](https://python.langchain.com/)
- Ollama: [https://ollama.ai/](https://ollama.ai/)
