# Architecture Overview: Gmail Article Search

**System Diagram:**

```mermaid
graph LR
    A[Streamlit UI] --> B{AG Grid}
    B --> C[Search Query];
    C --> D[Search Agent];
    D --> E[Vector Database (Postgres/pgvector)];
    E --> D;
    A --> F[Gmail Agent];
    F --> G[Gmail API];
    G --> H[Email Data];
    H --> I[Article Extractor];
    I --> J[Article Summarizer (Gemini)];
    J --> K[Embedding Generator (HF Transformers)];
    K --> E;
```

**Components:**

*   **Streamlit UI:**  A user-friendly interface for interacting with the system.
*   **AG Grid:**  A data grid component for displaying search results.
*   **Search Agent:**  A language model agent that handles search queries and retrieves relevant articles from the vector database.
*   **Vector Database:**  A database that stores article summaries and their embeddings for efficient semantic search.
*   **Gmail Agent:**  An agent that fetches "Medium Daily Digest" emails from the user's Gmail account.
*   **Gmail API:**  The API used to access the user's Gmail account.
*   **Article Extractor:**  A component that extracts article titles, links, and content from the Gmail emails.
*   **Article Summarizer:**  A language model agent that generates concise summaries of the extracted articles.
*   **Embedding Generator:**  A component that generates embeddings for the article summaries using a pre-trained language model.

**Data Flow:**

1.  The user interacts with the Streamlit UI to enter a search query or trigger article extraction.
2.  The Gmail Agent fetches "Medium Daily Digest" emails from the Gmail API.
3. The Article Extractor parses the email content and extracts article titles, links, and content.
4. The Article Summarizer generates concise summaries of the extracted articles.
5. The Embedding Generator creates vector embeddings of the article summaries.
6. The article summaries, embeddings, and metadata are stored in the vector database.
7. When the user enters a search query, the Search Agent creates a vector embedding of the query.
8. The Search Agent queries the vector database for similar embeddings.
9. The Search Agent returns the relevant articles to the Streamlit UI.
10. The Streamlit UI displays the search results in the AG Grid.

**Technology Stack:**

*   Python
*   Streamlit
*   AG-UI (ag-grid-streamlit)
*   Langchain
*   LlamaIndex
*   sentence-transformers (for embeddings)
*   google-api-python-client (for Gmail API)
*   psycopg2 (for Postgres connection)
*   pgvector (Postgres extension for vector storage)
*   Docker

## Agentic AI Implementation

This system exemplifies **true agentic AI principles** through:

- **Autonomous Decision Making**: Agents independently decide batch sizes, processing priorities, and optimization strategies
- **Event-Driven Coordination**: Redis pub/sub enables loose coupling while maintaining system coherence
- **Self-Healing Architecture**: Circuit breakers, graceful degradation, and automatic recovery
- **Adaptive Performance**: Continuous optimization based on observed metrics and system state
- **Specialized Agent Capabilities**: Each agent has domain expertise and operates autonomously

> **📖 For comprehensive details on agentic AI patterns and implementation:**
> **See [Agentic AI Reference Guide](AGENTIC_AI_REFERENCE.md)**

**Next Steps:**
*   Proceed to `03-setup-and-configuration.md` to prepare development environment.
*   Review `AGENTIC_AI_REFERENCE.md` for detailed agentic patterns and best practices.
