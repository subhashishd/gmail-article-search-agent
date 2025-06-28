# Project Overview: Gmail Article Search

**Goal:** Build a local application that automatically extracts, summarizes, and indexes articles from "Medium Daily Digest" emails in your Gmail account. Users can then search these articles efficiently.

**Key Features:**

*   **Automated Article Extraction:**  Fetch "Medium Daily Digest" emails, parse them, and extract article titles, links, and content.
*   **Summarization:**  Generate concise summaries of the extracted articles.
*   **Vector Database Indexing:**  Create and maintain a vector database of article summaries for efficient semantic search.
*   **Search Functionality:**  Allow users to enter a search query and retrieve relevant articles from the index.
*   **Persistent Memory:** The agent should remember its last updated date, and use that data to find new articles every time.
*   **User Interface:**  Provide a user-friendly interface using Streamlit and AG-UI for displaying search results.

**Target User:**  Someone overwhelmed by the volume of information in their email digests and who wants a quick and efficient way to search for relevant articles.

**Success Metrics:**

*   Accuracy of article extraction and summarization.
*   Relevance of search results.
*   Speed of search queries.
*   Ease of use of the user interface.

**Next Steps:**
* Proceed to `02-architecture.md` for detailed system architechture overview.