```markdown
# Setup and Configuration

**1. Install Python and Pip:**

*   Ensure you have Python 3.7+ installed.
*   Verify that `pip` (Python package installer) is installed.

**2. Create a Virtual Environment:**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

**3. Install Required Packages:**

```bash
pip install -r requirements.txt
# add these required packages in requirements.txt
# streamlit ag-grid-streamlit langchain llama-index psycopg2-binary pgvector transformers sentence-transformers google-api-python-client google-auth-httplib2 google-auth-oauthlib langchain-gmail
```

**4. Set up Postgres and pgvector in Docker:**
*   Create a docker-compose.yml file:
```bash
version: "3.9"
services:
  db:
    image: ankane/pgvector:latest
    ports:
      - "5432:5432"
    environment:
      POSTGRES_PASSWORD: postgres
      POSTGRES_USER: postgres
      POSTGRES_DB: pdf_search
    volumes:
      - db_data:/var/lib/postgresql/data  # Persist data across restarts

volumes:
  db_data: # Define volume for persistent storage
```
*   Start the database:
```bash
docker-compose up -d
```
**5. Configure Gmail API access:**
*   Enable the Gmail API in the Google Cloud Console (https://console.cloud.google.com/).
*   Create credentials (Service Account).
*   Download the credentials.json file.
*   Store the credentials.json file securely and note its path. You'll need this path in the gmail_agent.py file.

**6. Enable pgVector extension:**
*   If not enabled, enable the pgVector extension.

**Next Steps:**
* Proceed to `04-gmail-agent-implementation.md` for implemenation details.

