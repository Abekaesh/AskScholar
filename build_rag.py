import requests
from bs4 import BeautifulSoup
import re
from dotenv import load_dotenv
import google.generativeai as genai
from langchain_community.vectorstores import TiDBVectorStore
from langchain_core.embeddings.embeddings import Embeddings
import os

load_dotenv()

# Function to remove excessive newlines from text
def remove_excessive_newlines(text):
    cleaned_text = re.sub(r'\n+', '\n\n', text)
    return cleaned_text.strip()

# Function to scrape scholarship details from a webpage
def scrape_scholarship_details(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')

    # Example scraping logic; adjust as needed
    scholarship_name = soup.find('h1', class_='is-title post-title').text.strip()
    details = soup.find('div', class_='post-content cf entry-content content-spacious').text.strip()

    # Clean up the text to remove excessive newlines
    scholarship = scholarship_name + "\n\n" + details
    scholarship = remove_excessive_newlines(scholarship)

    return {'scholarship': scholarship, 'source_link': url}

# Configure Google Generative AI for embedding generation
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
embedding_model = 'models/embedding-001'  # Replace with the actual model name

def text_to_embedding(text):
    embeddings = genai.embed_content(model=embedding_model, content=[text], task_type="retrieval_document")
    return embeddings['embedding'][0] if 'embedding' in embeddings else None

class GeminiEmbeddings(Embeddings):
    def __init__(self, model_name='models/embedding-001'):
        self.model_name = model_name

    def embed_documents(self, texts):
        return [self.text_to_embedding(text) for text in texts]

    def embed_query(self, query):
        return self.text_to_embedding(query)

    def text_to_embedding(self, text):
        embeddings = genai.embed_content(
            model=self.model_name,
            content=[text],
            task_type="retrieval_document"
        )
        return embeddings['embedding'][0] if 'embedding' in embeddings else None
    


# Main script execution
def main():
    # Connect to TiDB
    #connection = connect_to_tidb()

    tidb_connection_string = os.getenv("TIDB_CONNECTION_STRING")
    TABLE_NAME = "scholarship_embeddings"
    # Example URLs for scraping (replace with actual URLs)
    urls = [
        'https://opportunitydesk.org/2024/08/20/paset-rsif-phd-scholarships-2024/',
        'https://opportunitydesk.org/2024/08/20/leigh-day-llb-scholarship-2024-2025/',
        'https://opportunitydesk.org/2024/08/15/us-south-pacific-scholarship-program-2025/',
        'https://opportunitydesk.org/2024/08/19/wigwe-university-scholarship-2024/',
        'https://opportunitydesk.org/2024/08/20/joel-joffe-llb-scholarships-2024-2025/',
        'https://opportunitydesk.org/2024/07/19/mastercard-foundation-scholars-program-at-uwc-2025/',
        'https://opportunitydesk.org/2024/08/10/nl-scholarship-2024-2025/',
        'https://opportunitydesk.org/2024/06/05/lila-fahlman-scholarship-2024/',
        'https://opportunitydesk.org/2024/06/03/open-futures-scholarship-for-black-students-2024-2025/',
        'https://opportunitydesk.org/2024/05/31/stanbic-ibtc-university-scholarship-2024/',
        'https://opportunitydesk.org/2024/05/07/vinuniversity-scholarships-for-international-students-to-study-in-vietnam-2024/',
        'https://opportunitydesk.org/2024/01/16/turkiye-scholarships-2024/',
        'https://opportunitydesk.org/2024/01/12/edinburgh-global-undergraduate-mathematics-scholarships-2024/',
        'https://opportunitydesk.org/2023/12/28/taiwanicdf-international-higher-education-scholarship-program-2024/',
        'https://opportunitydesk.org/2023/12/28/taiwanicdf-international-higher-education-scholarship-program-2024/',
        'https://opportunitydesk.org/2023/11/09/charity-kpabep-scholarship-programme-2023-2024/',
        'https://opportunitydesk.org/2023/10/10/10-fully-funded-scholarship-opportunities-to-study-abroad-still-open-october-10-2023/',
        'https://opportunitydesk.org/2023/10/02/humboldt-diversity-scholarships-for-study-abroad-2023-2024/',
        'https://opportunitydesk.org/2023/09/25/10-undergraduate-scholarships-to-study-in-new-zealand-2/',
        'https://opportunitydesk.org/2023/09/25/rig-future-academy-scholarship-2023-2024/'
    ]

    # Scrape, generate embeddings, and insert scholarship details into TiDB
    sources = []
    documents = []
    for url in urls:
        data = scrape_scholarship_details(url)
        documents.append(str(data['scholarship']))
        sources.append({'source': str(data['source_link'])})

    gemini_embed = GeminiEmbeddings()
    vector_store = TiDBVectorStore(
        embedding_function=gemini_embed,
        table_name=TABLE_NAME,
        connection_string=tidb_connection_string,
        distance_strategy="cosine",  # default, another option is "l2"
        drop_existing_table=True
    )
    vargs = {
        "texts":documents, 
        "metadatas":sources,
        "embedding":gemini_embed,
        "table_name":TABLE_NAME,
        "connection_string":tidb_connection_string,
        "distance_strategy":"cosine",  # default, another option is "l2"
        "drop_existing_table":True
    }
    db = vector_store.from_texts(**vargs)
    
    # Query the most similar documents based on a question
    query = "Tell about Leigh Day LLB Scholarship"
    docs_with_score = db.similarity_search_with_score(query, k=3)

    for doc, score in docs_with_score:
        print("-" * 80)
        print("Score: ", score)
        print(doc.page_content)
        print("-" * 80)

if __name__ == '__main__':
    main()
