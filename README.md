# AskScholar

AskScholar is a chatbot application designed to help students discover scholarships and provide answers to queries related to scholarships. The app leverages advanced AI technologies to provide relevant, up-to-date information in a conversational format.

## Key Features
- **Scholarship Discovery:** Easily search for scholarships based on your profile and requirements.
- **AI-Powered Query Handling:** AskScholar can handle questions related to eligibility, deadlines, application processes, and more.
- **Embeddings and Text Generation:** Utilizes Gemini for generating embeddings and text responses.
- **Retrieval-Augmented Generation (RAG):** Powered by TiDB Serverless and TiDB Vector Search for efficient retrieval and generation.

## Tech Stack
- **Operating System:** Ubuntu
- **Frontend:** Streamlit
- **Backend:**
  - **LangChain:** For handling language model operations and integrating components.
  - **Gemini:** Used for embeddings and text generation.
  - **TiDB Serverless & TiDB Vector Search:** For implementing retrieval-augmented generation.

## How It Works
1. **User Query Input:** Users can input their queries or search for scholarships using the chatbot interface.
2. **Query Processing:** The query is processed using LangChain and embeddings are generated using Gemini.
3. **RAG Implementation:** The system uses TiDB Vector Search to retrieve relevant documents, which are combined with text generation to deliver accurate and contextual responses.
4. **Response Display:** The final response is presented through the Streamlit interface.

## Setup Instructions
1. **Clone the Repository:**
   ```bash
   git clone https://github.com/yourusername/askscholar.git
   cd askscholar
   ```

2. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set Up TiDB Serverless:**
   - Sign up for a TiDB Cloud account and create a serverless cluster.
   - Follow the [TiDB documentation](https://docs.pingcap.com/tidbcloud) to configure the cluster and obtain connection details.
   - Update the connection details in the app’s configuration file.

4. **Run the Application:**
   ```bash
   streamlit run app.py
   ```

5. **Access the Chatbot:**
   - Open the URL displayed in your terminal (usually http://localhost:8501/) to interact with AskScholar.

## Future Improvements
- Add support for more regions and scholarship sources.
- Integrate additional language models for improved accuracy.
- Enhance the UI for better user experience.

## Contributing
Feel free to open issues or submit pull requests if you’d like to contribute!

## License
This project is licensed under the MIT License.
