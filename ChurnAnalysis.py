import streamlit as st
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from googlesearch import search
import openai
from bs4 import BeautifulSoup
import requests
import time
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re
import os
import nltk

# Add custom NLTK data path
nltk.data.path = ["/home/appuser/nltk_data"]

# Debug: NLTK Data Path Initialization
st.write("Initializing NLTK Data Path:", nltk.data.path)

# Force re-download required NLTK data
try:
    nltk.download("stopwords", download_dir="/home/appuser/nltk_data")
    nltk.download("punkt", download_dir="/home/appuser/nltk_data", force=True)
    st.write("Successfully downloaded or verified NLTK stopwords and punkt.")
except Exception as e:
    st.error(f"Error downloading NLTK data: {e}")

# Debug: Verify NLTK Data Path
punkt_path = os.path.join("/home/appuser/nltk_data", "tokenizers", "punkt")
if os.path.exists(punkt_path):
    st.write("Punkt Directory Exists. Contents:", os.listdir(punkt_path))
else:
    st.error("Punkt Directory Missing.")

# Streamlit app title
st.title("Customer Churn Tracker")

# User Inputs
company_name = st.text_input("Enter the Company Name", "Salesforce")
customer_name = st.text_input("Enter the Customer Name", "Spotify")
timeframe = st.text_input("Enter Timeframe (e.g., 'last 6 months')", "last 6 months")
openai_api_key = st.text_input("Enter your OpenAI API Key", type="password")
num_results = st.slider("Number of Search Results", min_value=1, max_value=20, value=5)
delay_between_requests = st.slider("Delay Between Requests (seconds)", min_value=1, max_value=10, value=1)

# Debug: Log user inputs
st.write("User Inputs:")
st.write("Company Name:", company_name)
st.write("Customer Name:", customer_name)
st.write("Timeframe:", timeframe)
st.write("Number of Results:", num_results)
st.write("Delay Between Requests:", delay_between_requests)

# Initialize OpenAI API Key
if openai_api_key:
    openai.api_key = openai_api_key
    st.write("OpenAI API Key initialized.")
else:
    st.warning("No OpenAI API Key provided.")

if st.button("Search Market Insights"):
    if not company_name or not customer_name:
        st.error("Please enter both Company Name and Customer Name.")
    else:
        st.write(f"Searching recent market insights between {customer_name} and {company_name}...")

        # Step 1: Perform Web Search
        search_query = f"{customer_name} {company_name} switch OR migrate OR churn OR move OR contract OR partnership OR deal {timeframe} OR 'moved from' {timeframe}"
        st.write("Search Query:", search_query)

        def perform_search(query, num_results, delay):
            results = []
            try:
                for i, result in enumerate(search(query)):
                    st.write(f"Search Result {i + 1}: {result}")
                    if i >= num_results:
                        break
                    results.append(result)
                    time.sleep(delay)
                return results
            except Exception as e:
                st.error(f"Error during search: {e}")
                return []

        urls = perform_search(search_query, num_results, delay_between_requests)
        st.write(f"Found {len(urls)} relevant results.")

        # Step 2: Scrape and Preprocess Content
        def clean_text(text):
            text = re.sub(r'[^\x00-\x7F]+', ' ', text)
            text = re.sub(r'\s+', ' ', text).strip()
            return text

        def preprocess_content(text):
            try:
                words = word_tokenize(text)
                stop_words = set(stopwords.words("english"))
                filtered_words = [w for w in words if w.lower() not in stop_words]
                keywords = ["churn", "contract termination", "partnership", "deal"]
                highlighted_text = re.sub(
                    r'\b(' + '|'.join(keywords) + r')\b',
                    lambda x: f"**{x.group(0).upper()}**",
                    ' '.join(filtered_words),
                    flags=re.IGNORECASE
                )
                return highlighted_text
            except Exception as e:
                st.warning(f"Tokenization failed. Error: {e}")
                return clean_text(text)

        content_list = []
        for url in urls:
            try:
                response = requests.get(url)
                soup = BeautifulSoup(response.text, "html.parser")
                content = clean_text(soup.get_text())
                st.write(f"Scraping content from URL: {url}")
                preprocessed_content = preprocess_content(content[:5000])  # Limit to 5000 characters
                content_list.append(preprocessed_content)
            except Exception as e:
                st.error(f"Error scraping {url}: {e}")

        # Step 3: Analyze Content with LLM
        if content_list:
            llm = ChatOpenAI(temperature=0.3, openai_api_key=openai_api_key)
            prompt = PromptTemplate(
                input_variables=["content", "company", "customer"],
                template="""You are a market analyst. Analyze the content below to identify recent market activities between {customer} and {company}. Focus on the following:

                        1. Any clear statements or indications of {customer} switching from {company} to another service provider.
                        2. Specific mentions of {customer} moving away from {company}'s services or adopting a competitor's product.
                        3. Evidence of {customer} migrating their data or services from {company} to another platform.
                        4. Signs of customer dissatisfaction or potential churn risks related to {company}.
                        5. Any positive outcomes reported after switching from {company} to another service.
                        
                Summarize your findings concisely, highlighting any concrete evidence of switching, migration, or churn. If there's a clear switch or migration mentioned, state it explicitly with any relevant details (e.g., which company they switched to, when it happened, reasons for the switch).
                Content: {content}
                Summary:
                """
            )
            summaries = []
            for i, content in enumerate(content_list):
                try:
                    st.write(f"Processing content {i + 1} with LLM...")
                    chain = LLMChain(llm=llm, prompt=prompt)
                    summary = chain.run(content=content, company=company_name, customer=customer_name)
                    summaries.append(summary)
                except Exception as e:
                    st.error(f"Error generating insights: {e}")

            # Step 4: Display Results
            st.subheader("Insights:")
            for i, summary in enumerate(summaries):
                st.write(f"**Source {i + 1}:**")
                st.write(summary)

            # New Overall Analysis Section
            if summaries:
                st.subheader("Overall Analysis:")
                try:
                    overall_summary = "\n".join(summaries)
                    final_prompt = PromptTemplate(
                        input_variables=["summaries", "company", "customer"],
                        template="""Based on the following summaries, provide a concise overall analysis of the relationship between {customer} and {company}. Highlight any clear evidence of switching, migration, or churn. If multiple sources confirm a switch or migration, state it confidently.

                        Summaries:
                        {summaries}

                    Overall Analysis:
                    """
                    )
                    overall_chain = LLMChain(llm=llm, prompt=final_prompt)
                    overall_analysis = overall_chain.run(summaries=overall_summary, company=company_name, customer=customer_name)
                    st.write(overall_analysis)
                except Exception as e:
                    st.error(f"Error generating overall analysis: {e}")

            st.subheader("Sources:")
            for url in urls:
                st.write(url)
                else:
                    st.error("No relevant content found.")
