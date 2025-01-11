import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from googlesearch import search
import openai
from bs4 import BeautifulSoup
import requests
import time
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re

# Streamlit app
st.title("Customer Churn Tracker")

# User Inputs
company_name = st.text_input("Enter the Company Name", "Salesforce")
customer_name = st.text_input("Enter the Customer Name", "Spotify")
timeframe = st.text_input("Enter Timeframe (e.g., 'last 6 months')", "last 6 months")
openai_api_key = st.text_input("Enter your OpenAI API Key", type="password")
num_results = st.slider("Number of Search Results", min_value=1, max_value=20, value=5)
delay_between_requests = st.slider("Delay Between Requests (seconds)", min_value=1, max_value=10, value=1)

# Initialize OpenAI API Key
if openai_api_key:
    openai.api_key = openai_api_key

if st.button("Search Market Insights"):
    if not company_name or not customer_name:
        st.error("Please enter both Company Name and Customer Name.")
    else:
        st.write(f"Searching recent market insights between {customer_name} and {company_name}...")

        # Step 1: Perform Web Search
        search_query = f"{customer_name} {company_name} churn OR contract OR partnership OR deal {timeframe}"
        
        def perform_search(query, num_results, delay):
            results = []
            try:
                for i, result in enumerate(search(query, pause=delay)):
                    if i >= num_results:
                        break
                    results.append(result)
                return results
            except Exception as e:
                st.error(f"Error during search: {e}")
                return []


        urls = perform_search(search_query, num_results, delay_between_requests)
        st.write(f"Found {len(urls)} relevant results.")

        # Step 2: Scrape and Preprocess Content
        def preprocess_content(text):
            stop_words = set(stopwords.words("english"))
            words = word_tokenize(text)
            filtered_words = [w for w in words if w.lower() not in stop_words]
            
            keywords = ["churn", "contract termination", "partnership", "deal"]
            highlighted_text = re.sub(
                r'\b(' + '|'.join(keywords) + r')\b',
                lambda x: f"**{x.group(0).upper()}**",
                ' '.join(filtered_words),
                flags=re.IGNORECASE
            )
            return highlighted_text
        
        content_list = []
        for url in urls:
            try:
                response = requests.get(url)
                soup = BeautifulSoup(response.text, "html.parser")
                content = soup.get_text()
                preprocessed_content = preprocess_content(content[:2000])  # Limit to 2000 characters
                content_list.append(preprocessed_content)
            except Exception as e:
                st.error(f"Error scraping {url}: {e}")

        # Step 3: Analyze Content with LLM
        if content_list:
            llm = ChatOpenAI(temperature=0.3, openai_api_key=openai_api_key)
            prompt = PromptTemplate(input_variables=["content", "company", "customer"],
                                     template="""
                You are a market analyst. Analyze the content below to identify recent market activities, 
                including partnerships, contracts, or potential risks of churn, between {customer} and {company}. 
                Summarize the findings and indicate if there's any sign of churn or market issues.

                Content: {content}
            """)
            summaries = []
            for content in content_list:
                try:
                    summary = llm.generate(
                        prompt.format(content=content, company=company_name, customer=customer_name)
                    )
                    summaries.append(summary)
                except Exception as e:
                    st.error(f"Error generating insights: {e}")

            # Step 4: Display Results
            st.subheader("Insights:")
            for i, summary in enumerate(summaries):
                st.write(f"**Source {i + 1}:**")
                st.write(summary)

            st.subheader("Sources:")
            for url in urls:
                st.write(url)
        else:
            st.error("No relevant content found.")
