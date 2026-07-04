\# AI Company Research Assistant



\## Overview



AI Company Research Assistant is a Streamlit-based web application that automatically researches a company using its name or website URL.



The application collects publicly available information, crawls the company's website, analyzes the content using an AI model through OpenRouter, and generates a structured business report with a downloadable PDF.



\---



\## Features



\- Search company by name

\- Automatic website detection

\- Website crawling

\- Company profile extraction

\- AI-generated business analysis

\- Competitor identification

\- Products and services extraction

\- Business summary

\- PDF report generation

\- Streamlit web interface



\---



\## Technologies Used



\- Python

\- Streamlit

\- BeautifulSoup

\- Requests

\- OpenRouter API

\- Serper API

\- ReportLab



\---



\## Installation



Clone the repository



```bash

git clone <repository\_url>

cd AI\_Company\_Research

```



Install dependencies



```bash

pip install -r requirements.txt

```



Create a `.env` file



```

OPENROUTER\_API\_KEY=your\_openrouter\_key

SERPER\_API\_KEY=your\_serper\_key

```



Run the application



```bash

streamlit run app.py

```



\---



\## Project Structure



```

AI\_Company\_Research/

│

├── app.py

├── ai.py

├── crawler.py

├── search.py

├── pdf\_generator.py

├── utils.py

├── requirements.txt

├── README.md

└── .env

```



\---



\## Workflow



1\. User enters company name or website.

2\. Company information is searched using Serper API.

3\. Website pages are crawled.

4\. Website content is cleaned.

5\. AI generates business insights.

6\. Report is displayed.

7\. PDF report is generated.



\---



\## APIs Used



\- OpenRouter API

\- Serper Search API



\---



\## Output



The application generates:



\- Company Summary

\- Products

\- Services

\- Pain Points

\- Competitors

\- Business Analysis

\- Downloadable PDF Report



\---



\## Author



Developed as an AI Company Research project using Python and Streamlit.

