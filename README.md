# Generic Web Scraping and AI Indexing Project

## Project Structure

```
Scrapy/
│
├── hasolidit/
│   ├── scrapper/
│   │   └── hasolidit-scrapper/
│   │       ├── spiders/
│   │       ├── items.py
│   │       ├── middlewares.py
│   │       ├── pipelines.py
│   │       └── settings.py
│   │
│   ├── embedding/
│   │   ├── embedding-v3.py
│   │   ├── chat.py
│   │   └── chat_simple.py
│   │
│   └── agent/
│
├── requirements.txt
└── README.md
```

## Project Workflow

1. **Web Scraping**
   - Use Scrapy to crawl and collect data
   
2. **Embedding Generation**
   - Create vector embeddings from scraped JSON

3. **Chatbot Interaction**
   - Use generated embeddings to power conversational AI

## Prerequisites

- Python 3.11
- Required dependencies (see `requirements.txt`)

## Installation

```bash
# Clone the repository
git clone https://github.com/noam412/Scrapy.git
cd Scrapy

# Install dependencies
pip install -r requirements.txt
```

## Usage

### 1. Web Scraping

Navigate to the scrapper directory and run Scrapy crawler:
```bash
cd hasolidit/scrapper
scrapy crawl [spider_name]
```

### 2. Generate Embeddings

Run the embedding script:
```bash
cd ../embedding
python3.11 embedding-v3.py
```
**Note:** Ensure the Scrapy crawler has finished and generated the JSON file before running embeddings.

### 3. Start Chatbot

Choose between verbose and simple chat modes:
```bash
# Verbose mode with debug information
python3.11 chat.py

# Simple mode
python3.11 chat_simple.py
```

## Important Files

- `scrapper/hasolidit_articles.json`: Output from web scraping
- `embedding/faiss_index/`: Generated vector index

## Troubleshooting

- Ensure you're using Python 3.11
- Verify that scraping is complete before embedding
- Check that required dependencies are installed

## Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Open a Pull Request

## License

Distributed under the MIT License. See LICENSE for more information.
