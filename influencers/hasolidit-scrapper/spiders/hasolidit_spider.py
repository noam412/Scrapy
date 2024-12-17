import scrapy

class HasoliditSpider(scrapy.Spider):
    name = 'hasolidit_spider'
    allowed_domains = ['hasolidit.com']
    start_urls = ['https://www.hasolidit.com/%d7%90%d7%a8%d7%9b%d7%99%d7%95%d7%9f']

    custom_settings = {
        #'ROBOTSTXT_OBEY': True,
        'DOWNLOAD_DELAY': 1,  # Respect the website's servers
        # 'USER_AGENT': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'DEPTH_LIMIT': 3,
        # Optional: Configure output
        'FEED_FORMAT': 'json',
        'FEED_URI': 'hasolidit_articles.json'
    }

    # def start_requests(self):
    #     
    #     for url in urls:
    #         yield scrapy.Request(url=url, callback=self.parse)



    def parse(self, response):
        # Find all article links on the page
        article_links = response.css('div.entry-content a::attr(href)').getall()
        
        # Follow each article link
        for link in article_links:
            # Ensure the link is within the same domain
            if self.allowed_domains[0] in link:
                yield response.follow(link, self.parse_article)
        
        # Handle pagination if exists
        next_page = response.css('a.next.page-numbers::attr(href)').get()
        if next_page:
            yield response.follow(next_page, self.parse)

    def parse_article(self, response):
        # Extract article number (ID)
        article_id = response.css('article::attr(id)').get()
        category = response.css('a[rel="category tag"]::text').get()
        
        # Extract title from entry header
        title = response.css('h1.entry-title.entry-title-cover-empty::text').get()
        
        # Extract body content
        body_content = ' '.join(response.css('div.entry-content.clearfix p::text').getall())
        
        # Prepare the scraped data
        yield {
            'url': response.url,
            'article_number': article_id,
            'category': category,
            'title': title.strip() if title else None,
            'body': body_content.strip() if body_content else None
        }

    