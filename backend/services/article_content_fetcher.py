# ArticleContentFetcher: Responsible for fetching full article content from Medium URLs
# Using the requests library for simplicity. In production, you might want to handle retries, rate-limiting, etc.

import requests
from bs4 import BeautifulSoup
from typing import Optional, Dict

class ArticleContentFetcher:
    def __init__(self, session_cookies: Dict[str, str]):
        self.session_cookies = session_cookies

    def fetch_article_content(self, url: str) -> Optional[str]:
        """
        Fetch the full article content from Medium.

        Args:
            url: The Medium article URL.

        Returns:
            The full text of the article or None if fetching fails.
        """
        try:
            response = requests.get(url, cookies=self.session_cookies)
            if response.status_code == 200:
                # Parse the article's content with BeautifulSoup
                soup = BeautifulSoup(response.content, 'html.parser')

                # Scrape main article text by looking for styles and tags typical in Medium
                paragraphs = soup.find_all('p')
                full_text = ' '.join(p.get_text() for p in paragraphs)

                return full_text
            else:
                print(f"Failed to fetch article content. HTTP Status: {response.status_code}")
                return None
        except requests.RequestException as e:
            print(f"Exception occurred while fetching article content: {e}")
            return None

# Example usage
# fetcher = ArticleContentFetcher(session_cookies={"sid": "your-medium-sid-here"})
# full_article_content = fetcher.fetch_article_content('https://medium.com/some-article')

# Async function for use in agents
async def fetch_article_content(url: str) -> str:
    """
    Async wrapper for fetching article content.
    
    Args:
        url: The Medium article URL
        
    Returns:
        The full text of the article or fallback content
    """
    try:
        import asyncio
        import aiohttp
        
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=30)) as response:
                if response.status == 200:
                    html_content = await response.text()
                    
                    # Parse the article's content with BeautifulSoup
                    soup = BeautifulSoup(html_content, 'html.parser')
                    
                    # Extract main article content from Medium's structure
                    # Medium typically uses article tags or specific classes
                    article_content = soup.find('article')
                    if article_content:
                        paragraphs = article_content.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6'])
                    else:
                        paragraphs = soup.find_all('p')
                    
                    full_text = '\n'.join(p.get_text().strip() for p in paragraphs if p.get_text().strip())
                    
                    if full_text:
                        return full_text
                    else:
                        return f"Unable to extract content from {url}"
                else:
                    return f"Failed to fetch article (HTTP {response.status}): {url}"
                    
    except Exception as e:
        return f"Error fetching article content: {str(e)}"

# For backwards compatibility
fetcher_instance = ArticleContentFetcher(session_cookies={})

def fetch_article_content_sync(url: str) -> Optional[str]:
    """Synchronous version for backwards compatibility"""
    return fetcher_instance.fetch_article_content(url)
