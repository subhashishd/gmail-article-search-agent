"""
Content Extractor Service for fetching article content from Medium URLs.
"""

import logging
import aiohttp
import asyncio
from typing import Optional
from bs4 import BeautifulSoup
import json
import os

logger = logging.getLogger("ContentExtractor")

async def fetch_article_content(url: str, timeout: int = 15) -> str:
    """
    Fetch article content from Medium URL with improved error handling.
    
    Args:
        url: Article URL to fetch
        timeout: Request timeout in seconds (increased to 15s)
        
    Returns:
        Extracted article content or descriptive error message
    """
    try:
        # Check for Medium member cookies
        cookies = {}
        try:
            cookie_file = "/app/credentials/medium_cookies.json"
            if os.path.exists(cookie_file):
                with open(cookie_file, 'r') as f:
                    cookie_data = json.load(f)
                    if 'medium_sid' in cookie_data:
                        cookies['sid'] = cookie_data['medium_sid']
                    if 'medium_uid' in cookie_data:
                        cookies['uid'] = cookie_data['medium_uid']
                        
                logger.debug(f"Using Medium cookies for authenticated access")
        except Exception as e:
            logger.debug(f"Could not load Medium cookies: {e}")
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Sec-Fetch-Dest': 'document',
            'Sec-Fetch-Mode': 'navigate',
            'Sec-Fetch-Site': 'none'
        }
        
        # Retry logic for better reliability
        max_retries = 2
        for attempt in range(max_retries + 1):
            try:
                async with aiohttp.ClientSession(
                    timeout=aiohttp.ClientTimeout(total=timeout),
                    cookies=cookies
                ) as session:
                    async with session.get(url, headers=headers) as response:
                        status_code = response.status
                        
                        if status_code == 200:
                            html_content = await response.text()
                            extracted_content = extract_content_from_html(html_content)
                            
                            # Validate content quality
                            if extracted_content and not extracted_content.startswith("Unable to fetch"):
                                word_count = len(extracted_content.split())
                                if word_count >= 50:  # Ensure meaningful content
                                    logger.info(f"Successfully fetched content ({word_count} words) from {url[:100]}...")
                                    return extracted_content
                                else:
                                    logger.warning(f"Content too short ({word_count} words) from {url[:100]}...")
                                    return f"Unable to fetch content: Content too short ({word_count} words)"
                            else:
                                return extracted_content or "Unable to fetch content: No content extracted"
                                
                        elif status_code == 403:
                            return "Unable to fetch content: Access forbidden (member-only content)"
                        elif status_code == 404:
                            return "Unable to fetch content: Article not found (404)"
                        elif status_code == 410:
                            return "Unable to fetch content: Article no longer available (410)"
                        elif status_code == 429:
                            if attempt < max_retries:
                                wait_time = (attempt + 1) * 2  # Exponential backoff
                                logger.warning(f"Rate limited (429), retrying in {wait_time}s (attempt {attempt + 1}/{max_retries + 1})")
                                await asyncio.sleep(wait_time)
                                continue
                            return "Unable to fetch content: Rate limited (too many requests)"
                        elif 500 <= status_code < 600:
                            if attempt < max_retries:
                                wait_time = (attempt + 1) * 1  # Shorter wait for server errors
                                logger.warning(f"Server error ({status_code}), retrying in {wait_time}s (attempt {attempt + 1}/{max_retries + 1})")
                                await asyncio.sleep(wait_time)
                                continue
                            return f"Unable to fetch content: Server error (HTTP {status_code})"
                        else:
                            return f"Unable to fetch content: HTTP {status_code}"
                            
                        break  # Exit retry loop for non-retryable status codes
                        
            except asyncio.TimeoutError:
                if attempt < max_retries:
                    logger.warning(f"Request timeout, retrying (attempt {attempt + 1}/{max_retries + 1})")
                    await asyncio.sleep(1)  # Brief wait before retry
                    continue
                return "Unable to fetch content: Request timed out after retries"
            except aiohttp.ClientError as e:
                if attempt < max_retries:
                    logger.warning(f"Network error, retrying (attempt {attempt + 1}/{max_retries + 1}): {e}")
                    await asyncio.sleep(1)
                    continue
                return f"Unable to fetch content: Network error - {str(e)}"
                
    except Exception as e:
        logger.error(f"Error fetching content from {url}: {e}")
        return f"Unable to fetch content: {str(e)}"

def extract_content_from_html(html_content: str) -> str:
    """
    Extract readable content from HTML.
    
    Args:
        html_content: Raw HTML content
        
    Returns:
        Extracted text content
    """
    try:
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()
        
        # Try to find Medium article content
        content_selectors = [
            'article',
            '[data-testid="storyContent"]',
            '.postArticle-content',
            '.section-content',
            'main',
            '.story-content'
        ]
        
        content_text = ""
        for selector in content_selectors:
            content_elements = soup.select(selector)
            if content_elements:
                for element in content_elements:
                    content_text += element.get_text(separator=' ', strip=True) + "\n"
                break
        
        # Fallback: get all paragraph content
        if not content_text:
            paragraphs = soup.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6'])
            content_text = "\n".join([p.get_text(strip=True) for p in paragraphs if p.get_text(strip=True)])
        
        # Clean up the content
        content_text = content_text.strip()
        
        # Remove excessive whitespace
        import re
        content_text = re.sub(r'\n\s*\n', '\n\n', content_text)
        content_text = re.sub(r' +', ' ', content_text)
        
        if len(content_text) < 100:
            return "Unable to fetch content: Content too short or blocked"
        
        return content_text
        
    except Exception as e:
        logger.error(f"Error extracting content from HTML: {e}")
        return f"Unable to fetch content: HTML parsing error - {str(e)}"
