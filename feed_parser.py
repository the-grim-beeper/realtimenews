import feedparser
import pandas as pd
import requests
import json
import time
import logging
import hashlib
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from bs4 import BeautifulSoup
import pytz
import os
import re
from urllib.parse import urlparse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class NewsFeedParser:
    """Handles fetching and parsing RSS feeds from Google News and other sources."""
    
    def __init__(self, feeds: List[str] = None):
        """
        Initialize the feed parser with optional feed URLs.
        
        Args:
            feeds: List of RSS feed URLs to monitor
        """
        self.feeds = feeds or []
        self.articles = []
        self.user_agent = "Mozilla/5.0 (compatible; NewsMonitor/1.0; +https://example.com)"
        
        # Create data directory if it doesn't exist
        os.makedirs("data/raw", exist_ok=True)
        
        # Load any cached articles
        self._load_cached_articles()
    
    def set_feeds(self, feeds: List[str]) -> None:
        """
        Update the list of feeds to monitor.
        
        Args:
            feeds: List of RSS feed URLs
        """
        self.feeds = feeds
        logger.info(f"Updated feed list with {len(feeds)} sources")
    
    def fetch_all(self) -> None:
        """Fetch all configured feeds and process their articles."""
        start_time = time.time()
        new_articles = 0
        
        for feed_url in self.feeds:
            try:
                articles = self._fetch_feed(feed_url)
                new_articles += len(articles)
            except Exception as e:
                logger.error(f"Error fetching feed {feed_url}: {str(e)}")
        
        logger.info(f"Fetched {new_articles} new articles in {time.time() - start_time:.2f}s")
        
        # Cache articles to disk
        self._cache_articles()
    
    def _fetch_feed(self, feed_url: str) -> List[Dict[str, Any]]:
        """
        Fetch and parse a single RSS feed.
        
        Args:
            feed_url: URL of the RSS feed
            
        Returns:
            List of article dictionaries
        """
        logger.info(f"Fetching feed: {feed_url}")
        
        # Use feedparser to parse the RSS feed
        feed = feedparser.parse(feed_url, agent=self.user_agent)
        
        if feed.bozo:
            logger.warning(f"Feed parse error: {feed.bozo_exception}")
        
        # Process feed entries into standardized article format
        new_articles = []
        for entry in feed.entries:
            # Extract source from URL or feed title
            source = self._extract_source(entry)
            
            # Create unique ID based on URL
            article_id = hashlib.md5(entry.link.encode()).hexdigest()
            
            # Parse publication date
            published = self._parse_date(entry)
            
            # Extract article text if possible
            content = self._extract_content(entry)
            
            # Create article dict
            article = {
                "id": article_id,
                "title": entry.title,
                "link": entry.link,
                "published": published,
                "source": source,
                "summary": entry.get("summary", ""),
                "content": content,
                "categories": entry.get("tags", []),
                "feed_url": feed_url
            }
            
            # Only add if we don't already have this article
            if not any(a["id"] == article_id for a in self.articles):
                self.articles.append(article)
                new_articles.append(article)
        
        logger.info(f"Found {len(new_articles)} new articles in feed")
        return new_articles
    
    def _extract_source(self, entry: Dict[str, Any]) -> str:
        """Extract the news source name from an entry."""
        # Try to get from source field
        if hasattr(entry, "source"):
            return entry.source.title
        
        # Try to extract from URL
        domain = urlparse(entry.link).netloc
        
        # Remove subdomains and remove TLD
        domain_parts = domain.split(".")
        if len(domain_parts) > 2:
            # Handle cases like news.bbc.co.uk
            if domain_parts[-2] == "co" or domain_parts[-2] == "com":
                return domain_parts[-3].capitalize()
            return domain_parts[-2].capitalize()
        
        return domain_parts[0].capitalize()
    
    def _parse_date(self, entry: Dict[str, Any]) -> datetime:
        """Parse the publication date from an entry."""
        # Try different date fields
        for date_field in ["published", "pubDate", "updated"]:
            if hasattr(entry, date_field):
                try:
                    # Parse the date string to a datetime object
                    dt = datetime(*entry[date_field][:6])
                    return dt.replace(tzinfo=pytz.UTC)
                except Exception:
                    # Try feedparser's parsed date
                    if hasattr(entry, f"{date_field}_parsed"):
                        struct_time = entry[f"{date_field}_parsed"]
                        return datetime.fromtimestamp(time.mktime(struct_time), pytz.UTC)
        
        # Default to current time if we can't parse the date
        return datetime.now(pytz.UTC)
    
    def _extract_content(self, entry: Dict[str, Any]) -> str:
        """
        Extract the content from an entry, trying different fields and cleaning HTML.
        
        Args:
            entry: Feed entry dictionary
            
        Returns:
            Cleaned text content
        """
        content = ""
        
        # Try different content fields
        for content_field in ["content", "description", "summary"]:
            if hasattr(entry, content_field):
                if content_field == "content" and isinstance(entry.content, list):
                    for content_item in entry.content:
                        content += content_item.value
                else:
                    content = getattr(entry, content_field)
                break
        
        # Clean HTML if present
        if content and ("<" in content and ">" in content):
            try:
                soup = BeautifulSoup(content, "html.parser")
                content = soup.get_text(separator=" ")
            except Exception as e:
                logger.warning(f"Error cleaning HTML: {str(e)}")
        
        # Clean up whitespace
        content = re.sub(r'\s+', ' ', content).strip()
        
        return content
    
    def get_recent_articles(self, time_delta: timedelta) -> List[Dict[str, Any]]:
        """
        Get articles published within a specific time window.
        
        Args:
            time_delta: Time window to look back
            
        Returns:
            List of recent articles
        """
        cutoff_time = datetime.now(pytz.UTC) - time_delta
        
        # Filter articles by published time
        recent_articles = [
            article for article in self.articles
            if article["published"] >= cutoff_time
        ]
        
        # Sort by publication date, newest first
        recent_articles.sort(key=lambda x: x["published"], reverse=True)
        
        return recent_articles
    
    def _cache_articles(self) -> None:
        """Save articles to disk cache."""
        # Convert datetime objects to strings for JSON serialization
        articles_for_cache = []
        for article in self.articles:
            article_copy = article.copy()
            article_copy["published"] = article_copy["published"].isoformat()
            articles_for_cache.append(article_copy)
        
        try:
            with open("data/raw/articles_cache.json", "w") as f:
                json.dump(articles_for_cache, f)
        except Exception as e:
            logger.error(f"Error caching articles: {str(e)}")
    
    def _load_cached_articles(self) -> None:
        """Load articles from disk cache."""
        try:
            if os.path.exists("data/raw/articles_cache.json"):
                with open("data/raw/articles_cache.json", "r") as f:
                    cached_articles = json.load(f)
                
                # Convert string dates back to datetime objects
                for article in cached_articles:
                    article["published"] = datetime.fromisoformat(article["published"])
                
                self.articles = cached_articles
                logger.info(f"Loaded {len(self.articles)} articles from cache")
        except Exception as e:
            logger.error(f"Error loading cached articles: {str(e)}")
    
    def get_all_sources(self) -> List[str]:
        """Get a list of all news sources in the collected articles."""
        return sorted(list(set(article["source"] for article in self.articles)))
    
    def get_articles_by_source(self, source: str) -> List[Dict[str, Any]]:
        """
        Get all articles from a specific source.
        
        Args:
            source: Name of the news source
            
        Returns:
            List of articles from that source
        """
        return [article for article in self.articles if article["source"] == source]
