import sqlite3
import json
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from collections import Counter, defaultdict
from typing import Dict, List, Tuple, Any, Set, Optional
import logging
import pytz

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TimeSeriesDB:
    """Handles storage and retrieval of time series data for terms and articles."""
    
    def __init__(self, db_path: str = "data/processed/news_monitor.db"):
        """
        Initialize the database connection.
        
        Args:
            db_path: Path to SQLite database file
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        
        self.db_path = db_path
        self.conn = None
        
        # In-memory cache for recent term data
        self.term_cache = {}
        self.article_cache = {}
        
        # Initialize database
        self._initialize_db()
        
        logger.info(f"Database initialized at {db_path}")
    
    def _initialize_db(self) -> None:
        """Initialize the database schema if it doesn't exist."""
        try:
            self.conn = sqlite3.connect(self.db_path)
            cursor = self.conn.cursor()
            
            # Articles table
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS articles (
                id TEXT PRIMARY KEY,
                title TEXT,
                link TEXT,
                source TEXT,
                published TIMESTAMP,
                sentiment_positive REAL,
                sentiment_negative REAL,
                data JSON
            )
            ''')
            
            # Terms table
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS terms (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                term TEXT UNIQUE,
                type TEXT
            )
            ''')
            
            # Article-Term relationship table
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS article_terms (
                article_id TEXT,
                term_id INTEGER,
                count INTEGER,
                timestamp TIMESTAMP,
                FOREIGN KEY (article_id) REFERENCES articles (id),
                FOREIGN KEY (term_id) REFERENCES terms (id),
                PRIMARY KEY (article_id, term_id)
            )
            ''')
            
            # Co-occurrence table
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS term_cooccurrences (
                term1_id INTEGER,
                term2_id INTEGER,
                count INTEGER,
                last_seen TIMESTAMP,
                FOREIGN KEY (term1_id) REFERENCES terms (id),
                FOREIGN KEY (term2_id) REFERENCES terms (id),
                PRIMARY KEY (term1_id, term2_id)
            )
            ''')
            
            # Term context table (for tracking context over time)
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS term_context (
                term_id INTEGER,
                timestamp TIMESTAMP,
                context_data JSON,
                FOREIGN KEY (term_id) REFERENCES terms (id)
            )
            ''')
            
            # Create indices for performance
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_articles_published ON articles (published)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_article_terms_timestamp ON article_terms (timestamp)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_term_context_timestamp ON term_context (timestamp)')
            
            self.conn.commit()
        except Exception as e:
            logger.error(f"Error initializing database: {str(e)}")
            raise
    
    def store_article_terms(self, article: Dict[str, Any], terms_data: Dict[str, Any]) -> None:
        """
        Store article and its extracted terms in the database.
        
        Args:
            article: Article dictionary
            terms_data: Dictionary of extracted terms and metadata
        """
        try:
            if self.conn is None:
                self.conn = sqlite3.connect(self.db_path)
            
            cursor = self.conn.cursor()
            
            # Store article
            cursor.execute(
                '''
                INSERT OR REPLACE INTO articles 
                (id, title, link, source, published, sentiment_positive, sentiment_negative, data)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''',
                (
                    article["id"],
                    article["title"],
                    article["link"],
                    article["source"],
                    article["published"],
                    terms_data["sentiment"]["positive"],
                    terms_data["sentiment"]["negative"],
                    json.dumps(article)
                )
            )
            
            # Extract and count terms
            term_counts = Counter()
            
            # Add entities
            for entity in terms_data["entities"]:
                term_counts[entity["text"].lower()] += 1
            
            # Add keywords
            for keyword in terms_data["keywords"]:
                term_counts[keyword.lower()] += 1
            
            # Add noun chunks
            for chunk in terms_data["noun_chunks"]:
                term_counts[chunk.lower()] += 1
            
            # Store each term
            term_ids = {}
            timestamp = article["published"]
            
            for term, count in term_counts.items():
                # Skip very short terms
                if len(term) < 3:
                    continue
                
                # Get or create term
                cursor.execute(
                    'SELECT id FROM terms WHERE term = ?',
                    (term,)
                )
                result = cursor.fetchone()
                
                if result:
                    term_id = result[0]
                else:
                    # Determine term type
                    term_type = "KEYWORD"
                    for entity in terms_data["entities"]:
                        if entity["text"].lower() == term:
                            term_type = entity["label"]
                            break
                    
                    # Insert new term
                    cursor.execute(
                        'INSERT INTO terms (term, type) VALUES (?, ?)',
                        (term, term_type)
                    )
                    term_id = cursor.lastrowid
                
                term_ids[term] = term_id
                
                # Store article-term relationship
                cursor.execute(
                    '''
                    INSERT OR REPLACE INTO article_terms
                    (article_id, term_id, count, timestamp)
                    VALUES (?, ?, ?, ?)
                    ''',
                    (article["id"], term_id, count, timestamp)
                )
            
            # Store co-occurrences
            term_list = list(term_ids.items())
            for i, (term1, term1_id) in enumerate(term_list):
                for term2, term2_id in term_list[i+1:]:
                    # Ensure consistent ordering (smaller ID first)
                    if term1_id > term2_id:
                        term1_id, term2_id = term2_id, term1_id
                    
                    # Update co-occurrence count
                    cursor.execute(
                        '''
                        INSERT INTO term_cooccurrences (term1_id, term2_id, count, last_seen)
                        VALUES (?, ?, 1, ?)
                        ON CONFLICT(term1_id, term2_id) DO UPDATE SET
                        count = count + 1,
                        last_seen = ?
                        ''',
                        (term1_id, term2_id, timestamp, timestamp)
                    )
            
            # Store context for significant terms
            for term, term_id in term_ids.items():
                # Get context terms that co-occur with this term
                context_terms = {}
                for other_term, other_id in term_ids.items():
                    if other_term != term:
                        context_terms[other_term] = term_counts[other_term]
                
                # Get sentences mentioning this term (from key sentences)
                term_sentences = [
                    sent for sent in terms_data["key_sentences"]
                    if term.lower() in sent.lower()
                ]
                
                # Create context data
                context_data = {
                    "context_terms": context_terms,
                    "sentences": term_sentences[:3],  # Top 3 sentences
                    "sentiment": terms_data["sentiment"],
                    "article_id": article["id"],
                    "source": article["source"]
                }
                
                # Store in term_context table
                cursor.execute(
                    '''
                    INSERT INTO term_context (term_id, timestamp, context_data)
                    VALUES (?, ?, ?)
                    ''',
                    (term_id, timestamp, json.dumps(context_data))
                )
            
            self.conn.commit()
            
            # Update in-memory cache
            self._update_term_cache(term_counts, timestamp)
            
        except Exception as e:
            logger.error(f"Error storing article terms: {str(e)}")
            if self.conn:
                self.conn.rollback()
    
    def _update_term_cache(self, term_counts: Counter, timestamp: datetime) -> None:
        """Update in-memory cache with new term counts."""
        # Add terms to cache
        for term, count in term_counts.items():
            if term in self.term_cache:
                self.term_cache[term]["count"] += count
                self.term_cache[term]["last_seen"] = timestamp
            else:
                self.term_cache[term] = {
                    "count": count,
                    "first_seen": timestamp,
                    "last_seen": timestamp
                }
    
    def get_term_data(self, time_delta: timedelta) -> Dict[str, Any]:
        """
        Get term data for a specific time window.
        
        Args:
            time_delta: Time window to look back
            
        Returns:
            Dictionary with term data
        """
        try:
            if self.conn is None:
                self.conn = sqlite3.connect(self.db_path)
            
            cursor = self.conn.cursor()
            
            # Calculate cutoff time
            cutoff_time = datetime.now(pytz.UTC) - time_delta
            
            # Get terms and counts
            cursor.execute(
                '''
                SELECT t.term, t.type, SUM(at.count) as total
                FROM terms t
                JOIN article_terms at ON t.id = at.term_id
                WHERE at.timestamp >= ?
                GROUP BY t.id
                ORDER BY total DESC
                LIMIT 100
                ''',
                (cutoff_time,)
            )
            
            terms = [(row[0], row[2]) for row in cursor.fetchall()]
            
            # Get co-occurrences
            cursor.execute(
                '''
                SELECT t1.term, t2.term, tc.count
                FROM term_cooccurrences tc
                JOIN terms t1 ON tc.term1_id = t1.id
                JOIN terms t2 ON tc.term2_id = t2.id
                WHERE tc.last_seen >= ?
                ORDER BY tc.count DESC
                LIMIT 500
                ''',
                (cutoff_time,)
            )
            
            co_occurrences = [(row[0], row[1], row[2]) for row in cursor.fetchall()]
            
            return {
                "terms": terms,
                "co_occurrences": co_occurrences,
                "cutoff_time": cutoff_time
            }
        except Exception as e:
            logger.error(f"Error getting term data: {str(e)}")
            return {}
    
    def get_article_count(self) -> int:
        """Get the total number of articles in the database."""
        try:
            if self.conn is None:
                self.conn = sqlite3.connect(self.db_path)
            
            cursor = self.conn.cursor()
            cursor.execute('SELECT COUNT(*) FROM articles')
            return cursor.fetchone()[0]
        except Exception as e:
            logger.error(f"Error getting article count: {str(e)}")
            return 0
    
    def get_unique_term_count(self) -> int:
        """Get the total number of unique terms in the database."""
        try:
            if self.conn is None:
                self.conn = sqlite3.connect(self.db_path)
            
            cursor = self.conn.cursor()
            cursor.execute('SELECT COUNT(*) FROM terms')
            return cursor.fetchone()[0]
        except Exception as e:
            logger.error(f"Error getting term count: {str(e)}")
            return 0
    
    def get_top_terms(self, limit: int, time_delta: timedelta) -> List[str]:
        """
        Get the top terms by frequency in a time window.
        
        Args:
            limit: Maximum number of terms to return
            time_delta: Time window to look back
            
        Returns:
            List of term strings
        """
        try:
            if self.conn is None:
                self.conn = sqlite3.connect(self.db_path)
            
            cursor = self.conn.cursor()
            
            # Calculate cutoff time
            cutoff_time = datetime.now(pytz.UTC) - time_delta
            
            # Get top terms
            cursor.execute(
                '''
                SELECT t.term, SUM(at.count) as total
                FROM terms t
                JOIN article_terms at ON t.id = at.term_id
                WHERE at.timestamp >= ?
                GROUP BY t.id
                ORDER BY total DESC
                LIMIT ?
                ''',
                (cutoff_time, limit)
            )
            
            return [row[0] for row in cursor.fetchall()]
        except Exception as e:
            logger.error(f"Error getting top terms: {str(e)}")
            return []
    
    def get_term_context_over_time(self, term: str, time_delta: timedelta) -> Dict[str, Any]:
        """
        Get context data for a term over time.
        
        Args:
            term: Term to get context for
            time_delta: Time window to look back
            
        Returns:
            Dictionary with timeline of context data
        """
        try:
            if self.conn is None:
                self.conn = sqlite3.connect(self.db_path)
            
            cursor = self.conn.cursor()
            
            # Calculate cutoff time
            cutoff_time = datetime.now(pytz.UTC) - time_delta
            
            # Get term ID
            cursor.execute('SELECT id FROM terms WHERE term = ?', (term,))
            result = cursor.fetchone()
            if not result:
                return {"term": term, "timeline": []}
            
            term_id = result[0]
            
            # Get context data over time
            cursor.execute(
                '''
                SELECT timestamp, context_data
                FROM term_context
                WHERE term_id = ? AND timestamp >= ?
                ORDER BY timestamp
                ''',
                (term_id, cutoff_time)
            )
            
            timeline = []
            for row in cursor.fetchall():
                timestamp = row[0]
                context_data = json.loads(row[1])
                
                timeline_item = {
                    "timestamp": timestamp,
                    "context_terms": context_data["context_terms"],
                    "sentiment": context_data["sentiment"],
                    "source": context_data["source"],
                    "article_id": context_data["article_id"]
                }
                
                timeline.append(timeline_item)
            
            # Aggregate similar timestamps
            aggregated_timeline = []
            current_time = None
            current_data = None
            
            for item in sorted(timeline, key=lambda x: x["timestamp"]):
                timestamp = datetime.fromisoformat(item["timestamp"]) if isinstance(item["timestamp"], str) else item["timestamp"]
                
                if current_time is None or (timestamp - current_time).total_seconds() > 3600:  # 1 hour window
                    # Start new time window
                    if current_data:
                        aggregated_timeline.append(current_data)
                    
                    current_time = timestamp
                    current_data = item.copy()
                else:
                    # Update current window
                    # Merge context terms
                    for term, count in item["context_terms"].items():
                        if term in current_data["context_terms"]:
                            current_data["context_terms"][term] += count
                        else:
                            current_data["context_terms"][term] = count
                    
                    # Average sentiment
                    for key in current_data["sentiment"]:
                        current_data["sentiment"][key] = (current_data["sentiment"][key] + item["sentiment"][key]) / 2
            
            # Add last window if exists
            if current_data:
                aggregated_timeline.append(current_data)
            
            return {
                "term": term,
                "timeline": aggregated_timeline
            }
        except Exception as e:
            logger.error(f"Error getting term context: {str(e)}")
            return {"term": term, "timeline": []}
    
    def get_articles_with_terms(self, terms: List[str], time_delta: timedelta) -> List[Dict[str, Any]]:
        """
        Get articles that mention all specified terms.
        
        Args:
            terms: List of terms to search for
            time_delta: Time window to look back
            
        Returns:
            List of article dictionaries
        """
        try:
            if self.conn is None:
                self.conn = sqlite3.connect(self.db_path)
            
            cursor = self.conn.cursor()
            
            # Calculate cutoff time
            cutoff_time = datetime.now(pytz.UTC) - time_delta
            
            # Get term IDs
            term_ids = []
            for term in terms:
                cursor.execute('SELECT id FROM terms WHERE term = ?', (term,))
                result = cursor.fetchone()
                if result:
                    term_ids.append(result[0])
                else:
                    # If any term doesn't exist, no articles can match all terms
                    return []
            
            # Construct query for articles matching all terms
            placeholders = ', '.join(['?'] * len(term_ids))
            query = f'''
            SELECT a.id, a.title, a.link, a.source, a.published
            FROM articles a
            WHERE a.published >= ?
            AND a.id IN (
                SELECT at.article_id
                FROM article_terms at
                WHERE at.term_id IN ({placeholders})
                GROUP BY at.article_id
                HAVING COUNT(DISTINCT at.term_id) = ?
            )
            ORDER BY a.published DESC
            LIMIT 20
            '''
            
            params = [cutoff_time] + term_ids + [len(term_ids)]
            cursor.execute(query, params)
            
            articles = []
            for row in cursor.fetchall():
                articles.append({
                    "id": row[0],
                    "title": row[1],
                    "link": row[2],
                    "source": row[3],
                    "published": row[4]
                })
            
            return articles
        except Exception as e:
            logger.error(f"Error getting articles with terms: {str(e)}")
            return []
    
    def get_articles_for_shift(self, term: str, timestamp: datetime) -> List[Dict[str, Any]]:
        """
        Get articles around a narrative shift time point.
        
        Args:
            term: Term to get articles for
            timestamp: Time point of the shift
            
        Returns:
            List of article dictionaries
        """
        try:
            if self.conn is None:
                self.conn = sqlite3.connect(self.db_path)
            
            cursor = self.conn.cursor()
            
            # Calculate time window (3 hours around the shift)
            start_time = timestamp - timedelta(hours=1.5)
            end_time = timestamp + timedelta(hours=1.5)
            
            # Get term ID
            cursor.execute('SELECT id FROM terms WHERE term = ?', (term,))
            result = cursor.fetchone()
            if not result:
                return []
            
            term_id = result[0]
            
            # Get articles in time window that mention the term
            cursor.execute(
                '''
                SELECT a.id, a.title, a.link, a.source, a.published
                FROM articles a
                JOIN article_terms at ON a.id = at.article_id
                WHERE at.term_id = ?
                AND a.published BETWEEN ? AND ?
                ORDER BY a.published
                LIMIT 10
                ''',
                (term_id, start_time, end_time)
            )
            
            articles = []
            for row in cursor.fetchall():
                articles.append({
                    "id": row[0],
                    "title": row[1],
                    "link": row[2],
                    "source": row[3],
                    "published": row[4]
                })
            
            return articles
        except Exception as e:
            logger.error(f"Error getting articles for shift: {str(e)}")
            return []
    
    def get_source_coverage(self, term: str, time_delta: timedelta) -> List[Dict[str, Any]]:
        """
        Get coverage data by source for a specific term.
        
        Args:
            term: Term to analyze coverage for
            time_delta: Time window to look back
            
        Returns:
            List of source coverage dictionaries
        """
        try:
            if self.conn is None:
                self.conn = sqlite3.connect(self.db_path)
            
            cursor = self.conn.cursor()
            
            # Calculate cutoff time
            cutoff_time = datetime.now(pytz.UTC) - time_delta
            
            # Get term ID
            cursor.execute('SELECT id FROM terms WHERE term = ?', (term,))
            result = cursor.fetchone()
            if not result:
                return []
            
            term_id = result[0]
            
            # Get coverage data by source
            cursor.execute(
                '''
                SELECT a.source, COUNT(a.id) as article_count,
                       AVG(a.sentiment_positive) as avg_positive,
                       AVG(a.sentiment_negative) as avg_negative
                FROM articles a
                JOIN article_terms at ON a.id = at.article_id
                WHERE at.term_id = ?
                AND a.published >= ?
                GROUP BY a.source
                ORDER BY article_count DESC
                LIMIT 15
                ''',
                (term_id, cutoff_time)
            )
            
            source_data = []
            for row in cursor.fetchall():
                source_data.append({
                    "source": row[0],
                    "count": row[1],
                    "sentiment": row[2] - row[3],  # Positive - negative for scale
                    "sentiment_positive": row[2],
                    "sentiment_negative": row[3]
                })
            
            return source_data
        except Exception as e:
            logger.error(f"Error getting source coverage: {str(e)}")
            return []
    
    def get_coverage_timeline(self, term: str, time_delta: timedelta) -> List[Dict[str, Any]]:
        """
        Get timeline of coverage by source for a term.
        
        Args:
            term: Term to get timeline for
            time_delta: Time window to look back
            
        Returns:
            List of timeline point dictionaries
        """
        try:
            if self.conn is None:
                self.conn = sqlite3.connect(self.db_path)
            
            cursor = self.conn.cursor()
            
            # Calculate cutoff time
            cutoff_time = datetime.now(pytz.UTC) - time_delta
            
            # Get term ID
            cursor.execute('SELECT id FROM terms WHERE term = ?', (term,))
            result = cursor.fetchone()
            if not result:
                return []
            
            term_id = result[0]
            
            # Get timeline data
            cursor.execute(
                '''
                SELECT a.id, a.source, a.published, a.title, 
                       a.sentiment_positive, a.sentiment_negative
                FROM articles a
                JOIN article_terms at ON a.id = at.article_id
                WHERE at.term_id = ?
                AND a.published >= ?
                ORDER BY a.published
                LIMIT 100
                ''',
                (term_id, cutoff_time)
            )
            
            timeline = []
            for row in cursor.fetchall():
                # Calculate simple impact score (can be refined)
                # Earlier articles have higher impact
                timeline.append({
                    "article_id": row[0],
                    "source": row[1],
                    "timestamp": row[2],
                    "title": row[3],
                    "sentiment": row[4] - row[5],  # Scale from -1 to 1
                    "impact_score": 5  # Default impact score
                })
            
            # Adjust impact scores based on timeline position
            # First articles for each source get higher impact
            sources_seen = set()
            for point in timeline:
                if point["source"] not in sources_seen:
                    point["impact_score"] = 10  # Higher impact for first coverage
                    sources_seen.add(point["source"])
            
            return timeline
        except Exception as e:
            logger.error(f"Error getting coverage timeline: {str(e)}")
            return []
    
    def get_first_sources(self, term: str, time_delta: timedelta) -> List[Tuple[str, datetime, Dict[str, Any]]]:
        """
        Get sources that first covered a term.
        
        Args:
            term: Term to analyze
            time_delta: Time window to look back
            
        Returns:
            List of (source, timestamp, article) tuples
        """
        try:
            if self.conn is None:
                self.conn = sqlite3.connect(self.db_path)
            
            cursor = self.conn.cursor()
            
            # Calculate cutoff time
            cutoff_time = datetime.now(pytz.UTC) - time_delta
            
            # Get term ID
            cursor.execute('SELECT id FROM terms WHERE term = ?', (term,))
            result = cursor.fetchone()
            if not result:
                return []
            
            term_id = result[0]
            
            # Find first article per source
            query = '''
            SELECT a.source, MIN(a.published) as first_time
            FROM articles a
            JOIN article_terms at ON a.id = at.article_id
            WHERE at.term_id = ?
            AND a.published >= ?
            GROUP BY a.source
            ORDER BY first_time
            LIMIT 10
            '''
            
            cursor.execute(query, (term_id, cutoff_time))
            first_times = cursor.fetchall()
            
            # Get the actual articles
            first_sources = []
            for source, timestamp in first_times:
                cursor.execute(
                    '''
                    SELECT a.id, a.title, a.link, a.published
                    FROM articles a
                    JOIN article_terms at ON a.id = at.article_id
                    WHERE at.term_id = ?
                    AND a.source = ?
                    AND a.published = ?
                    LIMIT 1
                    ''',
                    (term_id, source, timestamp)
                )
                
                result = cursor.fetchone()
                if result:
                    article = {
                        "id": result[0],
                        "title": result[1],
                        "link": result[2],
                        "published": result[3]
                    }
                    first_sources.append((source, timestamp, article))
            
            return first_sources
        except Exception as e:
            logger.error(f"Error getting first sources: {str(e)}")
            return []
    
    def get_related_terms(self, term: str, time_delta: timedelta) -> List[Tuple[str, int]]:
        """
        Get terms related to a specific term.
        
        Args:
            term: Term to find related terms for
            time_delta: Time window to look back
            
        Returns:
            List of (term, count) tuples
        """
        try:
            if self.conn is None:
                self.conn = sqlite3.connect(self.db_path)
            
            cursor = self.conn.cursor()
            
            # Calculate cutoff time
            cutoff_time = datetime.now(pytz.UTC) - time_delta
            
            # Get term ID
            cursor.execute('SELECT id FROM terms WHERE term = ?', (term,))
            result = cursor.fetchone()
            if not result:
                return []
            
            term_id = result[0]
            
            # Get co-occurring terms
            cursor.execute(
                '''
                SELECT t.term, SUM(tc.count) as co_count
                FROM term_cooccurrences tc
                JOIN terms t ON (tc.term1_id = t.id OR tc.term2_id = t.id)
                WHERE (tc.term1_id = ? OR tc.term2_id = ?)
                AND t.id != ?
                AND tc.last_seen >= ?
                GROUP BY t.id
                ORDER BY co_count DESC
                LIMIT 15
                ''',
                (term_id, term_id, term_id, cutoff_time)
            )
            
            return [(row[0], row[1]) for row in cursor.fetchall()]
        except Exception as e:
            logger.error(f"Error getting related terms: {str(e)}")
            return []
