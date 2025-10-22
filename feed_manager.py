import sqlite3
import os
import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import pytz

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class FeedManager:
    """Manages RSS feeds and their organization into folders."""

    def __init__(self, db_path: str = "data/feeds.db"):
        """
        Initialize the feed manager.

        Args:
            db_path: Path to SQLite database file
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(db_path), exist_ok=True)

        self.db_path = db_path
        self.conn = None

        # Initialize database
        self._initialize_db()

        logger.info(f"Feed manager initialized at {db_path}")

    def _initialize_db(self) -> None:
        """Initialize the database schema if it doesn't exist."""
        try:
            self.conn = sqlite3.connect(self.db_path)
            cursor = self.conn.cursor()

            # Folders table
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS folders (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE NOT NULL,
                description TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            ''')

            # Feeds table
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS feeds (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                url TEXT UNIQUE NOT NULL,
                title TEXT,
                folder_id INTEGER,
                active BOOLEAN DEFAULT 1,
                update_frequency INTEGER DEFAULT 15,
                last_fetched TIMESTAMP,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (folder_id) REFERENCES folders (id) ON DELETE SET NULL
            )
            ''')

            # Create indices for performance
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_feeds_folder ON feeds (folder_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_feeds_active ON feeds (active)')

            self.conn.commit()

            # Create default folder if no folders exist
            cursor.execute('SELECT COUNT(*) FROM folders')
            if cursor.fetchone()[0] == 0:
                self.create_folder("General", "Default folder for uncategorized feeds")

        except Exception as e:
            logger.error(f"Error initializing database: {str(e)}")
            raise

    # Folder operations

    def create_folder(self, name: str, description: str = "") -> int:
        """
        Create a new folder.

        Args:
            name: Name of the folder
            description: Optional description

        Returns:
            Folder ID
        """
        try:
            if self.conn is None:
                self.conn = sqlite3.connect(self.db_path)

            cursor = self.conn.cursor()
            cursor.execute(
                'INSERT INTO folders (name, description) VALUES (?, ?)',
                (name, description)
            )
            self.conn.commit()

            folder_id = cursor.lastrowid
            logger.info(f"Created folder '{name}' with ID {folder_id}")
            return folder_id
        except sqlite3.IntegrityError:
            logger.warning(f"Folder '{name}' already exists")
            # Return existing folder ID
            cursor.execute('SELECT id FROM folders WHERE name = ?', (name,))
            return cursor.fetchone()[0]
        except Exception as e:
            logger.error(f"Error creating folder: {str(e)}")
            raise

    def get_all_folders(self) -> List[Dict[str, Any]]:
        """
        Get all folders.

        Returns:
            List of folder dictionaries
        """
        try:
            if self.conn is None:
                self.conn = sqlite3.connect(self.db_path)

            cursor = self.conn.cursor()
            cursor.execute('''
                SELECT f.id, f.name, f.description, f.created_at,
                       COUNT(fe.id) as feed_count
                FROM folders f
                LEFT JOIN feeds fe ON f.id = fe.folder_id AND fe.active = 1
                GROUP BY f.id
                ORDER BY f.name
            ''')

            folders = []
            for row in cursor.fetchall():
                folders.append({
                    'id': row[0],
                    'name': row[1],
                    'description': row[2],
                    'created_at': row[3],
                    'feed_count': row[4]
                })

            return folders
        except Exception as e:
            logger.error(f"Error getting folders: {str(e)}")
            return []

    def rename_folder(self, folder_id: int, new_name: str) -> bool:
        """
        Rename a folder.

        Args:
            folder_id: ID of the folder
            new_name: New name for the folder

        Returns:
            True if successful
        """
        try:
            if self.conn is None:
                self.conn = sqlite3.connect(self.db_path)

            cursor = self.conn.cursor()
            cursor.execute(
                'UPDATE folders SET name = ? WHERE id = ?',
                (new_name, folder_id)
            )
            self.conn.commit()

            logger.info(f"Renamed folder {folder_id} to '{new_name}'")
            return True
        except Exception as e:
            logger.error(f"Error renaming folder: {str(e)}")
            return False

    def delete_folder(self, folder_id: int, move_feeds_to: Optional[int] = None) -> bool:
        """
        Delete a folder.

        Args:
            folder_id: ID of the folder to delete
            move_feeds_to: Optional folder ID to move feeds to (otherwise feeds become uncategorized)

        Returns:
            True if successful
        """
        try:
            if self.conn is None:
                self.conn = sqlite3.connect(self.db_path)

            cursor = self.conn.cursor()

            # Move or uncategorize feeds
            if move_feeds_to:
                cursor.execute(
                    'UPDATE feeds SET folder_id = ? WHERE folder_id = ?',
                    (move_feeds_to, folder_id)
                )
            else:
                cursor.execute(
                    'UPDATE feeds SET folder_id = NULL WHERE folder_id = ?',
                    (folder_id,)
                )

            # Delete folder
            cursor.execute('DELETE FROM folders WHERE id = ?', (folder_id,))
            self.conn.commit()

            logger.info(f"Deleted folder {folder_id}")
            return True
        except Exception as e:
            logger.error(f"Error deleting folder: {str(e)}")
            return False

    # Feed operations

    def add_feed(self, url: str, title: str = "", folder_id: Optional[int] = None,
                 update_frequency: int = 15) -> int:
        """
        Add a new RSS feed.

        Args:
            url: RSS feed URL
            title: Optional title for the feed
            folder_id: Optional folder ID to organize the feed
            update_frequency: Update frequency in minutes (default: 15)

        Returns:
            Feed ID
        """
        try:
            if self.conn is None:
                self.conn = sqlite3.connect(self.db_path)

            cursor = self.conn.cursor()
            cursor.execute(
                '''
                INSERT INTO feeds (url, title, folder_id, update_frequency)
                VALUES (?, ?, ?, ?)
                ''',
                (url, title, folder_id, update_frequency)
            )
            self.conn.commit()

            feed_id = cursor.lastrowid
            logger.info(f"Added feed '{url}' with ID {feed_id}")
            return feed_id
        except sqlite3.IntegrityError:
            logger.warning(f"Feed '{url}' already exists")
            # Return existing feed ID
            cursor.execute('SELECT id FROM feeds WHERE url = ?', (url,))
            result = cursor.fetchone()
            return result[0] if result else -1
        except Exception as e:
            logger.error(f"Error adding feed: {str(e)}")
            raise

    def get_all_feeds(self, active_only: bool = True) -> List[Dict[str, Any]]:
        """
        Get all feeds.

        Args:
            active_only: If True, return only active feeds

        Returns:
            List of feed dictionaries
        """
        try:
            if self.conn is None:
                self.conn = sqlite3.connect(self.db_path)

            cursor = self.conn.cursor()

            query = '''
                SELECT f.id, f.url, f.title, f.folder_id, f.active,
                       f.update_frequency, f.last_fetched, f.created_at,
                       fo.name as folder_name
                FROM feeds f
                LEFT JOIN folders fo ON f.folder_id = fo.id
            '''

            if active_only:
                query += ' WHERE f.active = 1'

            query += ' ORDER BY fo.name, f.title, f.url'

            cursor.execute(query)

            feeds = []
            for row in cursor.fetchall():
                feeds.append({
                    'id': row[0],
                    'url': row[1],
                    'title': row[2],
                    'folder_id': row[3],
                    'active': bool(row[4]),
                    'update_frequency': row[5],
                    'last_fetched': row[6],
                    'created_at': row[7],
                    'folder_name': row[8] if row[8] else 'Uncategorized'
                })

            return feeds
        except Exception as e:
            logger.error(f"Error getting feeds: {str(e)}")
            return []

    def get_feeds_by_folder(self, folder_id: Optional[int]) -> List[Dict[str, Any]]:
        """
        Get all feeds in a specific folder.

        Args:
            folder_id: Folder ID (None for uncategorized feeds)

        Returns:
            List of feed dictionaries
        """
        try:
            if self.conn is None:
                self.conn = sqlite3.connect(self.db_path)

            cursor = self.conn.cursor()

            if folder_id is None:
                cursor.execute('''
                    SELECT id, url, title, folder_id, active, update_frequency, last_fetched, created_at
                    FROM feeds
                    WHERE folder_id IS NULL AND active = 1
                    ORDER BY title, url
                ''')
            else:
                cursor.execute('''
                    SELECT id, url, title, folder_id, active, update_frequency, last_fetched, created_at
                    FROM feeds
                    WHERE folder_id = ? AND active = 1
                    ORDER BY title, url
                ''', (folder_id,))

            feeds = []
            for row in cursor.fetchall():
                feeds.append({
                    'id': row[0],
                    'url': row[1],
                    'title': row[2],
                    'folder_id': row[3],
                    'active': bool(row[4]),
                    'update_frequency': row[5],
                    'last_fetched': row[6],
                    'created_at': row[7]
                })

            return feeds
        except Exception as e:
            logger.error(f"Error getting feeds by folder: {str(e)}")
            return []

    def update_feed(self, feed_id: int, **kwargs) -> bool:
        """
        Update feed properties.

        Args:
            feed_id: Feed ID
            **kwargs: Properties to update (url, title, folder_id, active, update_frequency)

        Returns:
            True if successful
        """
        try:
            if self.conn is None:
                self.conn = sqlite3.connect(self.db_path)

            cursor = self.conn.cursor()

            # Build update query
            allowed_fields = ['url', 'title', 'folder_id', 'active', 'update_frequency']
            updates = []
            values = []

            for key, value in kwargs.items():
                if key in allowed_fields:
                    updates.append(f"{key} = ?")
                    values.append(value)

            if not updates:
                logger.warning("No valid fields to update")
                return False

            values.append(feed_id)
            query = f"UPDATE feeds SET {', '.join(updates)} WHERE id = ?"

            cursor.execute(query, values)
            self.conn.commit()

            logger.info(f"Updated feed {feed_id}")
            return True
        except Exception as e:
            logger.error(f"Error updating feed: {str(e)}")
            return False

    def move_feed(self, feed_id: int, new_folder_id: Optional[int]) -> bool:
        """
        Move a feed to a different folder.

        Args:
            feed_id: Feed ID
            new_folder_id: New folder ID (None for uncategorized)

        Returns:
            True if successful
        """
        return self.update_feed(feed_id, folder_id=new_folder_id)

    def delete_feed(self, feed_id: int, soft_delete: bool = True) -> bool:
        """
        Delete a feed.

        Args:
            feed_id: Feed ID
            soft_delete: If True, mark as inactive; if False, permanently delete

        Returns:
            True if successful
        """
        try:
            if self.conn is None:
                self.conn = sqlite3.connect(self.db_path)

            cursor = self.conn.cursor()

            if soft_delete:
                cursor.execute('UPDATE feeds SET active = 0 WHERE id = ?', (feed_id,))
            else:
                cursor.execute('DELETE FROM feeds WHERE id = ?', (feed_id,))

            self.conn.commit()

            logger.info(f"{'Deactivated' if soft_delete else 'Deleted'} feed {feed_id}")
            return True
        except Exception as e:
            logger.error(f"Error deleting feed: {str(e)}")
            return False

    def mark_feed_fetched(self, feed_id: int) -> bool:
        """
        Mark a feed as fetched (update last_fetched timestamp).

        Args:
            feed_id: Feed ID

        Returns:
            True if successful
        """
        try:
            if self.conn is None:
                self.conn = sqlite3.connect(self.db_path)

            cursor = self.conn.cursor()
            now = datetime.now(pytz.UTC).isoformat()
            cursor.execute(
                'UPDATE feeds SET last_fetched = ? WHERE id = ?',
                (now, feed_id)
            )
            self.conn.commit()

            return True
        except Exception as e:
            logger.error(f"Error marking feed as fetched: {str(e)}")
            return False

    def get_feed_urls(self, folder_id: Optional[int] = None) -> List[str]:
        """
        Get list of feed URLs, optionally filtered by folder.

        Args:
            folder_id: Optional folder ID to filter by

        Returns:
            List of feed URLs
        """
        try:
            if self.conn is None:
                self.conn = sqlite3.connect(self.db_path)

            cursor = self.conn.cursor()

            if folder_id is None:
                cursor.execute('SELECT url FROM feeds WHERE active = 1')
            else:
                cursor.execute('SELECT url FROM feeds WHERE folder_id = ? AND active = 1', (folder_id,))

            return [row[0] for row in cursor.fetchall()]
        except Exception as e:
            logger.error(f"Error getting feed URLs: {str(e)}")
            return []

    def close(self) -> None:
        """Close database connection."""
        if self.conn:
            self.conn.close()
            self.conn = None
