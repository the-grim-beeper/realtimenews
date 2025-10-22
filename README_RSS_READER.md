# RSS Feed Reader

A simple, organized RSS feed reader built with Streamlit that allows you to download and store RSS feeds in folders for further AI analysis.

## Features

- **Folder Organization**: Organize your RSS feeds into custom folders
- **Feed Management**: Add, edit, move, and delete feeds easily
- **Article Viewing**: Browse articles from your feeds with time-based filtering
- **Clean Interface**: Simple, intuitive UI for managing and reading feeds
- **Persistent Storage**: All feeds and folders are stored in a SQLite database

## Getting Started

### Prerequisites

Make sure you have the required dependencies installed:

```bash
pip install -r requirements.txt
```

### Running the Application

Launch the RSS reader:

```bash
streamlit run rss_reader.py
```

The application will open in your default web browser.

## How to Use

### 1. Create Folders

- Click **"Add Folder"** in the sidebar
- Enter a folder name and optional description
- Click **"Create Folder"**

Example folders:
- Technology News
- Business
- Science & Research
- Local News

### 2. Add RSS Feeds

- Click **"Add Feed"** in the sidebar
- Enter the RSS feed URL (required)
- Optionally add a title for the feed
- Select which folder to organize it in
- Set the update frequency
- Click **"Add Feed"**

Example RSS feeds to try:
- Technology: `https://news.google.com/rss/search?q=technology`
- Science: `https://news.google.com/rss/search?q=science`
- Business: `https://news.google.com/rss/search?q=business`

### 3. Browse Your Feeds

- Click on any folder in the sidebar to view its feeds
- Click **"All Feeds"** to see all feeds at once
- Click **"Uncategorized"** to see feeds without a folder

### 4. Read Articles

- Select a feed to view its articles
- Use the time range selector to filter articles (Last Hour, Last 24 Hours, etc.)
- Click **"Refresh Feeds"** to fetch the latest articles
- Click on article titles to read the full content on the source website

### 5. Manage Feeds

Each feed has a settings menu where you can:
- **Delete** the feed
- **Move** the feed to a different folder

## Database Storage

All data is stored in SQLite databases:

- **Feed configuration**: `data/feeds.db`
  - Stores folders and feed configurations
  - Tracks feed URLs, titles, and organization

- **Cached articles**: `data/raw/articles_cache.json`
  - Stores downloaded articles for quick access
  - Automatically managed by the feed parser

## Next Steps: AI Analysis

This RSS reader is designed as the foundation for AI-powered news analysis. Once you have feeds organized and downloading, you can:

1. Integrate NLP processing to extract key terms and entities
2. Perform sentiment analysis on articles
3. Track narrative shifts over time
4. Identify media bias patterns
5. Detect coverage blind spots

The article data is stored and ready for analysis when you're ready to add AI capabilities.

## File Structure

```
realtimenews/
├── rss_reader.py          # Main RSS reader application
├── feed_manager.py        # Feed and folder management logic
├── feed_parser.py         # RSS feed parsing and downloading
├── requirements.txt       # Python dependencies
├── data/
│   ├── feeds.db          # Feed configuration database
│   └── raw/
│       └── articles_cache.json  # Cached articles
```

## Tips

- Start with a few feeds in different categories to test the organization
- Use descriptive folder names to keep feeds organized
- Set appropriate update frequencies based on how often feeds publish
- The "All Feeds" view is useful for getting a comprehensive overview
- Articles are cached, so you can browse previously downloaded content even offline

## Troubleshooting

**No articles showing:**
- Make sure you clicked "Refresh Feeds" to download articles
- Check that the RSS feed URL is valid
- Try selecting a wider time range (e.g., "Last Week")

**Feed won't add:**
- Verify the URL is a valid RSS feed
- Check that the feed isn't already added
- Some feeds may require specific user agents or have access restrictions

**Application won't start:**
- Ensure all dependencies are installed: `pip install -r requirements.txt`
- Check that you're in the correct directory
- Make sure no other Streamlit apps are running on the same port
