import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import pytz

from feed_manager import FeedManager
from feed_parser import NewsFeedParser

# Page configuration
st.set_page_config(
    page_title="RSS Reader",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize components
@st.cache_resource
def initialize_components():
    feed_manager = FeedManager()
    feed_parser = NewsFeedParser()
    return feed_manager, feed_parser

feed_manager, feed_parser = initialize_components()

# Custom CSS for better styling
st.markdown("""
<style>
    .feed-item {
        padding: 15px;
        margin: 10px 0;
        border-left: 3px solid #1f77b4;
        background-color: #f0f2f6;
        border-radius: 5px;
    }
    .article-title {
        font-size: 18px;
        font-weight: bold;
        color: #1f77b4;
        text-decoration: none;
    }
    .article-meta {
        color: #666;
        font-size: 14px;
        margin-top: 5px;
    }
    .folder-header {
        background-color: #1f77b4;
        color: white;
        padding: 10px;
        border-radius: 5px;
        margin-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar - Feed Management
st.sidebar.title("📰 RSS Reader")

# Session state for UI
if 'current_folder' not in st.session_state:
    st.session_state.current_folder = None
if 'show_add_feed' not in st.session_state:
    st.session_state.show_add_feed = False
if 'show_add_folder' not in st.session_state:
    st.session_state.show_add_folder = False

# Sidebar - Folder Management
st.sidebar.header("Folders")

# Add new folder button
if st.sidebar.button("➕ Add Folder"):
    st.session_state.show_add_folder = not st.session_state.show_add_folder

# Add folder form
if st.session_state.show_add_folder:
    with st.sidebar.form("add_folder_form"):
        new_folder_name = st.text_input("Folder Name")
        new_folder_desc = st.text_area("Description (optional)")
        submit_folder = st.form_submit_button("Create Folder")

        if submit_folder and new_folder_name:
            try:
                folder_id = feed_manager.create_folder(new_folder_name, new_folder_desc)
                st.success(f"Created folder: {new_folder_name}")
                st.session_state.show_add_folder = False
                st.rerun()
            except Exception as e:
                st.error(f"Error creating folder: {str(e)}")

# Display folders
folders = feed_manager.get_all_folders()

# All feeds option
if st.sidebar.button("📚 All Feeds", use_container_width=True):
    st.session_state.current_folder = "all"
    st.rerun()

# Uncategorized feeds option
uncategorized_feeds = feed_manager.get_feeds_by_folder(None)
if uncategorized_feeds:
    if st.sidebar.button(f"📂 Uncategorized ({len(uncategorized_feeds)})", use_container_width=True):
        st.session_state.current_folder = None
        st.rerun()

# Display folder list
for folder in folders:
    if st.sidebar.button(
        f"📁 {folder['name']} ({folder['feed_count']})",
        key=f"folder_{folder['id']}",
        use_container_width=True
    ):
        st.session_state.current_folder = folder['id']
        st.rerun()

st.sidebar.divider()

# Add new feed button
if st.sidebar.button("➕ Add Feed"):
    st.session_state.show_add_feed = not st.session_state.show_add_feed

# Add feed form
if st.session_state.show_add_feed:
    with st.sidebar.form("add_feed_form"):
        st.subheader("Add New Feed")
        new_feed_url = st.text_input("Feed URL*", placeholder="https://example.com/rss")
        new_feed_title = st.text_input("Title (optional)", placeholder="My News Feed")

        # Folder selection
        folder_options = {"Uncategorized": None}
        for folder in folders:
            folder_options[folder['name']] = folder['id']

        selected_folder_name = st.selectbox("Folder", list(folder_options.keys()))
        new_feed_folder = folder_options[selected_folder_name]

        new_feed_freq = st.slider("Update Frequency (minutes)", 5, 120, 15)

        submit_feed = st.form_submit_button("Add Feed")

        if submit_feed and new_feed_url:
            try:
                feed_id = feed_manager.add_feed(
                    url=new_feed_url,
                    title=new_feed_title,
                    folder_id=new_feed_folder,
                    update_frequency=new_feed_freq
                )
                st.success(f"Added feed: {new_feed_title or new_feed_url}")
                st.session_state.show_add_feed = False
                st.rerun()
            except Exception as e:
                st.error(f"Error adding feed: {str(e)}")

# Sidebar - Settings
st.sidebar.divider()
st.sidebar.header("Settings")

auto_refresh = st.sidebar.checkbox("Auto-refresh feeds", value=False)
if auto_refresh:
    refresh_interval = st.sidebar.slider("Refresh interval (minutes)", 5, 60, 15)

# Main content area
st.title("📰 RSS Feed Reader")

# Determine which feeds to display
if st.session_state.current_folder == "all":
    st.subheader("All Feeds")
    display_feeds = feed_manager.get_all_feeds(active_only=True)
elif st.session_state.current_folder is None:
    st.subheader("Uncategorized Feeds")
    display_feeds = feed_manager.get_feeds_by_folder(None)
else:
    # Find folder name
    folder_info = next((f for f in folders if f['id'] == st.session_state.current_folder), None)
    if folder_info:
        st.subheader(f"📁 {folder_info['name']}")
        if folder_info['description']:
            st.caption(folder_info['description'])
    display_feeds = feed_manager.get_feeds_by_folder(st.session_state.current_folder)

# Display feed count and refresh button
col1, col2, col3 = st.columns([2, 1, 1])
with col1:
    st.write(f"**{len(display_feeds)} feed(s)**")
with col2:
    if st.button("🔄 Refresh Feeds"):
        with st.spinner("Fetching feeds..."):
            feed_urls = [feed['url'] for feed in display_feeds]
            feed_parser.set_feeds(feed_urls)
            feed_parser.fetch_all()

            # Update last fetched time for each feed
            for feed in display_feeds:
                feed_manager.mark_feed_fetched(feed['id'])

            st.success("Feeds refreshed!")
            st.rerun()

# Time range selector
with col3:
    time_range = st.selectbox(
        "Time range",
        ["Last Hour", "Last 6 Hours", "Last 24 Hours", "Last 3 Days", "Last Week"],
        index=2
    )

time_ranges = {
    "Last Hour": timedelta(hours=1),
    "Last 6 Hours": timedelta(hours=6),
    "Last 24 Hours": timedelta(hours=24),
    "Last 3 Days": timedelta(days=3),
    "Last Week": timedelta(days=7),
}
time_delta = time_ranges[time_range]

# Display feeds and their articles
if not display_feeds:
    st.info("No feeds in this folder. Click 'Add Feed' in the sidebar to add your first RSS feed!")
else:
    # Create tabs for each feed
    if len(display_feeds) == 1:
        # If only one feed, don't use tabs
        feed = display_feeds[0]
        st.markdown(f"### {feed['title'] or feed['url']}")

        # Feed management options
        with st.expander("⚙️ Feed Settings"):
            col1, col2 = st.columns(2)
            with col1:
                if st.button("🗑️ Delete Feed", key=f"delete_{feed['id']}"):
                    feed_manager.delete_feed(feed['id'])
                    st.success("Feed deleted")
                    st.rerun()
            with col2:
                # Move to folder
                folder_options = {"Uncategorized": None}
                for folder in folders:
                    folder_options[folder['name']] = folder['id']

                current_folder_name = "Uncategorized"
                if feed['folder_id']:
                    folder_info = next((f for f in folders if f['id'] == feed['folder_id']), None)
                    if folder_info:
                        current_folder_name = folder_info['name']

                new_folder_name = st.selectbox(
                    "Move to folder",
                    list(folder_options.keys()),
                    index=list(folder_options.keys()).index(current_folder_name),
                    key=f"move_{feed['id']}"
                )

                if new_folder_name != current_folder_name:
                    new_folder_id = folder_options[new_folder_name]
                    feed_manager.move_feed(feed['id'], new_folder_id)
                    st.success(f"Moved to {new_folder_name}")
                    st.rerun()

        # Display articles
        feed_parser.set_feeds([feed['url']])
        articles = feed_parser.get_recent_articles(time_delta)

        if articles:
            st.write(f"**{len(articles)} article(s)**")
            for article in articles:
                with st.container():
                    st.markdown(f"#### [{article['title']}]({article['link']})")

                    # Meta information
                    meta_col1, meta_col2, meta_col3 = st.columns(3)
                    with meta_col1:
                        st.caption(f"📅 {article['published'].strftime('%Y-%m-%d %H:%M')}")
                    with meta_col2:
                        st.caption(f"📰 {article['source']}")
                    with meta_col3:
                        st.caption(f"🔗 [Link]({article['link']})")

                    # Summary or content
                    if article.get('summary') or article.get('content'):
                        content = article.get('content') or article.get('summary')
                        # Truncate if too long
                        if len(content) > 500:
                            content = content[:500] + "..."
                        st.write(content)

                    st.divider()
        else:
            st.info(f"No articles found in the selected time range ({time_range})")

    else:
        # Multiple feeds - use tabs
        feed_tabs = st.tabs([feed['title'] or feed['url'] for feed in display_feeds])

        for idx, (tab, feed) in enumerate(zip(feed_tabs, display_feeds)):
            with tab:
                # Feed management options
                with st.expander("⚙️ Feed Settings"):
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button("🗑️ Delete Feed", key=f"delete_{feed['id']}"):
                            feed_manager.delete_feed(feed['id'])
                            st.success("Feed deleted")
                            st.rerun()
                    with col2:
                        # Move to folder
                        folder_options = {"Uncategorized": None}
                        for folder in folders:
                            folder_options[folder['name']] = folder['id']

                        current_folder_name = "Uncategorized"
                        if feed['folder_id']:
                            folder_info = next((f for f in folders if f['id'] == feed['folder_id']), None)
                            if folder_info:
                                current_folder_name = folder_info['name']

                        new_folder_name = st.selectbox(
                            "Move to folder",
                            list(folder_options.keys()),
                            index=list(folder_options.keys()).index(current_folder_name),
                            key=f"move_{feed['id']}"
                        )

                        if new_folder_name != current_folder_name:
                            new_folder_id = folder_options[new_folder_name]
                            feed_manager.move_feed(feed['id'], new_folder_id)
                            st.success(f"Moved to {new_folder_name}")
                            st.rerun()

                # Display articles
                feed_parser.set_feeds([feed['url']])
                articles = feed_parser.get_recent_articles(time_delta)

                if articles:
                    st.write(f"**{len(articles)} article(s)**")
                    for article in articles:
                        with st.container():
                            st.markdown(f"#### [{article['title']}]({article['link']})")

                            # Meta information
                            meta_col1, meta_col2, meta_col3 = st.columns(3)
                            with meta_col1:
                                st.caption(f"📅 {article['published'].strftime('%Y-%m-%d %H:%M')}")
                            with meta_col2:
                                st.caption(f"📰 {article['source']}")
                            with meta_col3:
                                st.caption(f"🔗 [Link]({article['link']})")

                            # Summary or content
                            if article.get('summary') or article.get('content'):
                                content = article.get('content') or article.get('summary')
                                # Truncate if too long
                                if len(content) > 500:
                                    content = content[:500] + "..."
                                st.write(content)

                            st.divider()
                else:
                    st.info(f"No articles found in the selected time range ({time_range})")

# Footer
st.divider()
st.caption("RSS Feed Reader - Organize and read your feeds for AI analysis")
