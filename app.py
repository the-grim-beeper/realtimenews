import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import networkx as nx
import time
from datetime import datetime, timedelta
import pytz
import plotly.express as px

from feed_parser import NewsFeedParser
from nlp_processor import NLPProcessor
from database import TimeSeriesDB
from visualizations import (
    create_term_network, 
    create_narrative_shift_chart,
    create_media_bias_chart,
    create_blindspot_visualization
)

# Page configuration
st.set_page_config(
    page_title="News Term Monitor",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize components
@st.cache_resource
def initialize_components():
    feed_parser = NewsFeedParser()
    nlp = NLPProcessor()
    db = TimeSeriesDB()
    return feed_parser, nlp, db

feed_parser, nlp, db = initialize_components()

# Sidebar controls
st.sidebar.title("News Term Monitor")

# Add RSS feeds
with st.sidebar.expander("RSS Feed Sources", expanded=False):
    default_feeds = [
        "https://news.google.com/rss/search?q=technology",
        "https://news.google.com/rss/search?q=politics",
        "https://news.google.com/rss/search?q=science",
        "https://news.google.com/rss/search?q=health",
        "https://news.google.com/rss/search?q=business"
    ]
    
    feeds = st.text_area("Add RSS Feeds (one per line)", "\n".join(default_feeds))
    feed_list = [f.strip() for f in feeds.split("\n") if f.strip()]
    
    if st.button("Update Feed Sources"):
        feed_parser.set_feeds(feed_list)
        st.success(f"Updated {len(feed_list)} feed sources!")

# Time range selector
time_ranges = {
    "Last Hour": timedelta(hours=1),
    "Last 6 Hours": timedelta(hours=6),
    "Last 24 Hours": timedelta(hours=24),
    "Last 3 Days": timedelta(days=3),
    "Last Week": timedelta(days=7),
}
selected_range = st.sidebar.selectbox("Time Range", list(time_ranges.keys()))
time_delta = time_ranges[selected_range]

# Update frequency
update_frequency = st.sidebar.slider("Update Frequency (minutes)", 5, 60, 15)

# Main dashboard
st.title("📰 News Term Monitor")
st.caption("Real-time analysis of trending terms in news feeds")

# Info row
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Active Feeds", len(feed_list))
with col2:
    st.metric("Articles Analyzed", db.get_article_count())
with col3:
    last_update = datetime.now(pytz.UTC).strftime("%H:%M:%S UTC")
    st.metric("Last Update", last_update)
with col4:
    st.metric("Unique Terms", db.get_unique_term_count())

# Auto-refresh mechanism
if st.sidebar.button("Refresh Data Now"):
    with st.spinner("Fetching and processing news feeds..."):
        feed_parser.fetch_all()
        articles = feed_parser.get_recent_articles(time_delta)
        for article in articles:
            terms = nlp.process_article(article)
            db.store_article_terms(article, terms)
    st.success(f"Processed {len(articles)} articles")

# Main dashboard tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Term Networks", 
    "📈 Narrative Shifts", 
    "🔍 Media Bias", 
    "⚠️ Blind Spots"
])

# Tab 1: Term Networks
with tab1:
    st.header("Contextual Term Networks")
    st.write("Discover surprising connections between terms and topics")
    
    # Controls for network
    min_edge_weight = st.slider("Minimum Connection Strength", 1, 10, 2)
    
    # Generate the network
    term_data = db.get_term_data(time_delta)
    if term_data:
        G, pos = create_term_network(term_data, min_edge_weight)
        if G.number_of_nodes() > 0:
            st.plotly_chart(
                go.Figure(
                    data=nx.to_plotly_graph(G, pos=pos),
                    layout=go.Layout(
                        title="Term Co-occurrence Network",
                        showlegend=False,
                        hovermode="closest",
                        margin=dict(b=20, l=5, r=5, t=40),
                    )
                ),
                use_container_width=True,
            )
        else:
            st.info("Not enough term co-occurrence data available for the selected time range.")
    else:
        st.info("No term data available. Try refreshing or selecting a wider time range.")

    # Unexpected connections
    st.subheader("Surprising Term Connections")
    surprises = nlp.get_surprising_connections(db.get_term_data(time_delta))
    if surprises:
        for i, (term1, term2, strength, explanation) in enumerate(surprises, 1):
            with st.expander(f"{term1} ⟷ {term2} (Surprise Score: {strength:.2f})"):
                st.write(explanation)
                st.write("**Articles mentioning both terms:**")
                articles = db.get_articles_with_terms([term1, term2], time_delta)
                for article in articles:
                    st.markdown(f"- [{article['title']}]({article['link']})")
    else:
        st.info("No surprising connections found in the current time range.")

# Tab 2: Narrative Shifts
with tab2:
    st.header("Narrative Shift Detection")
    st.write("Track how topic framing changes over time")
    
    # Term selector
    all_terms = db.get_top_terms(50, time_delta)
    selected_term = st.selectbox("Select Term to Analyze", all_terms)
    
    if selected_term:
        # Term context over time
        narrative_data = db.get_term_context_over_time(selected_term, time_delta)
        if narrative_data:
            fig = create_narrative_shift_chart(narrative_data)
            st.plotly_chart(fig, use_container_width=True)
            
            # Detect significant shifts
            shifts = nlp.detect_narrative_shifts(narrative_data)
            if shifts:
                st.subheader("Significant Narrative Shifts")
                for timestamp, score, description in shifts:
                    with st.expander(f"{timestamp.strftime('%Y-%m-%d %H:%M')} (Shift Score: {score:.2f})"):
                        st.write(description)
                        st.write("**Pivotal articles:**")
                        articles = db.get_articles_for_shift(selected_term, timestamp)
                        for article in articles:
                            st.markdown(f"- [{article['title']}]({article['link']}) - {article['source']}")
            else:
                st.info("No significant narrative shifts detected for this term in the time range.")
        else:
            st.info(f"Not enough context data for '{selected_term}' in the selected time range.")
    else:
        st.info("No terms available. Try refreshing or selecting a wider time range.")

# Tab 3: Media Bias
with tab3:
    st.header("Media Bias Analyzer") 
    st.write("Compare how different sources cover the same topics")
    
    # Term selector for bias analysis
    bias_term = st.selectbox("Select Term to Analyze Coverage", all_terms, key="bias_term")
    
    if bias_term:
        # Source comparison
        source_data = db.get_source_coverage(bias_term, time_delta)
        if source_data:
            # Volume chart
            fig_volume = px.bar(
                source_data, 
                x="source", 
                y="count", 
                title=f"Coverage Volume: {bias_term}",
                color="sentiment",
                color_continuous_scale="RdBu",
                labels={"count": "Article Count", "source": "News Source"}
            )
            st.plotly_chart(fig_volume, use_container_width=True)
            
            # Sentiment and framing
            fig_bias = create_media_bias_chart(source_data)
            st.plotly_chart(fig_bias, use_container_width=True)
            
            # Leaders vs Followers
            st.subheader("Coverage Timeline")
            timeline_data = db.get_coverage_timeline(bias_term, time_delta)
            if timeline_data:
                fig_timeline = px.scatter(
                    timeline_data,
                    x="timestamp",
                    y="source",
                    size="impact_score",
                    color="sentiment",
                    color_continuous_scale="RdBu",
                    title=f"Who Led Coverage on '{bias_term}'",
                    labels={"timestamp": "Time", "source": "News Source"}
                )
                st.plotly_chart(fig_timeline, use_container_width=True)
                
                # First to cover
                first_sources = db.get_first_sources(bias_term, time_delta)
                if first_sources:
                    st.subheader("First to Cover")
                    for source, timestamp, article in first_sources[:3]:
                        st.markdown(f"**{source}** - {timestamp.strftime('%Y-%m-%d %H:%M')}")
                        st.markdown(f"[{article['title']}]({article['link']})")
            else:
                st.info("Not enough timeline data available.")
        else:
            st.info(f"Not enough source data for '{bias_term}' in the selected time range.")
    else:
        st.info("No terms available. Try refreshing or selecting a wider time range.")

# Tab 4: Blind Spots
with tab4:
    st.header("Blind Spot Detector")
    st.write("Discover important topics getting minimal coverage")
    
    # Generate blind spots
    blind_spots = nlp.detect_blind_spots(db.get_term_data(time_delta))
    if blind_spots:
        fig = create_blindspot_visualization(blind_spots)
        st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("Potential Coverage Gaps")
        for topic, importance, coverage, explanation in blind_spots:
            with st.expander(f"{topic} (Importance: {importance:.2f}, Coverage: {coverage:.2f})"):
                st.write(explanation)
                st.write("**Related terms receiving more coverage:**")
                related = db.get_related_terms(topic, time_delta)
                for term, count in related:
                    st.markdown(f"- {term}: {count} mentions")
    else:
        st.info("No significant blind spots detected in the current time range.")

# Auto-refresh mechanism (hidden)
if st.sidebar.checkbox("Enable Auto-Refresh", True):
    time_placeholder = st.empty()
    while True:
        time_placeholder.metric("Next Update In", f"{update_frequency} min")
        time.sleep(update_frequency * 60)
        with st.spinner("Auto-refreshing data..."):
            feed_parser.fetch_all()
            articles = feed_parser.get_recent_articles(time_delta)
            for article in articles:
                terms = nlp.process_article(article)
                db.store_article_terms(article, terms)
        st.experimental_rerun()
