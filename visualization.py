import networkx as nx
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
import logging
from datetime import datetime
import plotly.figure_factory as ff
import re
from collections import Counter

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_term_network(term_data: Dict[str, Any], min_edge_weight: int = 2) -> Tuple[nx.Graph, Dict[str, Tuple[float, float]]]:
    """
    Create a network visualization of term co-occurrences.
    
    Args:
        term_data: Dictionary of term data from database
        min_edge_weight: Minimum co-occurrence count to include edge
        
    Returns:
        Tuple of (graph, positions)
    """
    try:
        # Create graph
        G = nx.Graph()
        
        # Add nodes with sizes based on term frequency
        term_counts = dict(term_data["terms"])
        max_count = max(term_counts.values()) if term_counts else 1
        
        for term, count in term_counts.items():
            # Normalize size
            size = 10 + (count / max_count) * 40
            G.add_node(term, size=size, count=count)
        
        # Add edges with weights
        for term1, term2, weight in term_data["co_occurrences"]:
            if weight >= min_edge_weight:
                G.add_edge(term1, term2, weight=weight)
        
        # Remove isolated nodes
        isolated_nodes = [node for node, degree in dict(G.degree()).items() if degree == 0]
        G.remove_nodes_from(isolated_nodes)
        
        # Limit to top nodes if graph is too large
        if len(G) > 50:
            # Keep top nodes by degree centrality
            centrality = nx.degree_centrality(G)
            top_nodes = sorted(centrality, key=centrality.get, reverse=True)[:50]
            G = G.subgraph(top_nodes)
        
        # Calculate layout
        if len(G) > 0:
            pos = nx.spring_layout(G, k=0.3, iterations=50)
        else:
            pos = {}
        
        return G, pos
    except Exception as e:
        logger.error(f"Error creating term network: {str(e)}")
        return nx.Graph(), {}

def create_narrative_shift_chart(narrative_data: Dict[str, Any]) -> go.Figure:
    """
    Create a chart showing narrative shift over time.
    
    Args:
        narrative_data: Dictionary with timeline of context data
        
    Returns:
        Plotly figure
    """
    try:
        # Extract timeline
        timeline = narrative_data["timeline"]
        term = narrative_data["term"]
        
        if not timeline:
            # Return empty figure
            return go.Figure()
        
        # Convert to DataFrame
        df_list = []
        for point in timeline:
            timestamp = point["timestamp"]
            
            # Extract top context terms
            context_terms = point["context_terms"]
            top_context = sorted(context_terms.items(), key=lambda x: x[1], reverse=True)[:10]
            
            for context_term, count in top_context:
                df_list.append({
                    "timestamp": timestamp,
                    "context_term": context_term,
                    "count": count,
                    "sentiment": point["sentiment"]["positive"] - point["sentiment"]["negative"]
                })
        
        df = pd.DataFrame(df_list)
        
        # Only keep terms that appear multiple times
        term_counts = Counter(df["context_term"])
        frequent_terms = [term for term, count in term_counts.items() if count >= 2]
        df = df[df["context_term"].isin(frequent_terms)]
        
        # Create heatmap
        pivot_table = df.pivot_table(
            index="context_term",
            columns="timestamp",
            values="count",
            aggfunc="sum",
            fill_value=0
        )
        
        # Sort by first appearance
        first_appearance = {}
        for term in pivot_table.index:
            series = pivot_table.loc[term]
            first_appear = series.ne(0).idxmax()
            first_appearance[term] = first_appear
        
        sorted_terms = sorted(pivot_table.index, key=lambda x: first_appearance[x])
        pivot_table = pivot_table.reindex(sorted_terms)
        
        # Convert timestamps to strings for display
        if isinstance(pivot_table.columns[0], str):
            # Already strings
            time_labels = [t.split("T")[0] + " " + t.split("T")[1][:5] for t in pivot_table.columns]
        else:
            # Convert datetime objects
            time_labels = [t.strftime("%Y-%m-%d %H:%M") for t in pivot_table.columns]
        
        # Create sentiment data
        sentiment_data = {}
        for point in timeline:
            timestamp = point["timestamp"]
            sentiment = point["sentiment"]["positive"] - point["sentiment"]["negative"]
            sentiment_data[timestamp] = sentiment
        
        sentiment_values = [sentiment_data.get(col, 0) for col in pivot_table.columns]
        
        # Create heatmap
        fig = go.Figure()
        
        # Add heatmap for co-occurrence
        fig.add_trace(go.Heatmap(
            z=pivot_table.values,
            x=time_labels,
            y=pivot_table.index,
            colorscale="Blues",
            showscale=True,
            colorbar=dict(title="Co-occurrence"),
            hovertemplate="Term: %{y}<br>Time: %{x}<br>Count: %{z}<extra></extra>"
        ))
        
        # Add line for sentiment
        fig.add_trace(go.Scatter(
            x=time_labels,
            y=[-1] * len(time_labels),  # Position below heatmap
            mode="lines+markers",
            line=dict(color="rgba(255, 0, 0, 0.7)", width=3),
            marker=dict(
                size=10,
                color=sentiment_values,
                colorscale="RdBu",
                cmin=-1,
                cmax=1,
                showscale=True,
                colorbar=dict(title="Sentiment", x=1.1)
            ),
            name="Sentiment",
            hovertemplate="Time: %{x}<br>Sentiment: %{marker.color:.2f}<extra></extra>"
        ))
        
        # Update layout
        fig.update_layout(
            title=f"Narrative Context Evolution for '{term}'",
            xaxis=dict(title="Time"),
            yaxis=dict(title="Associated Terms"),
            height=600,
            margin=dict(t=50, r=150),
            hovermode="closest"
        )
        
        return fig
    except Exception as e:
        logger.error(f"Error creating narrative shift chart: {str(e)}")
        return go.Figure()

def create_media_bias_chart(source_data: List[Dict[str, Any]]) -> go.Figure:
    """
    Create a chart showing media bias in coverage.
    
    Args:
        source_data: List of source coverage dictionaries
        
    Returns:
        Plotly figure
    """
    try:
        if not source_data:
            return go.Figure()
        
        # Convert to DataFrame
        df = pd.DataFrame(source_data)
        
        # Create scatter plot
        fig = px.scatter(
            df,
            x="sentiment",
            y="count",
            size="count",
            color="source",
            text="source",
            title="Source Coverage Bias Analysis",
            labels={
                "sentiment": "Sentiment (Negative ← → Positive)",
                "count": "Coverage Volume"
            },
            height=500
        )
        
        # Add vertical reference line at neutral sentiment
        fig.add_vline(x=0, line_dash="dash", line_color="gray")
        
        # Update layout
        fig.update_layout(
            xaxis=dict(
                range=[-1, 1],
                tickvals=[-0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75],
                ticktext=["Very Negative", "Negative", "Somewhat Negative", 
                          "Neutral", 
                          "Somewhat Positive", "Positive", "Very Positive"]
            ),
            showlegend=False
        )
        
        # Update traces to show source names
        fig.update_traces(
            textposition="top center",
            hovertemplate="<b>%{text}</b><br>Articles: %{y}<br>Sentiment: %{x:.2f}<extra></extra>"
        )
        
        return fig
    except Exception as e:
        logger.error(f"Error creating media bias chart: {str(e)}")
        return go.Figure()

def create_blindspot_visualization(blind_spots: List[Tuple[str, float, float, str]]) -> go.Figure:
    """
    Create a visualization of blind spots in coverage.
    
    Args:
        blind_spots: List of (topic, importance, coverage, explanation) tuples
        
    Returns:
        Plotly figure
    """
    try:
        if not blind_spots:
            return go.Figure()
        
        # Create dataframe
        df = pd.DataFrame([
            {
                "topic": topic,
                "importance": importance,
                "coverage": coverage,
                "gap": importance - coverage,
                "explanation": explanation
            }
            for topic, importance, coverage, explanation in blind_spots
        ])
        
        # Create quadrant chart
        fig = px.scatter(
            df,
            x="importance",
            y="coverage",
            size="gap",
            color="gap",
            color_continuous_scale="Reds",
            hover_name="topic",
            text="topic",
            title="Coverage Blind Spots",
            labels={
                "importance": "Topic Importance",
                "coverage": "Media Coverage"
            },
            height=600
        )
        
        # Add reference lines
        mid_importance = 0.5
        mid_coverage = 0.5
        
        fig.add_vline(x=mid_importance, line_dash="dash", line_color="gray")
        fig.add_hline(y=mid_coverage, line_dash="dash", line_color="gray")
        
        # Add annotations for quadrants
        fig.add_annotation(
            x=0.25, y=0.75,
            text="Low Importance,<br>High Coverage",
            showarrow=False,
            font=dict(size=12, color="gray")
        )
        
        fig.add_annotation(
            x=0.75, y=0.75,
            text="High Importance,<br>High Coverage",
            showarrow=False,
            font=dict(size=12, color="gray")
        )
        
        fig.add_annotation(
            x=0.25, y=0.25,
            text="Low Importance,<br>Low Coverage",
            showarrow=False,
            font=dict(size=12, color="gray")
        )
        
        fig.add_annotation(
            x=0.75, y=0.25,
            text="BLIND SPOT:<br>High Importance,<br>Low Coverage",
            showarrow=False,
            font=dict(size=14, color="red")
        )
        
        # Update traces
        fig.update_traces(
            textposition="top center",
            hovertemplate="<b>%{hovertext}</b><br>Importance: %{x:.2f}<br>Coverage: %{y:.2f}<br>Gap: %{marker.color:.2f}<extra></extra>"
        )
        
        # Update layout
        fig.update_layout(
            xaxis=dict(range=[0, 1]),
            yaxis=dict(range=[0, 1])
        )
        
        return fig
    except Exception as e:
        logger.error(f"Error creating blind spot visualization: {str(e)}")
        return go.Figure()
