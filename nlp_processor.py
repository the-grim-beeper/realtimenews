import spacy
import pandas as pd
import numpy as np
from collections import Counter, defaultdict
from typing import Dict, List, Tuple, Any, Set, Optional
import logging
from datetime import datetime
import networkx as nx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import random
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class NLPProcessor:
    """Handles NLP processing of news articles."""
    
    def __init__(self):
        """Initialize NLP components."""
        # Create directories for model caching
        os.makedirs("data/processed", exist_ok=True)
        
        logger.info("Loading NLP models...")
        
        # Load spaCy model for entity recognition and basic NLP
        try:
            self.nlp = spacy.load("en_core_web_sm")
            logger.info("Loaded spaCy model")
        except Exception as e:
            logger.error(f"Error loading spaCy model: {str(e)}")
            # Fallback to smaller model
            try:
                logger.info("Trying to download spaCy model...")
                spacy.cli.download("en_core_web_sm")
                self.nlp = spacy.load("en_core_web_sm")
            except:
                raise RuntimeError("Failed to load spaCy model. Please install it manually with 'python -m spacy download en_core_web_sm'")
        
        # Load sentence transformer for semantic analysis
        try:
            self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("Loaded sentence transformer model")
        except Exception as e:
            logger.error(f"Error loading sentence transformer: {str(e)}")
            self.sentence_model = None
        
        # Initialize sentiment analysis
        try:
            self.sentiment_analyzer = pipeline("sentiment-analysis")
            logger.info("Loaded sentiment analysis model")
        except Exception as e:
            logger.error(f"Error loading sentiment analyzer: {str(e)}")
            self.sentiment_analyzer = None
        
        # List of stop words and categories to ignore
        self.stop_entities = {"CARDINAL", "ORDINAL", "DATE", "TIME", "PERCENT", "MONEY", "QUANTITY"}
        
        # Create TF-IDF vectorizer for relevant term extraction
        self.vectorizer = TfidfVectorizer(
            max_df=0.85,
            min_df=2,
            max_features=5000,
            ngram_range=(1, 2),
            stop_words='english'
        )
        
        # Cache for term embeddings
        self.term_embeddings = {}
        
        # Knowledge graph for entity relationships
        self.knowledge_graph = nx.DiGraph()
        
        logger.info("NLP processor initialized")
    
    def process_article(self, article: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a single article to extract terms, entities, sentiment, etc.
        
        Args:
            article: Article dictionary with content
            
        Returns:
            Dictionary of extracted terms and metadata
        """
        # Combine title and content for processing
        text = f"{article['title']}. {article['content']}"
        
        # Process with spaCy
        doc = self.nlp(text)
        
        # Extract named entities
        entities = []
        for ent in doc.ents:
            if ent.label_ not in self.stop_entities:
                entities.append({
                    "text": ent.text,
                    "label": ent.label_,
                    "start": ent.start_char,
                    "end": ent.end_char
                })
        
        # Extract key noun phrases (terms)
        noun_chunks = []
        for chunk in doc.noun_chunks:
            if len(chunk.text.split()) <= 4:  # Limit to reasonable length
                noun_chunks.append(chunk.text.lower())
        
        # Extract keywords using TF-IDF on content
        keywords = self._extract_keywords(text)
        
        # Calculate article sentiment
        sentiment = self._get_sentiment(text)
        
        # Identify relationships between entities
        entity_relationships = self._extract_entity_relationships(doc)
        
        # Update knowledge graph
        self._update_knowledge_graph(entity_relationships)
        
        # Extract key sentences
        key_sentences = self._extract_key_sentences(doc)
        
        # Create result dictionary
        result = {
            "article_id": article["id"],
            "entities": entities,
            "noun_chunks": noun_chunks,
            "keywords": keywords,
            "sentiment": sentiment,
            "entity_relationships": entity_relationships,
            "key_sentences": key_sentences,
            "timestamp": article["published"],
            "source": article["source"]
        }
        
        return result
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract keywords using TF-IDF."""
        # Simple single-document case for individual articles
        try:
            # Create a single-document corpus
            corpus = [text]
            
            # Fit and transform the vectorizer
            tfidf_matrix = self.vectorizer.fit_transform(corpus)
            
            # Get feature names (terms)
            feature_names = self.vectorizer.get_feature_names_out()
            
            # Get the scores for the first document
            scores = tfidf_matrix.toarray()[0]
            
            # Pair feature names with their scores
            feature_scores = list(zip(feature_names, scores))
            
            # Sort by score in descending order
            feature_scores.sort(key=lambda x: x[1], reverse=True)
            
            # Return top keywords
            return [feature for feature, score in feature_scores[:25]]
        except Exception as e:
            logger.warning(f"Error extracting keywords: {str(e)}")
            return []
    
    def _get_sentiment(self, text: str) -> Dict[str, float]:
        """
        Get sentiment scores for text.
        
        Returns:
            Dictionary with sentiment scores
        """
        if self.sentiment_analyzer is None:
            return {"positive": 0.5, "negative": 0.5}
        
        try:
            # Truncate text if too long for model
            max_length = 512
            if len(text) > max_length:
                text = text[:max_length]
            
            # Get sentiment
            result = self.sentiment_analyzer(text)[0]
            
            # Convert to standardized format
            sentiment = {
                "positive": result["score"] if result["label"] == "POSITIVE" else 1 - result["score"],
                "negative": result["score"] if result["label"] == "NEGATIVE" else 1 - result["score"]
            }
            
            return sentiment
        except Exception as e:
            logger.warning(f"Error calculating sentiment: {str(e)}")
            return {"positive": 0.5, "negative": 0.5}
    
    def _extract_entity_relationships(self, doc) -> List[Dict[str, Any]]:
        """Extract relationships between entities in document."""
        relationships = []
        
        # Group entities by sentence
        sentences = list(doc.sents)
        for sent in sentences:
            sent_entities = [ent for ent in sent.ents if ent.label_ not in self.stop_entities]
            
            # If multiple entities in sentence, assume relationship
            if len(sent_entities) >= 2:
                for i, entity1 in enumerate(sent_entities):
                    for entity2 in sent_entities[i+1:]:
                        # Create relationship
                        relationship = {
                            "source": entity1.text,
                            "source_type": entity1.label_,
                            "target": entity2.text,
                            "target_type": entity2.label_,
                            "sentence": sent.text
                        }
                        relationships.append(relationship)
        
        return relationships
    
    def _update_knowledge_graph(self, relationships: List[Dict[str, Any]]) -> None:
        """Update knowledge graph with new entity relationships."""
        for rel in relationships:
            # Add nodes if they don't exist
            if not self.knowledge_graph.has_node(rel["source"]):
                self.knowledge_graph.add_node(rel["source"], type=rel["source_type"])
            
            if not self.knowledge_graph.has_node(rel["target"]):
                self.knowledge_graph.add_node(rel["target"], type=rel["target_type"])
            
            # Add or update edge
            if self.knowledge_graph.has_edge(rel["source"], rel["target"]):
                # Increment weight if relationship already exists
                self.knowledge_graph[rel["source"]][rel["target"]]["weight"] += 1
                # Add this sentence as evidence
                self.knowledge_graph[rel["source"]][rel["target"]]["sentences"].append(rel["sentence"])
            else:
                # Create new relationship
                self.knowledge_graph.add_edge(
                    rel["source"], 
                    rel["target"], 
                    weight=1, 
                    sentences=[rel["sentence"]]
                )
    
    def _extract_key_sentences(self, doc) -> List[str]:
        """Extract key sentences from document."""
        # Simple extraction based on entity density
        sentences = []
        for sent in doc.sents:
            # Count entities in sentence
            entity_count = len([ent for ent in sent.ents if ent.label_ not in self.stop_entities])
            
            # If sentence has multiple entities, consider it important
            if entity_count >= 2 and len(sent.text.split()) > 5:
                sentences.append(sent.text)
        
        # Return top sentences
        return sentences[:5]
    
    def get_surprising_connections(self, term_data: Dict[str, Any]) -> List[Tuple[str, str, float, str]]:
        """
        Identify surprising connections between terms.
        
        Args:
            term_data: Dictionary of term data from database
            
        Returns:
            List of (term1, term2, surprise_score, explanation) tuples
        """
        if not term_data:
            return []
        
        surprises = []
        
        # Extract co-occurrence network from term data
        G = nx.Graph()
        
        # Add all terms as nodes
        for term in term_data["terms"]:
            G.add_node(term)
        
        # Add edges for co-occurrences
        for term1, term2, weight in term_data["co_occurrences"]:
            G.add_edge(term1, term2, weight=weight)
        
        # Calculate semantic similarity between terms if we have embeddings
        if self.sentence_model is not None:
            term_pairs = []
            
            # Get all connected pairs
            for term1, term2 in G.edges():
                # Calculate similarity if we don't have it cached
                if (term1, term2) not in self.term_embeddings:
                    term_pairs.append((term1, term2))
                    
            if term_pairs:
                # Calculate embeddings for terms
                terms = list(set([t for pair in term_pairs for t in pair]))
                embeddings = self.sentence_model.encode(terms)
                
                # Create lookup dictionary
                term_to_embedding = {term: emb for term, emb in zip(terms, embeddings)}
                
                # Calculate similarities
                for term1, term2 in term_pairs:
                    if term1 in term_to_embedding and term2 in term_to_embedding:
                        emb1 = term_to_embedding[term1]
                        emb2 = term_to_embedding[term2]
                        similarity = cosine_similarity([emb1], [emb2])[0][0]
                        
                        # Cache the similarity
                        self.term_embeddings[(term1, term2)] = similarity
                        self.term_embeddings[(term2, term1)] = similarity
        
        # Identify surprising connections
        for term1, term2 in G.edges():
            # Get connection weight (co-occurrence frequency)
            weight = G[term1][term2]["weight"]
            
            # Get semantic similarity if available
            similarity = self.term_embeddings.get((term1, term2), 0.5)
            
            # Calculate surprise score
            # High weight but low similarity = surprising connection
            surprise_score = weight * (1 - similarity)
            
            if surprise_score > 1.0:  # Threshold for "surprising"
                # Generate explanation
                explanation = f"These terms co-occur frequently ({weight} times) despite having low semantic similarity. This suggests an unexpected relationship between concepts that aren't typically associated."
                
                # Add to results
                surprises.append((term1, term2, surprise_score, explanation))
        
        # Sort by surprise score
        surprises.sort(key=lambda x: x[2], reverse=True)
        
        return surprises[:10]  # Return top 10 surprising connections
    
    def detect_narrative_shifts(self, narrative_data: Dict[str, Any]) -> List[Tuple[datetime, float, str]]:
        """
        Detect shifts in narrative around a term over time.
        
        Args:
            narrative_data: Time series data about term context
            
        Returns:
            List of (timestamp, shift_score, description) tuples
        """
        if not narrative_data or "timeline" not in narrative_data:
            return []
        
        shifts = []
        timeline = narrative_data["timeline"]
        
        # Need at least 3 data points to detect shifts
        if len(timeline) < 3:
            return []
        
        # Sort timeline by timestamp
        timeline.sort(key=lambda x: x["timestamp"])
        
        # Create windows to compare
        window_size = min(3, len(timeline) // 3)
        if window_size < 1:
            window_size = 1
        
        for i in range(window_size, len(timeline) - window_size + 1):
            # Get before and after windows
            before_window = timeline[i-window_size:i]
            current_point = timeline[i]
            after_window = timeline[i+1:i+window_size+1]
            
            # Calculate sentiment change
            before_sentiment = sum([point["sentiment"]["positive"] for point in before_window]) / window_size
            after_sentiment = sum([point["sentiment"]["positive"] for point in after_window]) / window_size
            sentiment_change = abs(after_sentiment - before_sentiment)
            
            # Calculate context term change
            before_context = Counter()
            for point in before_window:
                for term, count in point["context_terms"].items():
                    before_context[term] += count
            
            after_context = Counter()
            for point in after_window:
                for term, count in point["context_terms"].items():
                    after_context[term] += count
            
            # Calculate Jaccard distance between context term sets
            before_terms = set(before_context.keys())
            after_terms = set(after_context.keys())
            
            if before_terms or after_terms:  # Avoid division by zero
                jaccard = 1 - len(before_terms & after_terms) / len(before_terms | after_terms)
            else:
                jaccard = 0
            
            # Combined shift score
            shift_score = (sentiment_change * 0.4) + (jaccard * 0.6)
            
            # If significant shift detected
            if shift_score > 0.2:  # Threshold for "significant"
                # Generate description
                if sentiment_change > 0.15:
                    sentiment_direction = "positive" if after_sentiment > before_sentiment else "negative"
                    sentiment_desc = f"The sentiment around this term shifted to more {sentiment_direction}."
                else:
                    sentiment_desc = "The sentiment remained relatively stable."
                
                # Identify new context terms
                new_terms = after_terms - before_terms
                disappeared_terms = before_terms - after_terms
                
                context_desc = "The narrative context changed: "
                if new_terms:
                    context_desc += f"New associations with {', '.join(list(new_terms)[:5])}. "
                if disappeared_terms:
                    context_desc += f"No longer associated with {', '.join(list(disappeared_terms)[:5])}."
                
                description = f"{sentiment_desc} {context_desc}"
                
                # Add to results
                shifts.append((current_point["timestamp"], shift_score, description))
        
        # Sort by shift score
        shifts.sort(key=lambda x: x[1], reverse=True)
        
        return shifts[:5]  # Return top 5 shifts
    
    def detect_blind_spots(self, term_data: Dict[str, Any]) -> List[Tuple[str, float, float, str]]:
        """
        Detect potential blind spots - important topics with less coverage than expected.
        
        Args:
            term_data: Dictionary of term data from database
            
        Returns:
            List of (topic, importance_score, coverage_score, explanation) tuples
        """
        if not term_data or "terms" not in term_data:
            return []
        
        blind_spots = []
        
        # Calculate expected coverage based on relationships in knowledge graph
        if len(self.knowledge_graph) < 10:
            # Not enough data in knowledge graph yet
            return []
        
        # Get betweenness centrality as proxy for term importance
        try:
            centrality = nx.betweenness_centrality(self.knowledge_graph)
        except:
            # Fall back to degree centrality if graph is disconnected
            centrality = nx.degree_centrality(self.knowledge_graph)
        
        # Get actual coverage from term data
        coverage = {term: count for term, count in term_data["terms"]}
        
        # Normalize centrality and coverage to 0-1 range
        if centrality:
            max_centrality = max(centrality.values())
            if max_centrality > 0:
                centrality = {k: v / max_centrality for k, v in centrality.items()}
        
        if coverage:
            max_coverage = max(coverage.values())
            if max_coverage > 0:
                coverage = {k: v / max_coverage for k, v in coverage.items()}
        
        # Find terms with high centrality but low coverage
        for term, importance in centrality.items():
            if importance > 0.3:  # Focus on somewhat important terms
                # Get coverage (default to 0 if not covered)
                term_coverage = coverage.get(term, 0)
                
                # Calculate blind spot score
                blind_spot_score = importance - term_coverage
                
                if blind_spot_score > 0.4:  # Threshold for "significant" blind spot
                    # Generate explanation
                    explanation = f"This topic is significant based on its connections to other covered topics, but is receiving less direct coverage. It is connected to {len(list(self.knowledge_graph.neighbors(term)))} other topics in the news."
                    
                    # Add to results
                    blind_spots.append((term, importance, term_coverage, explanation))
        
        # Sort by blind spot score (importance - coverage)
        blind_spots.sort(key=lambda x: x[1] - x[2], reverse=True)
        
        return blind_spots[:10]  # Return top 10 blind spots
