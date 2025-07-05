#!/usr/bin/env python3
"""
Streamlit Frontend for MOSDAC Knowledge Graph Chat
Bharatiya Antariksh Hackathon 2025 - Problem Statement 2 - Phase 2

A beautiful chat interface that connects users to the MOSDAC Knowledge Graph
through Neo4j and Gemini LLM integration.
"""

import streamlit as st
import json
import pandas as pd
from datetime import datetime
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, List, Any
import random

# Import our query agent
from query_agent import MOSDACQueryAgent

# Configure Streamlit page
st.set_page_config(
    page_title="MOSDAC Knowledge Graph Chat",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    
    .chat-message {
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 4px solid #2a5298;
    }
    
    .user-message {
        background-color: #f0f2f6;
        border-left-color: #ff6b6b;
    }
    
    .assistant-message {
        background-color: #e8f4fd;
        border-left-color: #4ecdc4;
    }
    
    .query-details {
        font-size: 0.8rem;
        color: #666;
        padding: 0.5rem;
        background-color: #f8f9fa;
        border-radius: 5px;
        margin-top: 0.5rem;
    }
    
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
    }
    
    .sidebar-section {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

class StreamlitMOSDACApp:
    """
    Main Streamlit application class
    """
    
    def __init__(self):
        self.agent = None
        self.initialize_session_state()
    
    def initialize_session_state(self):
        """Initialize Streamlit session state variables"""
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []
        
        if 'query_count' not in st.session_state:
            st.session_state.query_count = 0
        
        if 'agent_initialized' not in st.session_state:
            st.session_state.agent_initialized = False
        
        if 'show_cypher' not in st.session_state:
            st.session_state.show_cypher = False
        
        if 'conversation_stats' not in st.session_state:
            st.session_state.conversation_stats = {
                'neo4j_queries': 0,
                'gemini_queries': 0,
                'total_queries': 0
            }
    
    def setup_agent(self):
        """Initialize the query agent"""
        try:
            if not st.session_state.agent_initialized:
                with st.spinner("🚀 Initializing MOSDAC Query Agent..."):
                    self.agent = MOSDACQueryAgent()
                    st.session_state.agent_initialized = True
                    st.success("✅ Agent initialized successfully!")
            else:
                self.agent = MOSDACQueryAgent()
            return True
        except Exception as e:
            st.error(f"❌ Failed to initialize agent: {e}")
            st.error("Please check your credentials in the sidebar.")
            return False
    
    def render_header(self):
        """Render the main header"""
        st.markdown("""
        <div class="main-header">
            <h1>🛰️ MOSDAC Knowledge Graph Chat</h1>
            <p>Intelligent conversational interface for Meteorological & Oceanographic Satellite Data</p>
            <p><strong>Bharatiya Antariksh Hackathon 2025 - Problem Statement 2</strong></p>
        </div>
        """, unsafe_allow_html=True)
    
    def render_sidebar(self):
        """Render the sidebar with controls and information"""
        with st.sidebar:
            st.header("🔧 Configuration")
            
            # Credentials section
            with st.expander("🔐 API Credentials", expanded=False):
                st.text_input("Neo4j URI", value="bolt://xxxxx.databases.neo4j.io:7687", 
                             help="Your Neo4j Aura database URI")
                st.text_input("Neo4j Username", value="neo4j")
                st.text_input("Neo4j Password", type="password", 
                             help="Your Neo4j database password")
                st.text_input("Gemini API Key", type="password",
                             help="Get from https://aistudio.google.com/app/apikey")
            
            # Display options
            st.subheader("🎛️ Display Options")
            st.session_state.show_cypher = st.checkbox(
                "Show Cypher Queries", 
                value=st.session_state.show_cypher,
                help="Display the generated Cypher queries"
            )
            
            show_confidence = st.checkbox(
                "Show Confidence Scores", 
                value=True,
                help="Display confidence scores for responses"
            )
            
            # Sample questions
            st.subheader("💡 Sample Questions")
            sample_questions = [
                "Which satellite monitors tropical rainfall?",
                "What sensors are on INSAT-3D?",
                "List all products from SAPHIR sensor",
                "Tell me about Megha-Tropiques mission",
                "Which satellites observe ocean temperature?",
                "What products does MOSDAC provide?"
            ]
            
            for i, question in enumerate(sample_questions):
                if st.button(f"📝 {question[:30]}...", key=f"sample_{i}"):
                    st.session_state.current_query = question
            
            # Statistics
            st.subheader("📊 Session Statistics")
            stats = st.session_state.conversation_stats
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Total Queries", stats['total_queries'])
                st.metric("Neo4j Queries", stats['neo4j_queries'])
            with col2:
                st.metric("Gemini Queries", stats['gemini_queries'])
                success_rate = (stats['total_queries'] - 0) / max(1, stats['total_queries'])
                st.metric("Success Rate", f"{success_rate:.1%}")
            
            # Actions
            st.subheader("🔄 Actions")
            if st.button("🗑️ Clear Chat History"):
                st.session_state.chat_history = []
                st.session_state.conversation_stats = {
                    'neo4j_queries': 0,
                    'gemini_queries': 0,
                    'total_queries': 0
                }
                st.rerun()
            
            if st.button("💾 Export Conversation"):
                if self.agent and hasattr(self.agent, 'export_conversation'):
                    filename = self.agent.export_conversation()
                    st.success(f"Conversation exported to {filename}")
                else:
                    st.warning("Export not available.")
            
            # About section
            with st.expander("ℹ️ About", expanded=False):
                st.markdown("""
                **MOSDAC Knowledge Graph Chat** is an intelligent conversational interface 
                that combines:
                
                - 🗄️ **Neo4j Knowledge Graph** for structured satellite data
                - 🧠 **Gemini LLM** for natural language understanding
                - 🔍 **Smart Query Generation** from natural language to Cypher
                - 📊 **Real-time Visualization** of query results
                
                Built for **Bharatiya Antariksh Hackathon 2025** by Team [Your Team Name].
                """)
    
    def render_chat_interface(self):
        """Render the main chat interface"""
        # Chat container
        chat_container = st.container()
        
        with chat_container:
            # Display chat history
            for i, message in enumerate(st.session_state.chat_history):
                self.render_message(message, i)
        
        # Query input
        st.markdown("---")
        
        # Handle sample question selection
        query_input = st.session_state.get('current_query', '')
        if query_input:
            st.session_state.current_query = ''  # Clear after use
        
        # Text input for queries
        user_query = st.text_input(
            "💬 Ask me anything about satellites, sensors, missions, or meteorological data:",
            value=query_input,
            placeholder="e.g., Which satellite monitors tropical rainfall?",
            key="user_input"
        )
        
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            if st.button("🚀 Send Query", type="primary", use_container_width=True):
                if user_query.strip():
                    self.process_user_query(user_query)
                    st.rerun()
        
        with col2:
            if st.button("🎲 Random Question", use_container_width=True):
                samples = sample_questions
                if self.agent and hasattr(self.agent, 'get_sample_questions'):
                    samples = self.agent.get_sample_questions()
                random_question = random.choice(samples)
                st.session_state.current_query = random_question
                st.rerun()
        
        with col3:
            if st.button("🔄 Reset Agent", use_container_width=True):
                st.session_state.agent_initialized = False
                st.rerun()
    
    def render_message(self, message: Dict, index: int):
        """Render a single chat message"""
        timestamp = message.get('timestamp', datetime.now().isoformat())
        formatted_time = datetime.fromisoformat(timestamp).strftime("%H:%M:%S")
        
        # User message
        st.markdown(f"""
        <div class="chat-message user-message">
            <strong>🤔 You ({formatted_time}):</strong><br>
            {message['user_query']}
        </div>
        """, unsafe_allow_html=True)
        
        # Assistant response
        response = message.get('neo4j_response') or message.get('gemini_response', 'No response')
        source = message.get('source', 'unknown')
        confidence = message.get('confidence', 0.0)
        
        source_emoji = {
            'neo4j': '🗄️',
            'gemini': '🧠',
            'fallback': '⚠️'
        }.get(source, '🤖')
        
        st.markdown(f"""
        <div class="chat-message assistant-message">
            <strong>{source_emoji} Assistant ({source.title()}):</strong><br>
            {response}
        </div>
        """, unsafe_allow_html=True)
        
        # Query details (if enabled)
        if st.session_state.show_cypher and message.get('cypher_query'):
            with st.expander(f"🔍 Query Details #{index + 1}", expanded=False):
                st.code(message['cypher_query'], language='cypher')
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Source", source.title())
                with col2:
                    st.metric("Confidence", f"{confidence:.2f}")
                with col3:
                    st.metric("Query Type", message.get('query_type', 'N/A'))
    
    def process_user_query(self, user_query: str):
        """Process user query through the agent"""
        if not self.agent:
            st.error("Agent not initialized. Please check credentials.")
            return
        
        try:
            # Process query
            with st.spinner("🤖 Processing your query..."):
                response_data = self.agent.process_query(user_query)
            
            # Add to chat history
            st.session_state.chat_history.append(response_data)
            
            # Update statistics
            st.session_state.conversation_stats['total_queries'] += 1
            if response_data['source'] == 'neo4j':
                st.session_state.conversation_stats['neo4j_queries'] += 1
            elif response_data['source'] == 'gemini':
                st.session_state.conversation_stats['gemini_queries'] += 1
            
        except Exception as e:
            st.error(f"❌ Error processing query: {e}")
    
    def render_analytics_tab(self):
        """Render analytics and visualization tab"""
        st.header("📊 Analytics & Insights")
        
        if not st.session_state.chat_history:
            st.info("💬 Start chatting to see analytics!")
            return
        
        # Query source distribution
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📈 Query Source Distribution")
            source_counts = {}
            for message in st.session_state.chat_history:
                source = message.get('source', 'unknown')
                source_counts[source] = source_counts.get(source, 0) + 1
            
            if source_counts:
                fig = px.pie(
                    values=list(source_counts.values()),
                    names=list(source_counts.keys()),
                    title="Query Source Distribution"
                )
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("📊 Confidence Score Distribution")