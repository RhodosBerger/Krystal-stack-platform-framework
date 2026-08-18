"""
LLM Connector for CNC Copilot
Integrates Large Language Models throughout the system

Capabilities:
- Connect unrelated data across logging sessions
- Infer practical relationships
- Generate manufacturing insights
- Natural language interface to system
"""

import os
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass
from datetime import datetime
import json

# LLM Frameworks
from langchain.llms import OpenAI, Anthropic
from langchain.chat_models import ChatOpenAI, ChatAnthropic
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory, ConversationSummaryMemory
from langchain.vectorstores import Chroma, FAISS
from langchain.schema import Document

# Local LLM support
try:
    from llama_cpp import Llama
    LOCAL_LLM_AVAILABLE = True
except ImportError:
    LOCAL_LLM_AVAILABLE = False


@dataclass
class LLMConfig:
    """
    LLM Configuration
    
    Supports multiple providers:
    - OpenAI (GPT-4, GPT-3.5)
    - Anthropic (Claude)
    - Local (LLaMA, Mistral)
    """
    provider: str = 'openai'  # 'openai', 'anthropic', 'local'
    model: str = 'gpt-4'
    temperature: float = 0.7
    max_tokens: int = 2000
    use_embeddings: bool = True
    embedding_model: str = 'text-embedding-ada-002'
    local_model_path: Optional[str] = None


class LLMConnector:
    """
    Universal LLM Connector
    
    Paradigm: LLM as Universal Translator
    - Translates data → insights
    - Translates questions → queries
    - Translates patterns → recommendations
    """
    
    def __init__(self, config: Optional[LLMConfig] = None):
        """
        Initialize LLM connector
        
        Args:
            config: LLM configuration
        """
        self.config = config or LLMConfig()
        self.llm = self._initialize_llm()
        self.embeddings = self._initialize_embeddings()
        self.memory = ConversationBufferMemory(return_messages=True)
        
        # Vector store for cross-session memory
        self.vector_store = None
        self._initialize_vector_store()
    
    def _initialize_llm(self):
        """Initialize LLM based on provider"""
        if self.config.provider == 'openai':
            return ChatOpenAI(
                model_name=self.config.model,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                openai_api_key=os.getenv('OPENAI_API_KEY')
            )
        
        elif self.config.provider == 'anthropic':
            return ChatAnthropic(
                model=self.config.model,
                temperature=self.config.temperature,
                max_tokens_to_sample=self.config.max_tokens,
                anthropic_api_key=os.getenv('ANTHROPIC_API_KEY')
            )
        
        elif self.config.provider == 'local':
            if not LOCAL_LLM_AVAILABLE:
                raise ImportError("llama-cpp-python not installed")
            
            if not self.config.local_model_path:
                raise ValueError("local_model_path required for local LLM")
            
            return Llama(
                model_path=self.config.local_model_path,
                n_ctx=4096,
                n_threads=8
            )
        
        else:
            raise ValueError(f"Unknown provider: {self.config.provider}")
    
    def _initialize_embeddings(self):
        """Initialize embedding model"""
        if not self.config.use_embeddings:
            return None
        
        if self.config.provider == 'openai':
            return OpenAIEmbeddings(
                model=self.config.embedding_model,
                openai_api_key=os.getenv('OPENAI_API_KEY')
            )
        else:
            # Use local embeddings for privacy
            return HuggingFaceEmbeddings(
                model_name='sentence-transformers/all-MiniLM-L6-v2'
            )
    
    def _initialize_vector_store(self):
        """
        Initialize vector store for cross-session memory
        
        This stores ALL manufacturing data as embeddings
        Allows semantic search across sessions
        """
        if not self.embeddings:
            return
        
        persist_directory = "data/vector_store"
        
        # Try to load existing store
        try:
            self.vector_store = Chroma(
                persist_directory=persist_directory,
                embedding_function=self.embeddings
            )
        except:
            # Create new store
            self.vector_store = Chroma(
                persist_directory=persist_directory,
                embedding_function=self.embeddings
            )
    
    def infer_relationship(self, 
                          data_a: Dict[str, Any],
                          data_b: Dict[str, Any],
                          context: Optional[str] = None) -> Dict[str, Any]:
        """
        Infer practical relationship between two unrelated data points
        
        Example:
        - data_a: Tool wear from 3 months ago
        - data_b: Current vibration pattern
        - Inference: Similar pattern indicates similar wear progression
        
        Args:
            data_a: First data point
            data_b: Second data point
            context: Additional context
        
        Returns:
            {
                'has_relationship': bool,
                'relationship_type': str,
                'confidence': float,
                'explanation': str,
                'practical_action': str
            }
        """
        prompt = ChatPromptTemplate.from_template("""
You are an expert manufacturing engineer analyzing CNC machining data.

Given two data points from different time periods, determine if there's a practical relationship.

Data Point A (from {time_a}):
{data_a}

Data Point B (from {time_b}):
{data_b}

{context}

Analyze if these data points are related and if that relationship has practical manufacturing value.

Provide your analysis in JSON format:
{{
    "has_relationship": true/false,
    "relationship_type": "causal/correlational/sequential/pattern_match/none",
    "confidence": 0.0-1.0,
    "explanation": "detailed explanation",
    "practical_action": "specific recommended action or 'none'"
}}
""")
        
        time_a = data_a.get('timestamp', 'unknown time')
        time_b = data_b.get('timestamp', 'unknown time')
        
        # Format data for LLM
        data_a_str = self._format_data_for_llm(data_a)
        data_b_str = self._format_data_for_llm(data_b)
        
        # Generate inference
        chain = LLMChain(llm=self.llm, prompt=prompt)
        result = chain.run(
            time_a=time_a,
            data_a=data_a_str,
            time_b=time_b,
            data_b=data_b_str,
            context=context or "No additional context provided."
        )
        
        # Parse JSON response
        try:
            inference = json.loads(result)
        except json.JSONDecodeError:
            # Fallback if LLM doesn't return valid JSON
            inference = {
                'has_relationship': False,
                'relationship_type': 'none',
                'confidence': 0.0,
                'explanation': result,
                'practical_action': 'none'
            }
        
        return inference
    
    def find_similar_sessions(self, 
                             current_data: Dict[str, Any],
                             top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Find similar sessions from history using semantic search
        
        This connects current session to past sessions with similar patterns
        
        Args:
            current_data: Current session data
            top_k: Number of similar sessions to return
        
        Returns:
            List of similar sessions with similarity scores
        """
        if not self.vector_store:
            return []
        
        # Create query from current data
        query = self._format_data_for_llm(current_data)
        
        # Semantic search
        similar_docs = self.vector_store.similarity_search_with_score(
            query,
            k=top_k
        )
        
        results = []
        for doc, score in similar_docs:
            results.append({
                'session_data': json.loads(doc.page_content),
                'similarity_score': float(score),
                'metadata': doc.metadata
            })
        
        return results
    
    def connect_unrelated_data(self,
                               data_points: List[Dict[str, Any]],
                               purpose: str = 'manufacturing_optimization') -> Dict[str, Any]:
        """
        Connect multiple unrelated data points and find practical insights
        
        This is the CORE intelligence function
        
        Example:
        - Machine vibration from Session A
        - Tool wear from Session B
        - Part quality from Session C
        - Weather data (temperature) from external source
        
        LLM finds: "All sessions with high ambient temperature show 15% faster tool wear"
        
        Args:
            data_points: List of unrelated data points
            purpose: What we're trying to optimize
        
        Returns:
            Insights and recommendations
        """
        prompt = ChatPromptTemplate.from_template("""
You are analyzing manufacturing data to find hidden connections and practical insights.

Purpose: {purpose}

Data Points ({count} total):
{data_points}

Your task:
1. Find non-obvious connections between these data points
2. Identify patterns that have practical manufacturing value
3. Provide specific, actionable recommendations

Focus on practical insights that can:
- Reduce costs
- Improve quality
- Prevent failures
- Optimize processes

Provide your analysis as:
{{
    "connections_found": [
        {{
            "data_points_involved": ["point1", "point2", ...],
            "connection_type": "causal/correlational/temporal",
            "strength": 0.0-1.0,
            "description": "explanation"
        }}
    ],
    "insights": [
        "insight 1",
        "insight 2"
    ],
    "recommendations": [
        {{
            "priority": "high/medium/low",
            "action": "specific action",
            "expected_benefit": "quantified benefit",
            "implementation": "how to implement"
        }}
    ]
}}
""")
        
        # Format all data points
        data_str = ""
        for i, dp in enumerate(data_points):
            data_str += f"\nData Point {i+1}:\n"
            data_str += self._format_data_for_llm(dp)
            data_str += "\n" + "-" * 50 + "\n"
        
        # Generate connections
        chain = LLMChain(llm=self.llm, prompt=prompt)
        result = chain.run(
            purpose=purpose,
            count=len(data_points),
            data_points=data_str
        )
        
        # Parse results
        try:
            insights = json.loads(result)
        except json.JSONDecodeError:
            insights = {
                'connections_found': [],
                'insights': [result],
                'recommendations': []
            }
        
        return insights
    
    def store_session_data(self, 
                          session_data: Dict[str, Any],
                          metadata: Optional[Dict[str, Any]] = None):
        """
        Store session data in vector store for future retrieval
        
        This builds up knowledge across sessions
        
        Args:
            session_data: Data to store
            metadata: Optional metadata (timestamp, machine_id, etc.)
        """
        if not self.vector_store:
            return
        
        # Format data as text for embedding
        text = self._format_data_for_llm(session_data)
        
        # Create document
        doc = Document(
            page_content=json.dumps(session_data),
            metadata=metadata or {
                'timestamp': datetime.now().isoformat(),
                'session_id': session_data.get('session_id', 'unknown')
            }
        )
        
        # Add to vector store
        self.vector_store.add_documents([doc])
        self.vector_store.persist()
    
    def query_natural_language(self, query: str) -> str:
        """
        Answer questions about manufacturing data in natural language
        
        Example:
        Q: "Why did tool wear increase 20% last month?"
        A: LLM analyzes all data and provides explanation
        
        Args:
            query: Natural language question
        
        Returns:
            Natural language answer
        """
        # Retrieve relevant context from vector store
        if self.vector_store:
            relevant_docs = self.vector_store.similarity_search(query, k=5)
            context = "\n\n".join([doc.page_content for doc in relevant_docs])
        else:
            context = "No historical data available."
        
        prompt = ChatPromptTemplate.from_template("""
You are an expert manufacturing engineer with access to CNC machining data.

Context from manufacturing history:
{context}

User Question: {query}

Provide a clear, actionable answer based on the available data.
If the data doesn't support a definitive answer, say so and suggest what additional data would help.
""")
        
        chain = LLMChain(llm=self.llm, prompt=prompt)
        answer = chain.run(context=context, query=query)
        
        return answer
    
    def _format_data_for_llm(self, data: Dict[str, Any]) -> str:
        """Format data dictionary for LLM consumption"""
        formatted = []
        for key, value in data.items():
            if isinstance(value, (dict, list)):
                value_str = json.dumps(value, indent=2)
            else:
                value_str = str(value)
            formatted.append(f"  {key}: {value_str}")
        
        return "\n".join(formatted)


# Convenience functions
def create_llm_connector(provider: str = 'openai', 
                        model: str = 'gpt-4') -> LLMConnector:
    """
    Quick create LLM connector
    
    Args:
        provider: 'openai', 'anthropic', or 'local'
        model: Model name
    
    Returns:
        Configured LLMConnector
    """
    config = LLMConfig(provider=provider, model=model)
    return LLMConnector(config)


def infer_relationship_simple(data_a: Dict, data_b: Dict) -> bool:
    """
    Simple relationship check
    
    Returns:
        True if relationship found
    """
    connector = create_llm_connector()
    result = connector.infer_relationship(data_a, data_b)
    return result.get('has_relationship', False)
