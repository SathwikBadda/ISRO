#!/usr/bin/env python3
"""
Knowledge Graph Loader for Neo4j
Bharatiya Antariksh Hackathon 2025 - Problem Statement 2 - Phase 2

This script loads the extracted knowledge graph triples into Neo4j Aura Free Tier
Creates nodes for entities and relationships between them
"""

import os
import json
import logging
from pathlib import Path
from typing import List, Dict, Set, Tuple
from datetime import datetime

# Neo4j driver
from neo4j import GraphDatabase
import pandas as pd
from tqdm import tqdm

# Environment variables
from dotenv import load_dotenv
load_dotenv()

class Neo4jKGLoader:
    """
    Loads MOSDAC Knowledge Graph into Neo4j database
    """
    
    def __init__(self, uri: str = "", user: str = "", password: str = "", 
                 output_dir: str = "mosdac_data"):
        """
        Initialize Neo4j connection
        
        Args:
            uri: Neo4j database URI
            user: Database username  
            password: Database password
            output_dir: Directory containing KG data
        """
        self.output_dir = Path(output_dir)
        self.kg_dir = self.output_dir / "knowledge_graph"
        
        # Setup logging
        self.setup_logging()
        
        # Get Neo4j credentials from environment or parameters
        self.uri = uri or os.getenv('NEO4J_URI', 'bolt://localhost:7687')
        self.user = user or os.getenv('NEO4J_USER', 'neo4j')
        self.password = password or os.getenv('NEO4J_PASSWORD', 'password')
        
        # Initialize driver
        self.driver = None
        self.connect_to_neo4j()
        
        # Track loaded data
        self.loaded_entities = set()
        self.loaded_relationships = set()
        
        self.logger.info("Neo4j KG Loader initialized")
    
    def setup_logging(self):
        """Setup logging for KG loader"""
        log_file = self.output_dir / "logs" / "kg_loader.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        
        self.logger = logging.getLogger(__name__)
    
    def connect_to_neo4j(self):
        """Establish connection to Neo4j database"""
        try:
            self.driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))
            
            # Test connection
            if self.driver is None:
                raise RuntimeError("Neo4j driver is not initialized.")
            with self.driver.session() as session:
                result = session.run("RETURN 1")
                result.single()
            
            self.logger.info(f"Successfully connected to Neo4j at {self.uri}")
            
        except Exception as e:
            self.logger.error(f"Failed to connect to Neo4j: {e}")
            self.logger.error("Please check your Neo4j credentials and connection")
            raise
    
    def close_connection(self):
        """Close Neo4j connection"""
        if self.driver:
            self.driver.close()
            self.logger.info("Neo4j connection closed")
    
    def clear_database(self):
        """Clear all existing data in the database"""
        try:
            if self.driver is None:
                raise RuntimeError("Neo4j driver is not initialized.")
            with self.driver.session() as session:
                # Delete all relationships first
                session.run("MATCH ()-[r]-() DELETE r")
                # Delete all nodes
                session.run("MATCH (n) DELETE n")
                
                self.logger.info("Database cleared successfully")
                
        except Exception as e:
            self.logger.error(f"Error clearing database: {e}")
            raise
    
    def create_constraints_and_indexes(self):
        """Create constraints and indexes for better performance"""
        try:
            if self.driver is None:
                raise RuntimeError("Neo4j driver is not initialized.")
            with self.driver.session() as session:
                # Create constraints for unique entity names
                constraints = [
                    "CREATE CONSTRAINT entity_name IF NOT EXISTS FOR (e:Entity) REQUIRE e.name IS UNIQUE",
                    "CREATE CONSTRAINT satellite_name IF NOT EXISTS FOR (s:Satellite) REQUIRE s.name IS UNIQUE",
                    "CREATE CONSTRAINT sensor_name IF NOT EXISTS FOR (s:Sensor) REQUIRE s.name IS UNIQUE",
                    "CREATE CONSTRAINT mission_name IF NOT EXISTS FOR (m:Mission) REQUIRE m.name IS UNIQUE",
                    "CREATE CONSTRAINT organization_name IF NOT EXISTS FOR (o:Organization) REQUIRE o.name IS UNIQUE"
                ]
                
                for constraint in constraints:
                    try:
                        session.run(constraint)  # type: ignore
                        self.logger.info(f"Created constraint: {constraint.split()[2]}")
                    except Exception as e:
                        self.logger.warning(f"Constraint already exists or failed: {e}")
                
                # Create indexes for better query performance
                indexes = [
                    "CREATE INDEX entity_type IF NOT EXISTS FOR (e:Entity) ON (e.type)",
                    "CREATE INDEX entity_confidence IF NOT EXISTS FOR (e:Entity) ON (e.confidence)"
                    # Relationship property index removed due to Neo4j syntax/version issues
                    # "CREATE INDEX relationship_confidence IF NOT EXISTS FOR ()-[r]-() ON (r.confidence)"
                ]
                
                for index in indexes:
                    try:
                        session.run(index)  # type: ignore
                        self.logger.info(f"Created index: {index.split()[2]}")
                    except Exception as e:
                        self.logger.warning(f"Index already exists or failed: {e}")
                
        except Exception as e:
            self.logger.error(f"Error creating constraints and indexes: {e}")
    
    def load_entities(self, kg_data: Dict):
        """
        Load entities as nodes in Neo4j
        
        Args:
            kg_data: Knowledge graph data containing entities
        """
        self.logger.info("Loading entities into Neo4j")
        
        entities = kg_data.get('entities', [])
        
        with self.driver.session() as session:
            for entity in tqdm(entities, desc="Loading entities"):
                try:
                    # Determine entity type and create appropriate node
                    entity_name = entity['text']
                    entity_type = entity['label']
                    confidence = entity.get('confidence', 0.0)
                    sources = entity.get('sources', [])
                    
                    # Create base Entity node
                    cypher_query = """
                    MERGE (e:Entity {name: $name})
                    SET e.type = $type,
                        e.confidence = $confidence,
                        e.sources = $sources,
                        e.created_at = $created_at
                    """
                    
                    session.run(cypher_query, {
                        'name': entity_name,
                        'type': entity_type,
                        'confidence': confidence,
                        'sources': [str(source) for source in sources],
                        'created_at': datetime.now().isoformat()
                    })  # type: ignore
                    
                    # Create specific type node based on entity type
                    if entity_type in ['SATELLITE', 'ORGANIZATION', 'SENSOR', 'MISSION', 'LOCATION', 'PRODUCT']:
                        specific_query = f"""
                        MERGE (s:{entity_type.title()} {{name: $name}})
                        SET s.confidence = $confidence,
                            s.created_at = $created_at
                        WITH s
                        MATCH (e:Entity {{name: $name}})
                        MERGE (e)-[:IS_A]->(s)
                        """
                        
                        session.run(specific_query, {
                            'name': entity_name,
                            'confidence': confidence,
                            'created_at': datetime.now().isoformat()
                        })  # type: ignore
                    
                    self.loaded_entities.add(entity_name)
                    
                except Exception as e:
                    self.logger.error(f"Error loading entity {entity.get('text', 'unknown')}: {e}")
                    continue
        
        self.logger.info(f"Loaded {len(self.loaded_entities)} entities")
    
    def load_relationships(self, kg_data: Dict):
        """
        Load relationships between entities
        
        Args:
            kg_data: Knowledge graph data containing triples
        """
        self.logger.info("Loading relationships into Neo4j")
        
        triples = kg_data.get('triples', [])
        
        with self.driver.session() as session:
            for triple in tqdm(triples, desc="Loading relationships"):
                try:
                    head = triple['head']
                    relation = triple['relation'].upper().replace(' ', '_')
                    tail = triple['tail']
                    confidence = triple.get('confidence', 0.0)
                    source = triple.get('source', 'unknown')
                    source_url = triple.get('source_url', '')
                    document_type = triple.get('document_type', '')
                    section = triple.get('section', '')
                    
                    # Create relationship between entities
                    cypher_query = f"""
                    MATCH (h:Entity {{name: $head}})
                    MATCH (t:Entity {{name: $tail}})
                    MERGE (h)-[r:{relation}]->(t)
                    SET r.confidence = $confidence,
                        r.source = $source,
                        r.source_url = $source_url,
                        r.document_type = $document_type,
                        r.section = $section,
                        r.created_at = $created_at
                    """
                    
                    session.run(cypher_query, {
                        'head': head,
                        'tail': tail,
                        'confidence': confidence,
                        'source': source,
                        'source_url': source_url,
                        'document_type': document_type,
                        'section': section,
                        'created_at': datetime.now().isoformat()
                    })  # type: ignore
                    
                    self.loaded_relationships.add(f"{head}-{relation}->{tail}")
                    
                except Exception as e:
                    self.logger.error(f"Error loading relationship {triple}: {e}")
                    continue
        
        self.logger.info(f"Loaded {len(self.loaded_relationships)} relationships")
    
    def create_domain_specific_nodes(self):
        """Create domain-specific node types and relationships"""
        self.logger.info("Creating domain-specific structures")
        
        with self.driver.session() as session:
            # Create MOSDAC root node
            session.run("""
                MERGE (m:System {name: 'MOSDAC'})
                SET m.full_name = 'Meteorological & Oceanographic Satellite Data Archival Centre',
                    m.description = 'Data archival and distribution system for meteorological and oceanographic satellite data',
                    m.organization = 'ISRO',
                    m.created_at = $created_at
            """, {'created_at': datetime.now().isoformat()})
            
            # Connect all satellites to MOSDAC
            session.run("""
                MATCH (s:Satellite)
                MATCH (m:System {name: 'MOSDAC'})
                MERGE (s)-[:ARCHIVED_IN]->(m)
            """)
            
            # Create hierarchical relationships
            queries = [
                # Sensors belong to satellites
                """
                MATCH (sensor:Sensor)
                MATCH (satellite:Satellite)
                WHERE ANY(source IN sensor.sources WHERE source CONTAINS satellite.name)
                MERGE (sensor)-[:INSTALLED_ON]->(satellite)
                """,
                
                # Products generated by sensors
                """
                MATCH (product:Product)
                MATCH (sensor:Sensor)
                WHERE ANY(source IN product.sources WHERE source CONTAINS sensor.name)
                MERGE (sensor)-[:GENERATES]->(product)
                """,
                
                # Missions operate satellites
                """
                MATCH (mission:Mission)
                MATCH (satellite:Satellite)
                WHERE ANY(source IN mission.sources WHERE source CONTAINS satellite.name)
                MERGE (mission)-[:OPERATES]->(satellite)
                """
            ]
            
            for query in queries:
                try:
                    session.run(query)  # type: ignore
                except Exception as e:
                    self.logger.warning(f"Error creating domain relationship: {e}")
    
    def load_metadata_and_statistics(self, kg_data: Dict):
        """Load metadata and statistics as special nodes"""
        self.logger.info("Loading metadata and statistics")
        
        metadata = kg_data.get('metadata', {})
        statistics = kg_data.get('statistics', {})
        
        with self.driver.session() as session:
            # Create metadata node
            session.run("""
                CREATE (meta:Metadata {
                    name: 'MOSDAC_KG_Metadata',
                    created_at: $created_at,
                    total_documents: $total_documents,
                    total_entities: $total_entities,
                    total_triples: $total_triples,
                    entity_types: $entity_types,
                    relation_types: $relation_types
                })
            """, {
                'created_at': metadata.get('created_at', datetime.now().isoformat()),
                'total_documents': metadata.get('total_documents', 0),
                'total_entities': metadata.get('total_entities', 0),
                'total_triples': metadata.get('total_triples', 0),
                'entity_types': str(metadata.get('entity_types', {})),
                'relation_types': str(metadata.get('relation_types', {}))
            })
            
            # Create statistics nodes
            most_common_entities = statistics.get('most_common_entities', [])
            for i, entity_stat in enumerate(most_common_entities[:10]):
                session.run("""
                    CREATE (stat:EntityStatistic {
                        entity: $entity,
                        frequency: $frequency,
                        rank: $rank
                    })
                """, {
                    'entity': entity_stat.get('entity', ''),
                    'frequency': entity_stat.get('frequency', 0),
                    'rank': i + 1
                })
    
    def verify_loaded_data(self):
        """Verify that data was loaded correctly"""
        self.logger.info("Verifying loaded data")
        
        with self.driver.session() as session:
            # Count nodes
            result = session.run("MATCH (n) RETURN count(n) as node_count")
            record = result.single() if result else None
            node_count = record['node_count'] if record and 'node_count' in record else 0
            
            # Count relationships
            result = session.run("MATCH ()-[r]-() RETURN count(r) as rel_count")
            record = result.single() if result else None
            rel_count = record['rel_count'] if record and 'rel_count' in record else 0
            
            # Count by node types
            result = session.run("""
                MATCH (n)
                RETURN labels(n) as labels, count(n) as count
                ORDER BY count DESC
            """)
            
            node_types = {}
            for record in result:
                labels = record['labels']
                count = record['count']
                node_types[str(labels)] = count
            
            # Count by relationship types
            result = session.run("""
                MATCH ()-[r]->()
                RETURN type(r) as rel_type, count(r) as count
                ORDER BY count DESC
            """)
            
            rel_types = {}
            for record in result:
                rel_type = record['rel_type']
                count = record['count']
                rel_types[rel_type] = count
            
            verification_results = {
                'total_nodes': node_count,
                'total_relationships': rel_count,
                'node_types': node_types,
                'relationship_types': rel_types,
                'verified_at': datetime.now().isoformat()
            }
            
            # Save verification results
            verification_file = self.kg_dir / "neo4j_verification.json"
            with open(verification_file, 'w', encoding='utf-8') as f:
                json.dump(verification_results, f, ensure_ascii=False, indent=2)
            
            self.logger.info(f"Verification completed:")
            self.logger.info(f"  - Total nodes: {node_count}")
            self.logger.info(f"  - Total relationships: {rel_count}")
            self.logger.info(f"  - Node types: {len(node_types)}")
            self.logger.info(f"  - Relationship types: {len(rel_types)}")
            
            return verification_results
    
    def create_sample_queries(self):
        """Create and save sample Cypher queries for testing"""
        sample_queries = {
            "find_all_satellites": "MATCH (s:Satellite) RETURN s.name, s.confidence LIMIT 10",
            
            "find_satellite_sensors": """
                MATCH (satellite:Satellite)<-[:INSTALLED_ON]-(sensor:Sensor)
                RETURN satellite.name, collect(sensor.name) as sensors
                LIMIT 5
            """,
            
            "find_monitoring_relationships": """
                MATCH (entity1)-[:MONITORS]->(entity2)
                RETURN entity1.name, entity2.name, 'monitors' as relationship
                LIMIT 10
            """,
            
            "find_data_providers": """
                MATCH (provider)-[:PROVIDES]->(data)
                RETURN provider.name, data.name, 'provides' as relationship
                LIMIT 10
            """,
            
            "find_entity_by_name": """
                MATCH (e:Entity)
                WHERE e.name CONTAINS $search_term
                RETURN e.name, e.type, e.confidence
                LIMIT 10
            """,
            
            "find_shortest_path": """
                MATCH path = shortestPath((a:Entity {name: $start})-[*]-(b:Entity {name: $end}))
                RETURN path
            """,
            
            "get_entity_neighbors": """
                MATCH (e:Entity {name: $entity_name})-[r]-(neighbor)
                RETURN neighbor.name, type(r) as relationship, r.confidence
                LIMIT 20
            """,
            
            "find_high_confidence_triples": """
                MATCH (a)-[r]->(b)
                WHERE r.confidence > 0.8
                RETURN a.name, type(r), b.name, r.confidence
                ORDER BY r.confidence DESC
                LIMIT 20
            """
        }
        
        queries_file = self.kg_dir / "sample_cypher_queries.json"
        with open(queries_file, 'w', encoding='utf-8') as f:
            json.dump(sample_queries, f, ensure_ascii=False, indent=2)
        
        self.logger.info(f"Sample queries saved to: {queries_file}")
        return sample_queries
    
    def load_knowledge_graph(self, clear_existing: bool = True, kg_file: str = None):
        """
        Main method to load complete knowledge graph into Neo4j
        
        Args:
            clear_existing: Whether to clear existing data
            kg_file: Path to knowledge graph JSON file
        """
        self.logger.info("Starting Knowledge Graph loading into Neo4j")
        
        # Load KG data
        if not kg_file:
            kg_file = str(self.kg_dir / "mosdac_knowledge_graph.json")
        
        try:
            with open(kg_file, 'r', encoding='utf-8') as f:
                kg_data = json.load(f)
        except FileNotFoundError:
            self.logger.error(f"Knowledge graph file not found: {kg_file}")
            raise
        
        try:
            # Clear existing data if requested
            if clear_existing:
                self.logger.info("Clearing existing data")
                self.clear_database()
            
            # Create constraints and indexes
            self.create_constraints_and_indexes()
            
            # Load entities
            self.load_entities(kg_data)
            
            # Load relationships
            self.load_relationships(kg_data)
            
            # Create domain-specific structures
            self.create_domain_specific_nodes()
            
            # Load metadata
            self.load_metadata_and_statistics(kg_data)
            
            # Verify loaded data
            verification_results = self.verify_loaded_data()
            
            # Create sample queries
            self.create_sample_queries()
            
            self.logger.info("Knowledge Graph loading completed successfully!")
            
            return verification_results
            
        except Exception as e:
            self.logger.error(f"Error loading knowledge graph: {e}")
            raise
        finally:
            self.close_connection()


def setup_neo4j_credentials():
    """Setup Neo4j credentials interactively or from environment"""
    
    # Check if credentials exist in environment
    uri = os.getenv('NEO4J_URI')
    user = os.getenv('NEO4J_USER') 
    password = os.getenv('NEO4J_PASSWORD')
    
    if uri and user and password:
        print("✅ Found Neo4j credentials in environment variables")
        return uri, user, password
    
    print("🔐 Neo4j Credentials Setup")
    print("=" * 40)
    print("Please provide your Neo4j Aura credentials:")
    print("(You can get these from https://console.neo4j.io/)")
    print()
    
    if not uri:
        uri = input("Neo4j URI (e.g., bolt://xxxxx.databases.neo4j.io:7687): ").strip()
    
    if not user:
        user = input("Username (usually 'neo4j'): ").strip() or 'neo4j'
    
    if not password:
        import getpass
        password = getpass.getpass("Password: ").strip()
    
    # Save to .env file
    env_file = Path(".env")
    env_content = f"""# Neo4j Aura Credentials
NEO4J_URI={uri}
NEO4J_USER={user}
NEO4J_PASSWORD={password}

# Gemini API Key (add this later)
# GEMINI_API_KEY=your_gemini_api_key_here
"""
    
    with open(env_file, 'w') as f:
        f.write(env_content)
    
    print(f"✅ Credentials saved to {env_file}")
    return uri, user, password


def main():
    """
    Main function to load Knowledge Graph into Neo4j
    """
    print("🗄️ MOSDAC Knowledge Graph Loader")
    print("=" * 50)
    print("Loading Knowledge Graph into Neo4j Aura")
    print("=" * 50)
    
    try:
        # Setup credentials
        uri, user, password = setup_neo4j_credentials()
        
        # Initialize loader
        loader = Neo4jKGLoader(uri=uri, user=user, password=password)
        
        # Load knowledge graph
        verification_results = loader.load_knowledge_graph()
        
        print(f"\n✅ Knowledge Graph loaded successfully!")
        print(f"📊 Loading Statistics:")
        print(f"  - Total nodes: {verification_results['total_nodes']}")
        print(f"  - Total relationships: {verification_results['total_relationships']}")
        print(f"  - Node types: {len(verification_results['node_types'])}")
        print(f"  - Relationship types: {len(verification_results['relationship_types'])}")
        
        print(f"\n📁 Output files:")
        print(f"  - Verification: mosdac_data/knowledge_graph/neo4j_verification.json")
        print(f"  - Sample queries: mosdac_data/knowledge_graph/sample_cypher_queries.json")
        
        print(f"\n🔗 Access your Neo4j database at: https://console.neo4j.io/")
        print(f"🎯 Ready for LangChain integration!")
        
    except Exception as e:
        print(f"\n❌ Knowledge Graph loading failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()