#!/usr/bin/env python3
"""
NER + Relation Extraction Pipeline for MOSDAC Knowledge Graph
Bharatiya Antariksh Hackathon 2025 - Problem Statement 2 - Phase 2

This script performs:
1. Named Entity Recognition (NER) to extract satellites, sensors, missions
2. Relation Extraction (RE) to identify relationships
3. Triple generation for Knowledge Graph construction
"""

import os
import json
import re
from pathlib import Path
from typing import List, Dict, Tuple, Set
import logging
from datetime import datetime

# Core libraries
import pandas as pd
import numpy as np
from tqdm import tqdm

# HuggingFace libraries
from transformers import AutoTokenizer, AutoModelForTokenClassification, AutoModelForSeq2SeqLM
from transformers.pipelines import pipeline  # type: ignore
import torch

# Additional NLP libraries
import spacy
from spacy import displacy
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.chunk import ne_chunk
from nltk.tag import pos_tag
from dotenv import load_dotenv

load_dotenv()

class MOSDACNERPipeline:
    """
    Named Entity Recognition pipeline for MOSDAC data
    Extracts satellites, sensors, missions, locations, etc.
    """
    
    def __init__(self, output_dir: str = "mosdac_data"):
        self.output_dir = Path(output_dir)
        self.kg_output_dir = self.output_dir / "knowledge_graph"
        self.kg_output_dir.mkdir(exist_ok=True)
        
        # Setup logging
        self.setup_logging()
        
        # Initialize models
        self.ner_model = None
        self.tokenizer = None
        self.nlp = None
        
        # Load models
        self.load_models()
        
        # Define entity categories specific to MOSDAC/Satellite domain
        self.entity_categories = {
            'SATELLITE': ['insat', 'goes', 'meteosat', 'megha', 'tropiques', 'kalpana', 'cartosat', 'resourcesat'],
            'SENSOR': ['imager', 'sounder', 'vhrr', 'cwc', 'saphir', 'madras', 'scarab', 'viirs', 'modis'],
            'MISSION': ['mission', 'program', 'programme', 'project', 'campaign'],
            'LOCATION': ['india', 'indian', 'ocean', 'asia', 'pacific', 'antarctic', 'arctic'],
            'PRODUCT': ['temperature', 'rainfall', 'humidity', 'pressure', 'wind', 'cloud', 'sst', 'ndvi'],
            'ORGANIZATION': ['isro', 'mosdac', 'sac', 'dos', 'imd', 'nasa', 'noaa', 'eumetsat'],
            'PARAMETER': ['celsius', 'kelvin', 'mbar', 'hpa', 'mm', 'km', 'degree', 'percent']
        }
        
        self.logger.info("MOSDAC NER Pipeline initialized successfully")
    
    def setup_logging(self):
        """Setup logging for NER pipeline"""
        log_file = self.output_dir / "logs" / "ner_pipeline.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        
        self.logger = logging.getLogger(__name__)
    
    def load_models(self):
        """Load HuggingFace NER models and spaCy"""
        try:
            # Load BERT-based NER model
            model_name = "dslim/bert-base-NER"
            self.logger.info(f"Loading NER model: {model_name}")
            
            hf_token = os.getenv("HUGGINGFACE_HUB_TOKEN", "")
            if not hf_token:
                raise RuntimeError("HUGGINGFACE_HUB_TOKEN not set in .env file. Please add your HuggingFace API token.")
            
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                token=hf_token,
                local_files_only=False
            )
            self.ner_model = AutoModelForTokenClassification.from_pretrained(
                model_name,
                token=hf_token,
                local_files_only=False
            )
            
            # Create NER pipeline
            self.ner_pipeline = pipeline(
                "ner",
                model=self.ner_model,
                tokenizer=self.tokenizer,
                aggregation_strategy="simple"
            )  # type: ignore
            
            # Load spaCy model
            try:
                self.nlp = spacy.load("en_core_web_sm")
                self.logger.info("spaCy model loaded successfully")
            except OSError:
                self.logger.warning("spaCy model not found. Using NLTK fallback.")
                self.nlp = None
            
            self.logger.info("NER models loaded successfully")
            
        except Exception as e:
            self.logger.error(f"Error loading NER models: {e}")
            raise
    
    def extract_entities_bert(self, text: str) -> List[Dict]:
        """
        Extract entities using BERT-based NER model
        
        Args:
            text: Input text for entity extraction
            
        Returns:
            List of extracted entities with labels and confidence scores
        """
        try:
            # Use HuggingFace pipeline for NER
            entities = self.ner_pipeline(text)
            
            # Process and clean entities
            processed_entities = []
            for entity in entities:
                processed_entities.append({
                    'text': entity.get('word', ''),
                    'label': entity.get('entity_group', entity.get('label', '')),
                    'confidence': float(entity.get('score', 0.0)),
                    'start': entity.get('start', 0),
                    'end': entity.get('end', 0)
                })
            
            return processed_entities
            
        except Exception as e:
            self.logger.error(f"Error in BERT NER extraction: {e}")
            return []
    
    def extract_entities_spacy(self, text: str) -> List[Dict]:
        """
        Extract entities using spaCy NER
        
        Args:
            text: Input text for entity extraction
            
        Returns:
            List of extracted entities
        """
        if not self.nlp:
            return []
        
        try:
            doc = self.nlp(text)
            entities = []
            
            for ent in doc.ents:
                entities.append({
                    'text': ent.text,
                    'label': ent.label_,
                    'confidence': 1.0,  # spaCy doesn't provide confidence scores
                    'start': ent.start_char,
                    'end': ent.end_char
                })
            
            return entities
            
        except Exception as e:
            self.logger.error(f"Error in spaCy NER extraction: {e}")
            return []
    
    def extract_domain_entities(self, text: str) -> List[Dict]:
        """
        Extract domain-specific entities using keyword matching
        
        Args:
            text: Input text for entity extraction
            
        Returns:
            List of domain-specific entities
        """
        text_lower = text.lower()
        entities = []
        
        for category, keywords in self.entity_categories.items():
            for keyword in keywords:
                # Find all occurrences of the keyword
                pattern = r'\b' + re.escape(keyword) + r'\b'
                matches = re.finditer(pattern, text_lower)
                
                for match in matches:
                    # Get the original case from the text
                    start, end = match.span()
                    original_text = text[start:end]
                    
                    entities.append({
                        'text': original_text,
                        'label': category,
                        'confidence': 0.9,  # High confidence for domain keywords
                        'start': start,
                        'end': end,
                        'source': 'domain_specific'
                    })
        
        return entities
    
    def merge_entities(self, bert_entities: List[Dict], spacy_entities: List[Dict], 
                      domain_entities: List[Dict]) -> List[Dict]:
        """
        Merge entities from different sources and remove duplicates
        
        Args:
            bert_entities: Entities from BERT model
            spacy_entities: Entities from spaCy
            domain_entities: Domain-specific entities
            
        Returns:
            Merged and deduplicated entity list
        """
        all_entities = bert_entities + spacy_entities + domain_entities
        
        # Sort by start position
        all_entities.sort(key=lambda x: x['start'])
        
        # Remove duplicates based on text overlap
        merged_entities = []
        for entity in all_entities:
            # Check for overlap with existing entities
            overlaps = False
            for existing in merged_entities:
                if (entity['start'] < existing['end'] and 
                    entity['end'] > existing['start']):
                    # Choose entity with higher confidence
                    if entity['confidence'] > existing['confidence']:
                        merged_entities.remove(existing)
                        merged_entities.append(entity)
                    overlaps = True
                    break
            
            if not overlaps:
                merged_entities.append(entity)
        
        return merged_entities
    
    def process_document(self, document: Dict) -> Dict:
        """
        Process a single document to extract entities
        
        Args:
            document: Document dictionary with text content
            
        Returns:
            Document with extracted entities
        """
        text = document.get('text', '')
        title = document.get('title', '')
        
        # Combine title and text for entity extraction
        full_text = f"{title}. {text}"
        
        # Extract entities using different methods
        bert_entities = self.extract_entities_bert(full_text)
        spacy_entities = self.extract_entities_spacy(full_text)
        domain_entities = self.extract_domain_entities(full_text)
        
        # Merge entities
        entities = self.merge_entities(bert_entities, spacy_entities, domain_entities)
        
        # Add entities to document
        document['entities'] = entities
        document['entity_count'] = len(entities)
        document['processed_at'] = datetime.now().isoformat()
        
        return document
    
    def process_corpus(self, input_file: str = "") -> List[Dict]:
        """
        Process entire corpus to extract entities
        
        Args:
            input_file: Path to input JSON file (default: all_cleaned.jsonl)
            
        Returns:
            List of processed documents with entities
        """
        if not input_file:
            input_file = str(self.output_dir / "all_cleaned.jsonl")
        
        self.logger.info(f"Processing corpus from: {input_file}")
        
        # Load documents
        documents = []
        try:
            with open(str(input_file), 'r', encoding='utf-8') as f:
                for line in f:
                    documents.append(json.loads(line))
        except FileNotFoundError:
            self.logger.error(f"Input file not found: {input_file}")
            return []
        
        # Process documents
        processed_documents = []
        for doc in tqdm(documents, desc="Extracting entities"):
            try:
                processed_doc = self.process_document(doc)
                processed_documents.append(processed_doc)
            except Exception as e:
                self.logger.error(f"Error processing document: {e}")
                continue
        
        # Save processed documents
        output_file = self.kg_output_dir / "documents_with_entities.jsonl"
        with open(output_file, 'w', encoding='utf-8') as f:
            for doc in processed_documents:
                f.write(json.dumps(doc, ensure_ascii=False) + '\n')
        
        self.logger.info(f"Processed {len(processed_documents)} documents")
        self.logger.info(f"Saved entities to: {output_file}")
        
        return processed_documents


class MOSDACRelationExtractor:
    """
    Relation Extraction pipeline for MOSDAC Knowledge Graph
    Extracts relationships between entities using REBEL model
    """
    
    def __init__(self, output_dir: str = "mosdac_data"):
        self.output_dir = Path(output_dir)
        self.kg_output_dir = self.output_dir / "knowledge_graph"
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        # Initialize REBEL model for relation extraction
        self.relation_model = None
        self.relation_tokenizer = None
        
        # Load models
        self.load_relation_model()
        
        # Define domain-specific relation patterns
        self.relation_patterns = {
            'provides': [
                r'(\w+)\s+(provides?|generates?|produces?)\s+(\w+)',
                r'(\w+)\s+(data|information|observations?)\s+(from|of)\s+(\w+)'
            ],
            'monitors': [
                r'(\w+)\s+(monitors?|observes?|tracks?|measures?)\s+(\w+)',
                r'(\w+)\s+(for|used for)\s+(monitoring|observation|measurement)\s+(of\s+)?(\w+)'
            ],
            'part_of': [
                r'(\w+)\s+(sensor|instrument|payload)\s+(on|aboard|in)\s+(\w+)',
                r'(\w+)\s+(is|are)\s+(part of|component of|aboard)\s+(\w+)'
            ],
            'located_in': [
                r'(\w+)\s+(in|over|above)\s+(\w+)',
                r'(\w+)\s+(region|area|zone)\s+(of|in)\s+(\w+)'
            ],
            'operated_by': [
                r'(\w+)\s+(operated by|managed by|controlled by)\s+(\w+)',
                r'(\w+)\s+(mission|satellite|program)\s+(of|by)\s+(\w+)'
            ]
        }
    
    def load_relation_model(self):
        """Load REBEL relation extraction model"""
        try:
            model_name = "Babelscape/rebel-large"
            self.logger.info(f"Loading relation extraction model: {model_name}")
            
            hf_token = os.getenv("HUGGINGFACE_HUB_TOKEN", "")
            if not hf_token:
                raise RuntimeError("HUGGINGFACE_HUB_TOKEN not set in .env file. Please add your HuggingFace API token.")
            
            self.relation_tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                token=hf_token,
                local_files_only=False
            )
            self.relation_model = AutoModelForSeq2SeqLM.from_pretrained(
                model_name,
                token=hf_token,
                local_files_only=False
            )
            # Create relation extraction pipeline
            self.relation_pipeline = pipeline(
                "text2text-generation",
                model=self.relation_model,
                tokenizer=self.relation_tokenizer
            )  # type: ignore
            
            self.logger.info("Relation extraction model loaded successfully")
            
        except Exception as e:
            self.logger.error(f"Error loading relation model: {e}")
            self.relation_pipeline = None
    
    def extract_relations_rebel(self, text: str) -> List[Dict]:
        """
        Extract relations using REBEL model
        
        Args:
            text: Input text for relation extraction
            
        Returns:
            List of extracted relations
        """
        if not self.relation_pipeline:
            return []
        
        try:
            # Process text with REBEL
            relations = []
            
            # Split text into sentences for better processing
            sentences = sent_tokenize(text)
            
            for sentence in sentences:
                if len(sentence) < 10:  # Skip very short sentences
                    continue
                
                # Generate relations
                outputs = self.relation_pipeline(sentence, max_length=512)
                
                for output in outputs:
                    generated_text = output['generated_text']
                    # Parse REBEL output format
                    parsed_relations = self.parse_rebel_output(generated_text)
                    relations.extend(parsed_relations)
            
            return relations
            
        except Exception as e:
            self.logger.error(f"Error in REBEL relation extraction: {e}")
            return []
    
    def parse_rebel_output(self, text: str) -> List[Dict]:
        """
        Parse REBEL model output to extract triples
        
        Args:
            text: Generated text from REBEL model
            
        Returns:
            List of relation triples
        """
        relations = []
        
        # REBEL output format: <triplet> head <subj> relation <obj> tail
        triplet_pattern = r'<triplet>\s*(.+?)\s*<subj>\s*(.+?)\s*<obj>\s*(.+?)(?=<triplet>|$)'
        matches = re.findall(triplet_pattern, text)
        
        for match in matches:
            if len(match) == 3:
                head, relation, tail = match
                relations.append({
                    'head': head.strip(),
                    'relation': relation.strip(),
                    'tail': tail.strip(),
                    'confidence': 0.8,
                    'source': 'rebel'
                })
        
        return relations
    
    def extract_relations_patterns(self, text: str, entities: List[Dict]) -> List[Dict]:
        """
        Extract relations using regex patterns
        
        Args:
            text: Input text
            entities: List of extracted entities
            
        Returns:
            List of extracted relations
        """
        relations = []
        text_lower = text.lower()
        
        # Create entity lookup
        entity_texts = [ent['text'].lower() for ent in entities]
        
        for relation_type, patterns in self.relation_patterns.items():
            for pattern in patterns:
                matches = re.finditer(pattern, text_lower)
                
                for match in matches:
                    groups = match.groups()
                    if len(groups) >= 2:
                        # Extract head and tail entities
                        head = groups[0].strip()
                        tail = groups[-1].strip()
                        
                        # Verify entities exist in our entity list
                        if head in entity_texts or tail in entity_texts:
                            relations.append({
                                'head': head,
                                'relation': relation_type,
                                'tail': tail,
                                'confidence': 0.7,
                                'source': 'pattern_matching'
                            })
        
        return relations
    
    def extract_document_relations(self, document: Dict) -> List[Dict]:
        """
        Extract relations from a single document
        
        Args:
            document: Document with entities
            
        Returns:
            List of extracted relations
        """
        text = document.get('text', '')
        entities = document.get('entities', [])
        
        # Extract relations using REBEL
        rebel_relations = self.extract_relations_rebel(text)
        
        # Extract relations using patterns
        pattern_relations = self.extract_relations_patterns(text, entities)
        
        # Combine and deduplicate relations
        all_relations = rebel_relations + pattern_relations
        
        # Remove duplicates
        unique_relations = []
        seen = set()
        
        for rel in all_relations:
            rel_tuple = (rel['head'].lower(), rel['relation'].lower(), rel['tail'].lower())
            if rel_tuple not in seen:
                seen.add(rel_tuple)
                unique_relations.append(rel)
        
        return unique_relations
    
    def process_corpus_relations(self, documents: List[Dict]) -> List[Dict]:
        """
        Process entire corpus to extract relations
        
        Args:
            documents: List of documents with entities
            
        Returns:
            List of all extracted relations (triples)
        """
        self.logger.info("Extracting relations from corpus")
        
        all_triples = []
        
        for doc in tqdm(documents, desc="Extracting relations"):
            try:
                relations = self.extract_document_relations(doc)
                
                # Add document metadata to relations
                for rel in relations:
                    rel['source_url'] = doc.get('source_url', '')
                    rel['document_type'] = doc.get('document_type', '')
                    rel['section'] = doc.get('section', '')
                
                all_triples.extend(relations)
                
            except Exception as e:
                self.logger.error(f"Error extracting relations from document: {e}")
                continue
        
        # Save triples
        output_file = self.kg_output_dir / "knowledge_triples.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(all_triples, f, ensure_ascii=False, indent=2)
        
        self.logger.info(f"Extracted {len(all_triples)} triples")
        self.logger.info(f"Saved triples to: {output_file}")
        
        return all_triples


class KnowledgeGraphBuilder:
    """
    Main class to orchestrate NER + RE pipeline and build Knowledge Graph
    """
    
    def __init__(self, output_dir: str = "mosdac_data"):
        self.output_dir = Path(output_dir)
        self.kg_output_dir = self.output_dir / "knowledge_graph"
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.ner_pipeline = MOSDACNERPipeline(output_dir)
        self.relation_extractor = MOSDACRelationExtractor(output_dir)
    
    def build_knowledge_graph(self, input_file: str = "") -> Dict:
        """
        Build complete Knowledge Graph from cleaned MOSDAC data
        
        Args:
            input_file: Path to cleaned data file
            
        Returns:
            Dictionary containing entities and relations
        """
        self.logger.info("Starting Knowledge Graph construction")
        
        # Step 1: Extract entities
        self.logger.info("Step 1: Extracting entities")
        documents = self.ner_pipeline.process_corpus(input_file)
        
        # Step 2: Extract relations
        self.logger.info("Step 2: Extracting relations")
        triples = self.relation_extractor.process_corpus_relations(documents)
        
        # Step 3: Build final KG structure
        self.logger.info("Step 3: Building final KG structure")
        kg_data = self.create_kg_structure(documents, triples)
        
        # Save final KG
        output_file = self.kg_output_dir / "mosdac_knowledge_graph.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(kg_data, f, ensure_ascii=False, indent=2)
        
        self.logger.info(f"Knowledge Graph construction completed")
        self.logger.info(f"Final KG saved to: {output_file}")
        
        return kg_data
    
    def create_kg_structure(self, documents: List[Dict], triples: List[Dict]) -> Dict:
        """
        Create final Knowledge Graph structure
        
        Args:
            documents: Processed documents with entities
            triples: Extracted relation triples
            
        Returns:
            Structured Knowledge Graph data
        """
        # Collect all entities
        all_entities = {}
        entity_types = {}
        
        for doc in documents:
            for entity in doc.get('entities', []):
                entity_text = entity['text']
                if entity_text not in all_entities:
                    all_entities[entity_text] = {
                        'text': entity_text,
                        'label': entity['label'],
                        'confidence': entity['confidence'],
                        'sources': []
                    }
                
                all_entities[entity_text]['sources'].append({
                    'document': doc.get('title', ''),
                    'url': doc.get('source_url', ''),
                    'section': doc.get('section', '')
                })
                
                # Track entity types
                entity_types[entity['label']] = entity_types.get(entity['label'], 0) + 1
        
        # Collect relation types
        relation_types = {}
        for triple in triples:
            rel_type = triple['relation']
            relation_types[rel_type] = relation_types.get(rel_type, 0) + 1
        
        # Create final structure
        kg_structure = {
            'metadata': {
                'created_at': datetime.now().isoformat(),
                'total_documents': len(documents),
                'total_entities': len(all_entities),
                'total_triples': len(triples),
                'entity_types': entity_types,
                'relation_types': relation_types
            },
            'entities': list(all_entities.values()),
            'triples': triples,
            'statistics': {
                'most_common_entities': self.get_most_common_entities(all_entities),
                'most_common_relations': self.get_most_common_relations(triples)
            }
        }
        
        return kg_structure
    
    def get_most_common_entities(self, entities: Dict, top_k: int = 10) -> List[Dict]:
        """Get most common entities by frequency"""
        entity_freq = {}
        for entity_data in entities.values():
            entity_freq[entity_data['text']] = len(entity_data['sources'])
        
        sorted_entities = sorted(entity_freq.items(), key=lambda x: x[1], reverse=True)
        return [{'entity': ent, 'frequency': freq} for ent, freq in sorted_entities[:top_k]]
    
    def get_most_common_relations(self, triples: List[Dict], top_k: int = 10) -> List[Dict]:
        """Get most common relations by frequency"""
        relation_freq = {}
        for triple in triples:
            rel = triple['relation']
            relation_freq[rel] = relation_freq.get(rel, 0) + 1
        
        sorted_relations = sorted(relation_freq.items(), key=lambda x: x[1], reverse=True)
        return [{'relation': rel, 'frequency': freq} for rel, freq in sorted_relations[:top_k]]


def main():
    """
    Main function to run NER + RE pipeline
    """
    print("🧠 MOSDAC Knowledge Graph Construction")
    print("=" * 50)
    print("Phase 2: NER + Relation Extraction + KG Building")
    print("=" * 50)
    
    try:
        # Initialize KG Builder
        kg_builder = KnowledgeGraphBuilder()
        
        # Build Knowledge Graph
        kg_data = kg_builder.build_knowledge_graph()
        
        print(f"\n✅ Knowledge Graph construction completed!")
        print(f"📊 Statistics:")
        print(f"  - Total entities: {kg_data['metadata']['total_entities']}")
        print(f"  - Total triples: {kg_data['metadata']['total_triples']}")
        print(f"  - Entity types: {len(kg_data['metadata']['entity_types'])}")
        print(f"  - Relation types: {len(kg_data['metadata']['relation_types'])}")
        
        print(f"\n📁 Output files:")
        print(f"  - Entities: mosdac_data/knowledge_graph/documents_with_entities.jsonl")
        print(f"  - Triples: mosdac_data/knowledge_graph/knowledge_triples.json")
        print(f"  - Final KG: mosdac_data/knowledge_graph/mosdac_knowledge_graph.json")
        
    except Exception as e:
        print(f"\n❌ Knowledge Graph construction failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()