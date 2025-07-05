#!/usr/bin/env python3
"""
Setup Script for MOSDAC Knowledge Graph & Chat Interface - Phase 2
Bharatiya Antariksh Hackathon 2025 - Problem Statement 2

This script sets up Phase 2 components:
- NER and Relation Extraction models
- Neo4j Knowledge Graph
- LangChain integration
- Gemini API
- Streamlit chat interface
"""

import os
import sys
import subprocess
import platform
from pathlib import Path

def run_command(command, description):
    """Run a system command with error handling"""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e}")
        print(f"Error output: {e.stderr}")
        return False

def check_python_version():
    """Check if Python version is compatible"""
    print("🐍 Checking Python version...")
    
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print(f"❌ Python 3.8+ required, found {version.major}.{version.minor}")
        return False
    
    print(f"✅ Python {version.major}.{version.minor}.{version.micro} is compatible")
    return True

def install_phase2_requirements():
    """Install Phase 2 Python packages"""
    print("📦 Installing Phase 2 Python packages...")
    
    # Install from requirements file
    if Path("requirements_phase2.txt").exists():
        success = run_command(
            f"{sys.executable} -m pip install -r requirements_phase2.txt",
            "Installing packages from requirements_phase2.txt"
        )
        if not success:
            return False
    else:
        # Install packages individually
        packages = [
            "transformers>=4.25.0",
            "torch>=1.13.0",
            "neo4j>=5.3.0",
            "langchain>=0.1.0",
            "langchain-community>=0.0.10",
            "langchain-google-genai>=0.0.5",
            "google-generativeai>=0.3.0",
            "streamlit>=1.28.0",
            "plotly>=5.17.0",
            "python-dotenv>=1.0.0",
            "scikit-learn>=1.3.0",
            "networkx>=3.0"
        ]
        
        for package in packages:
            success = run_command(
                f"{sys.executable} -m pip install {package}",
                f"Installing {package.split('>=')[0]}"
            )
            if not success:
                print(f"⚠️  Failed to install {package}, continuing...")
    
    return True

def download_models():
    """Download required ML models"""
    print("🧠 Downloading ML models...")
    
    # Download HuggingFace models
    try:
        print("📥 Downloading BERT NER model...")
        from transformers import AutoTokenizer, AutoModelForTokenClassification
        
        # Download dslim/bert-base-NER
        tokenizer = AutoTokenizer.from_pretrained("dslim/bert-base-NER")
        model = AutoModelForTokenClassification.from_pretrained("dslim/bert-base-NER")
        print("✅ BERT NER model downloaded")
        
        # Download REBEL model for relation extraction
        print("📥 Downloading REBEL relation extraction model...")
        from transformers import AutoModelForSeq2SeqLM
        
        rebel_tokenizer = AutoTokenizer.from_pretrained("Babelscape/rebel-large")
        rebel_model = AutoModelForSeq2SeqLM.from_pretrained("Babelscape/rebel-large")
        print("✅ REBEL model downloaded")
        
    except Exception as e:
        print(f"⚠️  Error downloading models: {e}")
        print("   Models will be downloaded on first use")
    
    return True

def setup_environment_file():
    """Create .env file template with API keys"""
    print("⚙️  Setting up environment file...")
    
    env_content = """# MOSDAC Knowledge Graph Environment Variables
# Phase 2: API Keys and Configuration

# Neo4j Aura Database (Free Tier)
# Get from: https://console.neo4j.io/
NEO4J_URI=bolt://xxxxx.databases.neo4j.io:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_neo4j_password

# Gemini API Key
# Get from: https://aistudio.google.com/app/apikey
GEMINI_API_KEY=your_gemini_api_key_here

# HuggingFace API Token (Optional, for private models)
# Get from: https://huggingface.co/settings/tokens
# HUGGINGFACE_API_TOKEN=your_hf_token_here

# Model Configuration
NER_MODEL=dslim/bert-base-NER
RELATION_MODEL=Babelscape/rebel-large

# Application Settings
STREAMLIT_PORT=8501
LOG_LEVEL=INFO
MAX_QUERY_LENGTH=500
CACHE_ENABLED=true

# Performance Settings
MAX_WORKERS=4
BATCH_SIZE=16
MAX_MEMORY_GB=8
"""
    
    env_file = Path(".env")
    if not env_file.exists():
        with open(env_file, 'w') as f:
            f.write(env_content)
        print("✅ Created .env template file")
    else:
        print("✅ .env file already exists")
    
    return True

def create_phase2_directories():
    """Create Phase 2 directory structure"""
    print("📁 Creating Phase 2 directory structure...")
    
    directories = [
        "mosdac_data/knowledge_graph",
        "mosdac_data/models",
        "mosdac_data/streamlit_cache",
        "mosdac_data/exports",
        "mosdac_data/logs"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✅ Created directory: {directory}")
    
    return True

def test_imports():
    """Test critical imports for Phase 2"""
    print("🧪 Testing Phase 2 imports...")
    
    test_imports = [
        ("transformers", "HuggingFace Transformers"),
        ("torch", "PyTorch"),
        ("neo4j", "Neo4j Driver"),
        ("langchain", "LangChain"),
        ("google.generativeai", "Gemini API"),
        ("streamlit", "Streamlit"),
        ("plotly", "Plotly"),
        ("sklearn", "Scikit-learn"),
        ("networkx", "NetworkX")
    ]
    
    failed_imports = []
    for module, description in test_imports:
        try:
            __import__(module)
            print(f"✅ {description} import successful")
        except ImportError:
            print(f"❌ {description} import failed")
            failed_imports.append(description)
    
    if failed_imports:
        print(f"⚠️  Failed imports: {failed_imports}")
        return False
    
    return True

def test_model_loading():
    """Test loading of ML models"""
    print("🔬 Testing model loading...")
    
    try:
        # Test BERT NER model
        from transformers.pipelines import pipeline  # type: ignore
        ner_pipeline = pipeline("ner", model="dslim/bert-base-NER", aggregation_strategy="simple")  # type: ignore
        test_result = ner_pipeline("INSAT-3D monitors weather patterns over India")
        print("✅ BERT NER model loaded and tested successfully")
        
        # Test spaCy model
        import spacy
        try:
            nlp = spacy.load("en_core_web_sm")
            print("✅ spaCy model loaded successfully")
        except OSError:
            print("⚠️  spaCy model not found. Run: python -m spacy download en_core_web_sm")
        
    except Exception as e:
        print(f"⚠️  Model loading test failed: {e}")
        print("   Models will be downloaded on first use")
        return False
    
    return True

def check_api_connections():
    """Check API connections"""
    print("🔗 Checking API connections...")
    
    # Check if .env exists with keys
    env_file = Path(".env")
    if not env_file.exists():
        print("⚠️  .env file not found. Please configure API keys.")
        return False
    
    # Load environment variables
    from dotenv import load_dotenv
    load_dotenv()
    
    # Check Neo4j credentials
    neo4j_uri = os.getenv('NEO4J_URI')
    neo4j_user = os.getenv('NEO4J_USER')
    neo4j_password = os.getenv('NEO4J_PASSWORD')
    
    if not all([neo4j_uri, neo4j_user, neo4j_password]):
        print("⚠️  Neo4j credentials not configured in .env")
    else:
        print("✅ Neo4j credentials found in .env")
    
    # Check Gemini API key
    gemini_key = os.getenv('GEMINI_API_KEY')
    if not gemini_key or gemini_key == 'your_gemini_api_key_here':
        print("⚠️  Gemini API key not configured in .env")
    else:
        print("✅ Gemini API key found in .env")
    
    return True

def create_test_scripts():
    """Create test scripts for Phase 2"""
    print("📝 Creating test scripts...")
    
    # Test script for NER pipeline
    ner_test_script = """#!/usr/bin/env python3
# Test script for NER pipeline
from ner_re_pipeline import MOSDACNERPipeline

def test_ner():
    print("Testing NER pipeline...")
    pipeline = MOSDACNERPipeline()
    
    test_text = "INSAT-3D satellite monitors weather patterns using IMAGER sensor over India"
    bert_entities = pipeline.extract_entities_bert(test_text)
    domain_entities = pipeline.extract_domain_entities(test_text)
    
    print(f"BERT entities: {bert_entities}")
    print(f"Domain entities: {domain_entities}")
    print("NER pipeline test completed!")

if __name__ == "__main__":
    test_ner()
"""
    
    with open("test_ner.py", "w") as f:
        f.write(ner_test_script)
    
    # Test script for query agent
    agent_test_script = """#!/usr/bin/env python3
# Test script for query agent
import os
from query_agent import MOSDACQueryAgent

def test_agent():
    print("Testing query agent...")
    
    # Check environment variables
    required_vars = ['NEO4J_URI', 'NEO4J_USER', 'NEO4J_PASSWORD', 'GEMINI_API_KEY']
    missing = [var for var in required_vars if not os.getenv(var)]
    
    if missing:
        print(f"Missing environment variables: {missing}")
        print("Please configure .env file first")
        return
    
    try:
        agent = MOSDACQueryAgent()
        response = agent.process_query("List all satellites")
        print(f"Test query response: {response}")
        print("Query agent test completed!")
    except Exception as e:
        print(f"Query agent test failed: {e}")

if __name__ == "__main__":
    test_agent()
"""
    
    with open("test_agent.py", "w") as f:
        f.write(agent_test_script)
    
    print("✅ Test scripts created: test_ner.py, test_agent.py")
    return True

def display_next_steps():
    """Display next steps for the user"""
    print("\n" + "="*60)
    print("🎯 PHASE 2 SETUP COMPLETED!")
    print("="*60)
    
    print("\n📋 NEXT STEPS:")
    print("1. Configure API credentials in .env file:")
    print("   - Get Neo4j Aura database: https://console.neo4j.io/")
    print("   - Get Gemini API key: https://aistudio.google.com/app/apikey")
    print("   - Update .env file with your credentials")
    
    print("\n2. Run the complete pipeline:")
    print("   python ner_re_pipeline.py      # Extract entities and relations")
    print("   python kg_loader.py            # Load into Neo4j")
    print("   python query_agent.py          # Test query agent")
    print("   streamlit run app.py           # Launch chat interface")
    
    print("\n3. Test individual components:")
    print("   python test_ner.py             # Test NER pipeline")
    print("   python test_agent.py           # Test query agent")
    
    print("\n4. Access the applications:")
    print("   - Streamlit Chat: http://localhost:8501")
    print("   - Neo4j Browser: https://console.neo4j.io/")
    
    print("\n📊 PROJECT STRUCTURE:")
    print("mosdac_data/")
    print("├── knowledge_graph/           # KG data and triples")
    print("├── models/                    # Downloaded ML models")
    print("├── streamlit_cache/           # Streamlit cache")
    print("├── exports/                   # Exported conversations")
    print("└── logs/                      # Application logs")
    
    print("\n🚀 READY FOR HACKATHON!")
    print("Your MOSDAC Knowledge Graph Chat system is ready!")

def main():
    """Main setup function for Phase 2"""
    print("🚀 MOSDAC Knowledge Graph & Chat Interface Setup - Phase 2")
    print("=" * 70)
    print("Bharatiya Antariksh Hackathon 2025 - Problem Statement 2")
    print("=" * 70)
    
    steps = [
        ("Check Python version", check_python_version),
        ("Install Phase 2 packages", install_phase2_requirements),
        ("Download ML models", download_models),
        ("Setup environment file", setup_environment_file),
        ("Create directories", create_phase2_directories),
        ("Test imports", test_imports),
        ("Test model loading", test_model_loading),
        ("Check API connections", check_api_connections),
        ("Create test scripts", create_test_scripts)
    ]
    
    success_count = 0
    total_steps = len(steps)
    
    for step_name, step_function in steps:
        print(f"\n📍 Step: {step_name}")
        try:
            if step_function():
                success_count += 1
            else:
                print(f"⚠️  Step '{step_name}' completed with warnings")
        except Exception as e:
            print(f"❌ Step '{step_name}' failed: {e}")
    
    print(f"\n🏁 Setup completed: {success_count}/{total_steps} steps successful")
    
    if success_count >= total_steps - 1:
        display_next_steps()
    else:
        print("⚠️  Setup completed with issues. Please review the errors above.")
    
    print("\n🎯 Good luck with Bharatiya Antariksh Hackathon 2025!")

if __name__ == "__main__":
    main()