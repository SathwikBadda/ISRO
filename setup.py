#!/usr/bin/env python3
"""
Setup script for MOSDAC Data Pipeline
Bharatiya Antariksh Hackathon 2025 - Problem Statement 2
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

def install_requirements():
    """Install required Python packages"""
    print("📦 Installing Python packages...")
    
    # Install from requirements.txt
    if Path("requirements.txt").exists():
        success = run_command(
            f"{sys.executable} -m pip install -r requirements.txt",
            "Installing packages from requirements.txt"
        )
        if not success:
            return False
    else:
        # Install packages individually
        packages = [
            "requests>=2.28.0",
            "beautifulsoup4>=4.11.0", 
            "selenium>=4.8.0",
            "pandas>=1.5.0",
            "pdfplumber>=0.7.0",
            "PyMuPDF>=1.21.0",
            "python-docx>=0.8.11",
            "openpyxl>=3.1.0",
            "nltk>=3.8",
            "spacy>=3.5.0",
            "langdetect>=1.0.9",
            "ftfy>=6.1.0",
            "tqdm>=4.64.0"
        ]
        
        for package in packages:
            success = run_command(
                f"{sys.executable} -m pip install {package}",
                f"Installing {package.split('>=')[0]}"
            )
            if not success:
                print(f"⚠️  Failed to install {package}, continuing...")
    
    return True

def download_nltk_data():
    """Download required NLTK data"""
    print("📚 Downloading NLTK data...")
    
    nltk_downloads = ['punkt', 'stopwords', 'averaged_perceptron_tagger']
    
    try:
        import nltk
        for dataset in nltk_downloads:
            try:
                nltk.download(dataset, quiet=True)
                print(f"✅ Downloaded NLTK {dataset}")
            except Exception as e:
                print(f"⚠️  Failed to download NLTK {dataset}: {e}")
    except ImportError:
        print("❌ NLTK not installed")
        return False
    
    return True

def install_spacy_model():
    """Install spaCy English model"""
    print("🧠 Installing spaCy English model...")
    
    return run_command(
        f"{sys.executable} -m spacy download en_core_web_sm",
        "Installing spaCy English model"
    )

def check_chromedriver():
    """Check if ChromeDriver is available"""
    print("🌐 Checking ChromeDriver...")
    
    try:
        result = subprocess.run(["chromedriver", "--version"], 
                              capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            print(f"✅ ChromeDriver found: {result.stdout.strip()}")
            return True
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    
    print("⚠️  ChromeDriver not found in PATH")
    print("   Please install ChromeDriver:")
    
    system = platform.system().lower()
    if system == "windows":
        print("   - Download from: https://chromedriver.chromium.org/")
        print("   - Add to PATH or place in project directory")
    elif system == "darwin":  # macOS
        print("   - Install via Homebrew: brew install chromedriver")
    else:  # Linux
        print("   - Install via package manager or download manually")
        print("   - Ubuntu/Debian: sudo apt-get install chromium-chromedriver")
    
    return False

def create_directories():
    """Create required directory structure"""
    print("📁 Creating directory structure...")
    
    directories = [
        "mosdac_data",
        "mosdac_data/raw_html",
        "mosdac_data/raw_docs",
        "mosdac_data/parsed_docs", 
        "mosdac_data/cleaned_json",
        "mosdac_data/logs"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✅ Created directory: {directory}")
    
    return True

def create_config_files():
    """Create default configuration files if they don't exist"""
    print("⚙️  Creating configuration files...")
    
    # Create .gitignore
    gitignore_content = """# MOSDAC Pipeline Generated Files
mosdac_data/
*.log
__pycache__/
*.pyc
*.pyo
*.pyd
.Python
env/
venv/
.venv/
.env
.DS_Store
Thumbs.db
chromedriver.exe
chromedriver
geckodriver.exe
geckodriver
"""
    
    if not Path(".gitignore").exists():
        with open(".gitignore", "w") as f:
            f.write(gitignore_content)
        print("✅ Created .gitignore")
    
    # Create environment file template
    env_content = """# MOSDAC Pipeline Environment Variables
# Uncomment and set values as needed

# MOSDAC_BASE_URL=https://www.mosdac.gov.in
# MOSDAC_OUTPUT_DIR=mosdac_data
# MOSDAC_MAX_PAGES=50
# MOSDAC_DELAY=2.0
# MOSDAC_TIMEOUT=10

# Chrome/ChromeDriver settings
# CHROMEDRIVER_PATH=/path/to/chromedriver
# CHROME_BINARY_PATH=/path/to/chrome

# Logging level (DEBUG, INFO, WARNING, ERROR)
# LOG_LEVEL=INFO
"""
    
    if not Path(".env.example").exists():
        with open(".env.example", "w") as f:
            f.write(env_content)
        print("✅ Created .env.example")
    
    return True

def run_tests():
    """Run basic tests to ensure setup is working"""
    print("🧪 Running basic tests...")
    
    # Test imports
    test_imports = [
        "requests",
        "bs4", 
        "selenium",
        "pandas",
        "pdfplumber",
        "fitz",  # PyMuPDF
        "docx",
        "openpyxl",
        "nltk",
        "spacy",
        "langdetect",
        "ftfy",
        "tqdm"
    ]
    
    failed_imports = []
    for module in test_imports:
        try:
            __import__(module)
            print(f"✅ {module} import successful")
        except ImportError:
            print(f"❌ {module} import failed")
            failed_imports.append(module)
    
    if failed_imports:
        print(f"⚠️  Failed imports: {failed_imports}")
        return False
    
    # Test Selenium WebDriver
    try:
        from selenium import webdriver
        from selenium.webdriver.chrome.options import Options
        
        chrome_options = Options()
        chrome_options.add_argument("--headless")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        
        driver = webdriver.Chrome(options=chrome_options)
        driver.get("https://www.google.com")
        driver.quit()
        print("✅ Selenium WebDriver test successful")
    except Exception as e:
        print(f"⚠️  Selenium WebDriver test failed: {e}")
        print("   This is not critical - static scraping will still work")
    
    return True

def main():
    """Main setup function"""
    print("🚀 MOSDAC Data Pipeline Setup")
    print("=" * 50)
    print("Bharatiya Antariksh Hackathon 2025 - Problem Statement 2")
    print("=" * 50)
    
    # Run setup steps
    steps = [
        ("Check Python version", check_python_version),
        ("Install Python packages", install_requirements),
        ("Download NLTK data", download_nltk_data),
        ("Install spaCy model", install_spacy_model), 
        ("Check ChromeDriver", check_chromedriver),
        ("Create directories", create_directories),
        ("Create config files", create_config_files),
        ("Run basic tests", run_tests)
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
    
    if success_count >= total_steps - 1:  # Allow one warning
        print("✅ Setup completed successfully!")
        print("\n📋 Next steps:")
        print("1. Review mosdac_config.json for customization")
        print("2. Run: python mosdac_pipeline.py")
        print("3. Check output in mosdac_data/ directory")
        print("4. Monitor logs in mosdac_data/logs/scraping.log")
    else:
        print("⚠️  Setup completed with issues. Please review the errors above.")
    
    print("\n🎯 Good luck with Bharatiya Antariksh Hackathon 2025!")

if __name__ == "__main__":
    main()