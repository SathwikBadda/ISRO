#!/usr/bin/env python3
"""
MOSDAC Data Extraction Pipeline
Bharatiya Antariksh Hackathon 2025 - Problem Statement 2

A comprehensive pipeline for scraping, parsing, cleaning, and structuring
MOSDAC portal data for NLP tasks (NER, chunking, KG, RAG).
"""

import os
import json
import time
import logging
import re
import unicodedata
from datetime import datetime
from urllib.parse import urljoin, urlparse
from typing import Dict, List, Optional, Union
from pathlib import Path
import hashlib

# Core libraries
import requests
from bs4 import BeautifulSoup
from bs4.element import Tag
import pandas as pd
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException

# Document parsing libraries
import pdfplumber
import fitz  # PyMuPDF
from docx import Document
import openpyxl

# Text processing libraries
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
import spacy
from langdetect import detect, DetectorFactory
import ftfy
from tqdm import tqdm

# Set seed for consistent language detection
DetectorFactory.seed = 0

class MOSDACPipeline:
    """
    Main pipeline class for MOSDAC data extraction and processing
    """
    
    def __init__(self, base_url: str = "https://www.mosdac.gov.in", 
                 output_dir: str = "mosdac_data"):
        """
        Initialize the MOSDAC pipeline
        
        Args:
            base_url: Base URL for MOSDAC portal
            output_dir: Base output directory for all data
        """
        self.base_url = base_url
        self.output_dir = Path(output_dir)
        self.session = requests.Session()
        
        # Setup directories
        self.setup_directories()
        
        # Setup logging
        self.setup_logging()
        
        # Initialize components
        self.scraper = WebScraper(self.base_url, self.output_dir, self.logger)
        self.parser = DocumentParser(self.output_dir, self.logger)
        self.cleaner = TextCleaner(self.output_dir, self.logger)
        
        # Download required NLTK data
        self.download_nltk_data()
        
        # Initialize spaCy model
        self.nlp = self.load_spacy_model()
        
        self.logger.info("MOSDAC Pipeline initialized successfully")
    
    def setup_directories(self):
        """Create required directory structure"""
        dirs = [
            "raw_html",
            "raw_docs", 
            "parsed_docs",
            "cleaned_json",
            "logs"
        ]
        
        for dir_name in dirs:
            dir_path = self.output_dir / dir_name
            dir_path.mkdir(parents=True, exist_ok=True)
    
    def setup_logging(self):
        """Setup logging configuration"""
        log_file = self.output_dir / "logs" / "scraping.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        
        self.logger = logging.getLogger(__name__)
    
    def download_nltk_data(self):
        """Download required NLTK data"""
        try:
            nltk.download('punkt', quiet=True)
            nltk.download('stopwords', quiet=True)
            nltk.download('averaged_perceptron_tagger', quiet=True)
            self.logger.info("NLTK data downloaded successfully")
        except Exception as e:
            self.logger.error(f"Error downloading NLTK data: {e}")
    
    def load_spacy_model(self):
        """Load spaCy model for text processing"""
        try:
            nlp = spacy.load("en_core_web_sm")
            self.logger.info("spaCy model loaded successfully")
            return nlp
        except OSError:
            self.logger.warning("spaCy model not found. Please install: python -m spacy download en_core_web_sm")
            return None
    
    def run_pipeline(self, max_pages: int = 50, delay: float = 2.0):
        """
        Run the complete pipeline
        
        Args:
            max_pages: Maximum number of pages to scrape
            delay: Delay between requests in seconds
        """
        self.logger.info("Starting MOSDAC data pipeline")
        
        # Phase 1: Web Scraping
        self.logger.info("Phase 1: Starting web scraping")
        scraped_data = self.scraper.scrape_portal(max_pages=max_pages, delay=delay)
        
        # Phase 2: Document Parsing
        self.logger.info("Phase 2: Starting document parsing")
        parsed_data = self.parser.parse_documents()
        
        # Phase 3: Text Cleaning and Processing
        self.logger.info("Phase 3: Starting text cleaning and processing")
        cleaned_data = self.cleaner.clean_and_process(scraped_data + parsed_data)
        
        # Save final output
        self.save_final_output(cleaned_data)
        
        self.logger.info(f"Pipeline completed successfully. Processed {len(cleaned_data)} items")
        return cleaned_data
    
    def save_final_output(self, data: List[Dict]):
        """Save final processed data"""
        # Save individual JSON files
        for i, item in enumerate(data):
            filename = f"item_{i:04d}_{hashlib.md5(str(item).encode()).hexdigest()[:8]}.json"
            filepath = self.output_dir / "cleaned_json" / filename
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(item, f, ensure_ascii=False, indent=2)
        
        # Save master JSONL file
        master_file = self.output_dir / "all_cleaned.jsonl"
        with open(master_file, 'w', encoding='utf-8') as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        
        self.logger.info(f"Saved {len(data)} items to cleaned_json/ and all_cleaned.jsonl")


class WebScraper:
    """
    Web scraping component for static and dynamic content
    """
    
    def __init__(self, base_url: str, output_dir: Path, logger):
        self.base_url = base_url
        self.output_dir = output_dir
        self.logger = logger
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        
        # Setup Selenium driver
        self.driver = None
        self.setup_selenium()
        
        # Track visited URLs
        self.visited_urls = set()
        self.failed_urls = []
    
    def setup_selenium(self):
        """Setup Selenium WebDriver for dynamic content"""
        try:
            chrome_options = Options()
            chrome_options.add_argument("--headless")
            chrome_options.add_argument("--no-sandbox")
            chrome_options.add_argument("--disable-dev-shm-usage")
            chrome_options.add_argument("--disable-gpu")
            chrome_options.add_argument("--window-size=1920,1080")
            
            self.driver = webdriver.Chrome(options=chrome_options)
            self.logger.info("Selenium WebDriver initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize Selenium: {e}")
            self.driver = None
    
    def check_robots_txt(self):
        """Check and respect robots.txt"""
        try:
            robots_url = urljoin(self.base_url, '/robots.txt')
            response = self.session.get(robots_url)
            
            if response.status_code == 200:
                self.logger.info("robots.txt found and checked")
                return response.text
            else:
                self.logger.info("No robots.txt found")
                return None
        except Exception as e:
            self.logger.error(f"Error checking robots.txt: {e}")
            return None
    
    def scrape_static_content(self, url: str) -> Optional[Dict]:
        """
        Scrape static content using requests and BeautifulSoup
        
        Args:
            url: URL to scrape
            
        Returns:
            Dictionary containing scraped data
        """
        try:
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extract content
            data = {
                'url': url,
                'title': self.extract_title(soup),
                'content': self.extract_content(soup),
                'meta_tags': self.extract_meta_tags(soup),
                'links': self.extract_links(soup, url),
                'download_links': self.extract_download_links(soup, url),
                'scraped_at': datetime.now().isoformat(),
                'type': 'static'
            }
            
            # Save raw HTML
            self.save_raw_html(url, response.text)
            
            self.logger.info(f"Successfully scraped static content from {url}")
            return data
            
        except Exception as e:
            self.logger.error(f"Error scraping static content from {url}: {e}")
            self.failed_urls.append(url)
            return None
    
    def scrape_dynamic_content(self, url: str) -> Optional[Dict]:
        """
        Scrape dynamic content using Selenium
        
        Args:
            url: URL to scrape
            
        Returns:
            Dictionary containing scraped data
        """
        if not self.driver:
            self.logger.error("Selenium driver not available for dynamic scraping")
            return None
        
        try:
            self.driver.get(url)
            
            # Wait for page to load
            WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.TAG_NAME, "body"))
            )
            
            # Additional wait for dynamic content
            time.sleep(3)
            
            # Get page source after JavaScript execution
            page_source = self.driver.page_source
            soup = BeautifulSoup(page_source, 'html.parser')
            
            # Extract content
            data = {
                'url': url,
                'title': self.extract_title(soup),
                'content': self.extract_content(soup),
                'meta_tags': self.extract_meta_tags(soup),
                'links': self.extract_links(soup, url),
                'download_links': self.extract_download_links(soup, url),
                'scraped_at': datetime.now().isoformat(),
                'type': 'dynamic'
            }
            
            # Save raw HTML
            self.save_raw_html(url, page_source)
            
            self.logger.info(f"Successfully scraped dynamic content from {url}")
            return data
            
        except Exception as e:
            self.logger.error(f"Error scraping dynamic content from {url}: {e}")
            self.failed_urls.append(url)
            return None
    
    def extract_title(self, soup: BeautifulSoup) -> str:
        """Extract page title"""
        title_tag = soup.find('title')
        if title_tag:
            return title_tag.get_text().strip()
        
        h1_tag = soup.find('h1')
        if h1_tag:
            return h1_tag.get_text().strip()
        
        return "No title found"
    
    def extract_content(self, soup: BeautifulSoup) -> str:
        """Extract main content from page"""
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()
        
        # Try to find main content areas
        content_selectors = [
            'main', 'article', '.content', '#content', 
            '.main-content', '#main-content', '.post-content'
        ]
        
        for selector in content_selectors:
            content = soup.select_one(selector)
            if content:
                return content.get_text(separator=' ', strip=True)
        
        # Fall back to body content
        body = soup.find('body')
        if body:
            return body.get_text(separator=' ', strip=True)
        
        return soup.get_text(separator=' ', strip=True)
    
    def extract_meta_tags(self, soup: BeautifulSoup) -> Dict:
        """Extract meta tags"""
        meta_tags = {}
        
        for meta in soup.find_all('meta'):
            if not isinstance(meta, Tag):
                continue
            name = meta.get('name')
            prop = meta.get('property')
            content = meta.get('content', '')
            if name:
                meta_tags[name] = content
            elif prop:
                meta_tags[prop] = content
        
        return meta_tags
    
    def extract_links(self, soup: BeautifulSoup, base_url: str) -> List[str]:
        """Extract all links from page"""
        links = []
        
        for link in soup.find_all('a', href=True):
            if not isinstance(link, Tag):
                continue
            href = link.get('href', '')
            if not isinstance(href, str):
                continue
            full_url = urljoin(base_url, href)
            links.append(full_url)
        
        return list(set(links))  # Remove duplicates
    
    def extract_download_links(self, soup: BeautifulSoup, base_url: str) -> List[Dict]:
        """Extract download links for documents"""
        download_links = []
        
        # Common document extensions
        doc_extensions = ['.pdf', '.docx', '.doc', '.xlsx', '.xls', '.ppt', '.pptx']
        
        for link in soup.find_all('a', href=True):
            if not isinstance(link, Tag):
                continue
            href = link.get('href', '')
            if not isinstance(href, str):
                continue
            full_url = urljoin(base_url, href)
            
            # Check if it's a document link
            if any(ext in href.lower() for ext in doc_extensions):
                download_links.append({
                    'url': full_url,
                    'text': link.get_text().strip(),
                    'type': self.get_file_type(href)
                })
        
        return download_links
    
    def get_file_type(self, url: str) -> str:
        """Determine file type from URL"""
        url_lower = url.lower()
        
        if '.pdf' in url_lower:
            return 'pdf'
        elif '.docx' in url_lower or '.doc' in url_lower:
            return 'docx'
        elif '.xlsx' in url_lower or '.xls' in url_lower:
            return 'xlsx'
        elif '.ppt' in url_lower or '.pptx' in url_lower:
            return 'ppt'
        else:
            return 'unknown'
    
    def save_raw_html(self, url: str, html_content: str):
        """Save raw HTML content"""
        # Create safe filename
        safe_filename = re.sub(r'[^\w\-_.]', '_', url.replace('https://', '').replace('http://', ''))
        filename = f"{safe_filename[:100]}.html"
        filepath = self.output_dir / "raw_html" / filename
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(html_content)
    
    def download_document(self, url: str, doc_type: str) -> Optional[str]:
        """Download document from URL"""
        try:
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            
            # Create safe filename
            parsed_url = urlparse(url)
            filename = os.path.basename(parsed_url.path)
            if not filename:
                filename = f"document_{hashlib.md5(url.encode()).hexdigest()[:8]}.{doc_type}"
            
            filepath = self.output_dir / "raw_docs" / filename
            
            with open(filepath, 'wb') as f:
                f.write(response.content)
            
            self.logger.info(f"Downloaded document: {filename}")
            return str(filepath)
            
        except Exception as e:
            self.logger.error(f"Error downloading document from {url}: {e}")
            return None
    
    def scrape_portal(self, max_pages: int = 50, delay: float = 2.0) -> List[Dict]:
        """
        Main scraping method for MOSDAC portal
        
        Args:
            max_pages: Maximum number of pages to scrape
            delay: Delay between requests
            
        Returns:
            List of scraped data
        """
        self.logger.info("Starting MOSDAC portal scraping")
        
        # Check robots.txt
        self.check_robots_txt()
        
        # Define important pages to scrape
        important_pages = [
            "/",
            "/product",
            "/data",
            "/satellite",
            "/faq",
            "/documentation",
            "/about",
            "/contact"
        ]
        
        all_data = []
        pages_scraped = 0
        
        # Start with important pages
        for page in important_pages:
            if pages_scraped >= max_pages:
                break
                
            url = urljoin(self.base_url, page)
            
            if url in self.visited_urls:
                continue
                
            self.visited_urls.add(url)
            
            # Try static scraping first
            data = self.scrape_static_content(url)
            
            # If static fails, try dynamic
            if not data:
                data = self.scrape_dynamic_content(url)
            
            if data:
                all_data.append(data)
                
                # Download any documents found
                for doc_link in data.get('download_links', []):
                    doc_path = self.download_document(doc_link['url'], doc_link['type'])
                    if doc_path:
                        doc_link['local_path'] = doc_path
                
                pages_scraped += 1
                
                # Follow internal links
                for link in data.get('links', []):
                    if (pages_scraped >= max_pages or 
                        link in self.visited_urls or 
                        not link.startswith(self.base_url)):
                        continue
                    
                    self.visited_urls.add(link)
                    
                    # Scrape linked page
                    link_data = self.scrape_static_content(link)
                    if not link_data:
                        link_data = self.scrape_dynamic_content(link)
                    
                    if link_data:
                        all_data.append(link_data)
                        pages_scraped += 1
                        
                        # Download documents from linked pages
                        for doc_link in link_data.get('download_links', []):
                            doc_path = self.download_document(doc_link['url'], doc_link['type'])
                            if doc_path:
                                doc_link['local_path'] = doc_path
            
            # Polite delay
            time.sleep(delay)
        
        self.logger.info(f"Scraping completed. Scraped {pages_scraped} pages")
        
        # Cleanup
        if self.driver:
            self.driver.quit()
        
        return all_data


class DocumentParser:
    """
    Document parsing component for PDF, DOCX, and XLSX files
    """
    
    def __init__(self, output_dir: Path, logger):
        self.output_dir = output_dir
        self.logger = logger
        self.raw_docs_dir = output_dir / "raw_docs"
        self.parsed_docs_dir = output_dir / "parsed_docs"
    
    def parse_documents(self) -> List[Dict]:
        """Parse all documents in raw_docs directory"""
        self.logger.info("Starting document parsing")
        
        parsed_data = []
        
        # Get all files in raw_docs directory
        if not self.raw_docs_dir.exists():
            self.logger.warning("raw_docs directory not found")
            return parsed_data
        
        files = list(self.raw_docs_dir.glob("*"))
        
        for file_path in tqdm(files, desc="Parsing documents"):
            try:
                file_ext = file_path.suffix.lower()
                
                if file_ext == '.pdf':
                    data = self.parse_pdf(file_path)
                elif file_ext in ['.docx', '.doc']:
                    data = self.parse_docx(file_path)
                elif file_ext in ['.xlsx', '.xls']:
                    data = self.parse_xlsx(file_path)
                else:
                    self.logger.warning(f"Unsupported file type: {file_ext}")
                    continue
                
                if data:
                    parsed_data.append(data)
                    self.save_parsed_data(data, file_path.stem)
                    
            except Exception as e:
                self.logger.error(f"Error parsing {file_path}: {e}")
        
        self.logger.info(f"Parsed {len(parsed_data)} documents")
        return parsed_data
    
    def parse_pdf(self, file_path: Path) -> Optional[Dict]:
        """Parse PDF file"""
        try:
            text_content = ""
            metadata = {}
            
            # Try pdfplumber first
            try:
                with pdfplumber.open(file_path) as pdf:
                    metadata = pdf.metadata or {}
                    
                    for page in pdf.pages:
                        page_text = page.extract_text()
                        if page_text:
                            text_content += page_text + "\n"
            except Exception as e:
                self.logger.warning(f"pdfplumber failed for {file_path}, trying PyMuPDF: {e}")
                
                # Fallback to PyMuPDF
                doc = fitz.open(file_path)
                metadata = doc.metadata
                
                for page in doc:
                    text_content += page.get_text() + "\n"  # type: ignore
                
                doc.close()
            
            return {
                'title': metadata.get('title', file_path.stem) if metadata else file_path.stem,
                'content': text_content,
                'metadata': metadata,
                'file_path': str(file_path),
                'file_type': 'pdf',
                'parsed_at': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error parsing PDF {file_path}: {e}")
            return None
    
    def parse_docx(self, file_path: Path) -> Optional[Dict]:
        """Parse DOCX file"""
        try:
            doc = Document(str(file_path))
            
            # Extract text
            text_content = ""
            for paragraph in doc.paragraphs:
                text_content += paragraph.text + "\n"
            
            # Extract metadata
            metadata = {}
            core_props = doc.core_properties
            
            if core_props:
                metadata = {
                    'title': core_props.title,
                    'author': core_props.author,
                    'subject': core_props.subject,
                    'created': str(core_props.created) if core_props.created else None,
                    'modified': str(core_props.modified) if core_props.modified else None
                }
            
            return {
                'title': metadata.get('title', file_path.stem),
                'content': text_content,
                'metadata': metadata,
                'file_path': str(file_path),
                'file_type': 'docx',
                'parsed_at': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error parsing DOCX {file_path}: {e}")
            return None
    
    def parse_xlsx(self, file_path: Path) -> Optional[Dict]:
        """Parse XLSX file"""
        try:
            # Read all sheets
            xlsx_data = pd.read_excel(file_path, sheet_name=None)
            
            text_content = ""
            sheets_info = []
            
            for sheet_name, df in xlsx_data.items():
                # Convert DataFrame to text
                sheet_text = f"Sheet: {sheet_name}\n"
                sheet_text += df.to_string(index=False) + "\n\n"
                text_content += sheet_text
                
                sheets_info.append({
                    'name': sheet_name,
                    'shape': df.shape,
                    'columns': list(df.columns)
                })
            
            metadata = {
                'sheets': sheets_info,
                'total_sheets': len(xlsx_data)
            }
            
            return {
                'title': file_path.stem,
                'content': text_content,
                'metadata': metadata,
                'file_path': str(file_path),
                'file_type': 'xlsx',
                'parsed_at': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error parsing XLSX {file_path}: {e}")
            return None
    
    def save_parsed_data(self, data: Dict, filename: str):
        """Save parsed data to JSON file"""
        output_file = self.parsed_docs_dir / f"{filename}_parsed.json"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2, default=str)


class TextCleaner:
    """
    Text cleaning and preprocessing component
    """
    
    def __init__(self, output_dir: Path, logger):
        self.output_dir = output_dir
        self.logger = logger
        
        # Load stopwords
        try:
            self.stop_words = set(stopwords.words('english'))
        except:
            self.stop_words = set()
            self.logger.warning("English stopwords not available")
    
    def clean_and_process(self, data_list: List[Dict]) -> List[Dict]:
        """
        Clean and process all text data
        
        Args:
            data_list: List of data dictionaries
            
        Returns:
            List of cleaned data dictionaries
        """
        self.logger.info("Starting text cleaning and processing")
        
        cleaned_data = []
        
        for item in tqdm(data_list, desc="Cleaning text"):
            try:
                cleaned_item = self.clean_item(item)
                if cleaned_item:
                    cleaned_data.append(cleaned_item)
            except Exception as e:
                self.logger.error(f"Error cleaning item: {e}")
        
        self.logger.info(f"Cleaned {len(cleaned_data)} items")
        return cleaned_data
    
    def clean_item(self, item: Dict) -> Optional[Dict]:
        """Clean individual item"""
        try:
            # Extract text content
            text = item.get('content', '')
            
            if not text or len(text.strip()) < 10:
                return None
            
            # Clean text
            cleaned_text = self.clean_text(text)
            
            if not cleaned_text or len(cleaned_text.strip()) < 10:
                return None
            
            # Detect language
            try:
                lang = detect(cleaned_text)
                if lang != 'en':
                    self.logger.info(f"Skipping non-English text: {lang}")
                    return None
            except:
                pass  # If language detection fails, keep the text
            
            # Extract metadata
            metadata = self.extract_metadata(item)
            
            # Create cleaned item
            cleaned_item = {
                'title': self.clean_text(item.get('title', 'No title')),
                'type': self.determine_type(item),
                'section': self.determine_section(item),
                'text': cleaned_text,
                'source_url': item.get('url', item.get('file_path', '')),
                'document_type': item.get('file_type', item.get('type', 'web')),
                'meta': metadata
            }
            
            return cleaned_item
            
        except Exception as e:
            self.logger.error(f"Error cleaning item: {e}")
            return None
    
    def clean_text(self, text: str) -> str:
        """
        Clean and normalize text
        
        Args:
            text: Raw text to clean
            
        Returns:
            Cleaned text
        """
        if not text:
            return ""
        
        # Fix encoding issues
        text = ftfy.fix_text(text)
        
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', ' ', text)
        
        # Remove JavaScript
        text = re.sub(r'<script.*?</script>', ' ', text, flags=re.DOTALL | re.IGNORECASE)
        
        # Remove CSS
        text = re.sub(r'<style.*?</style>', ' ', text, flags=re.DOTALL | re.IGNORECASE)
        
        # Normalize unicode
        text = unicodedata.normalize('NFKD', text)
        
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s.,!?;:()\-\[\]{}"\'/]', ' ', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove short lines (likely noise)
        lines = text.split('\n')
        cleaned_lines = [line.strip() for line in lines if len(line.strip()) > 3]
        text = '\n'.join(cleaned_lines)
        
        return text.strip()
    
    def determine_type(self, item: Dict) -> str:
        """Determine content type"""
        file_type = item.get('file_type', item.get('type', ''))
        
        if file_type in ['pdf', 'docx', 'xlsx']:
            return 'documentation'
        elif 'faq' in str(item.get('url', '')).lower():
            return 'faq'
        elif 'product' in str(item.get('url', '')).lower():
            return 'product'
        else:
            return 'general'
    
    def determine_section(self, item: Dict) -> str:
        """Determine content section"""
        url = str(item.get('url', item.get('file_path', ''))).lower()
        title = str(item.get('title', '')).lower()
        
        # Define section keywords
        sections = {
            'Climate Monitoring': ['climate', 'weather', 'temperature', 'rainfall', 'precipitation'],
            'Satellite Data': ['satellite', 'insat', 'goes', 'meteosat', 'imagery'],
            'Ocean Monitoring': ['ocean', 'sea', 'marine', 'coastal', 'sst'],
            'Land Surface': ['land', 'vegetation', 'ndvi', 'surface', 'soil'],
            'Atmospheric': ['atmosphere', 'aerosol', 'ozone', 'winds', 'pressure'],
            'Cyclone': ['cyclone', 'hurricane', 'typhoon', 'storm', 'tropical'],
            'Products': ['product', 'dataset', 'data', 'download'],
            'Documentation': ['guide', 'manual', 'documentation', 'help', 'tutorial']
        }
        
        # Check for section keywords
        for section, keywords in sections.items():
            if any(keyword in url or keyword in title for keyword in keywords):
                return section
        
        return 'General'
    
    def extract_metadata(self, item: Dict) -> Dict:
        """Extract and clean metadata"""
        meta = {}
        
        # Extract tags from title and content
        text = (item.get('title', '') + ' ' + item.get('content', '')).lower()
        
        # Define common tags
        tag_keywords = {
            'rainfall': ['rain', 'precipitation', 'rainfall'],
            'temperature': ['temperature', 'temp', 'thermal'],
            'satellite': ['satellite', 'insat', 'goes'],
            'ocean': ['ocean', 'sea', 'sst'],
            'climate': ['climate', 'weather'],
            'imagery': ['image', 'imagery', 'visual'],
            'data': ['data', 'dataset', 'product'],
            'monitoring': ['monitoring', 'observation', 'analysis']
        }
        
        tags = []
        for tag, keywords in tag_keywords.items():
            if any(keyword in text for keyword in keywords):
                tags.append(tag)
        
        meta['tags'] = tags
        meta['geolocation'] = 'India'  # MOSDAC focuses on India
        
        # Extract date if available
        if 'metadata' in item and item['metadata']:
            item_meta = item['metadata']
            if 'created' in item_meta:
                meta['date_published'] = item_meta['created']
            elif 'modified' in item_meta:
                meta['date_published'] = item_meta['modified']
        
        # Add processing date
        meta['processed_at'] = datetime.now().isoformat()
        
        return meta


class ConfigManager:
    """
    Configuration manager for the pipeline
    """
    
    def __init__(self):
        self.config = {
            'scraping': {
                'max_pages': 50,
                'delay': 2.0,
                'timeout': 10,
                'max_retries': 3
            },
            'parsing': {
                'min_text_length': 10,
                'max_file_size_mb': 50
            },
            'cleaning': {
                'min_cleaned_length': 10,
                'remove_non_english': True,
                'remove_stopwords': False
            },
            'output': {
                'save_individual_files': True,
                'save_master_file': True,
                'pretty_print_json': True
            }
        }
    
    def load_config(self, config_file: str):
        """Load configuration from file"""
        try:
            with open(config_file, 'r') as f:
                custom_config = json.load(f)
            
            # Update config with custom values
            self.update_config(self.config, custom_config)
            
        except FileNotFoundError:
            print(f"Config file {config_file} not found. Using default configuration.")
        except Exception as e:
            print(f"Error loading config: {e}. Using default configuration.")
    
    def update_config(self, base_config: Dict, update_config: Dict):
        """Recursively update configuration"""
        for key, value in update_config.items():
            if key in base_config and isinstance(base_config[key], dict) and isinstance(value, dict):
                self.update_config(base_config[key], value)
            else:
                base_config[key] = value
    
    def save_config(self, config_file: str):
        """Save current configuration to file"""
        with open(config_file, 'w') as f:
            json.dump(self.config, f, indent=2)


def setup_environment():
    """
    Setup required environment and dependencies
    """
    print("Setting up MOSDAC Pipeline environment...")
    
    # Required packages
    required_packages = [
        'requests',
        'beautifulsoup4',
        'selenium',
        'pandas',
        'pdfplumber',
        'PyMuPDF',
        'python-docx',
        'openpyxl',
        'nltk',
        'spacy',
        'langdetect',
        'ftfy',
        'tqdm'
    ]
    
    print("Required packages:")
    for package in required_packages:
        print(f"  - {package}")
    
    print("\nAdditional setup required:")
    print("1. Install Chrome/Chromium browser for Selenium")
    print("2. Download ChromeDriver and add to PATH")
    print("3. Install spaCy English model: python -m spacy download en_core_web_sm")
    print("4. Ensure sufficient disk space for downloaded documents")
    
    return True


def main():
    """
    Main function to run the MOSDAC pipeline
    """
    print("🚀 MOSDAC Data Extraction Pipeline")
    print("=" * 50)
    
    # Setup environment
    setup_environment()
    
    # Initialize configuration
    config_manager = ConfigManager()
    
    # Load custom config if exists
    config_file = "mosdac_config.json"
    if os.path.exists(config_file):
        config_manager.load_config(config_file)
    else:
        # Save default config
        config_manager.save_config(config_file)
        print(f"Created default configuration file: {config_file}")
    
    config = config_manager.config
    
    # Initialize pipeline
    try:
        pipeline = MOSDACPipeline(
            base_url="https://www.mosdac.gov.in",
            output_dir="mosdac_data"
        )
        
        # Run pipeline
        results = pipeline.run_pipeline(
            max_pages=config['scraping']['max_pages'],
            delay=config['scraping']['delay']
        )
        
        print(f"\n✅ Pipeline completed successfully!")
        print(f"📊 Processed {len(results)} items")
        print(f"📁 Output saved to: mosdac_data/")
        print(f"📋 Check logs at: mosdac_data/logs/scraping.log")
        
        # Print summary statistics
        print("\n📈 Summary Statistics:")
        
        types = {}
        sections = {}
        doc_types = {}
        
        for item in results:
            # Count by type
            item_type = item.get('type', 'unknown')
            types[item_type] = types.get(item_type, 0) + 1
            
            # Count by section
            section = item.get('section', 'unknown')
            sections[section] = sections.get(section, 0) + 1
            
            # Count by document type
            doc_type = item.get('document_type', 'unknown')
            doc_types[doc_type] = doc_types.get(doc_type, 0) + 1
        
        print(f"  Content Types: {types}")
        print(f"  Sections: {sections}")
        print(f"  Document Types: {doc_types}")
        
    except KeyboardInterrupt:
        print("\n⚠️  Pipeline interrupted by user")
    except Exception as e:
        print(f"\n❌ Pipeline failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
            