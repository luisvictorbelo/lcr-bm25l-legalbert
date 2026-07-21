import os
import json
import re
import subprocess
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from tqdm import tqdm

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

import spacy
from langdetect import detect, LangDetectException

# Ensure NLTK data is downloaded
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('tokenizers/punkt_tab')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt')
    nltk.download('punkt_tab')
    nltk.download('stopwords')

# Ensure SpaCy model is downloaded
try:
    nlp = spacy.load('en_core_web_sm')
except OSError:
    print("Baixando modelo do spaCy...")
    subprocess.run(['python3', '-m', 'spacy', 'download', 'en_core_web_sm'])
    nlp = spacy.load('en_core_web_sm')


@dataclass
class ProcessedParagraph:
    """Estrutura para armazenar paragrafo processado"""
    doc_id: str
    paragraph_id: str
    original_text: str
    cleaned_text: str
    tokens: List[str]
    processed_tokens: List[str]
    has_suppressed_marker: bool
    marker_types: List[str]
    is_french: bool

    def to_dict(self) -> Dict:
        return {
            'doc_id': self.doc_id,
            'paragraph_id': self.paragraph_id,
            'original_text': self.original_text,
            'cleaned_text': self.cleaned_text,
            'tokens': self.tokens,
            'processed_tokens': self.processed_tokens,
            'has_suppressed_marker': self.has_suppressed_marker,
            'marker_types': self.marker_types,
            'is_french': self.is_french
        }


class LegalDocumentPreprocessor:
    """
    Pre-processador especializado para documentos juridicos.
    Extrai e processa paragrafos com marcadores especiais.
    """

    def __init__(self, use_spacy: bool = True):
        self.use_spacy = use_spacy
        self.stemmer = PorterStemmer()
        self.stop_words = set(stopwords.words('english'))

        # Demarcadores especiais a buscar
        self.suppressed_markers = [
            'FRAGMENT_SUPPRESSED',
            'FRAGMENT_SUPRESSED',
            'CITATION_SUPPRESSED',
            'CITATION_SUPRESSED',
            'REFERENCE_SUPPRESSED',
            'REFERENCE_SUPRESSED',
        ]

        # Padrao para identificar numeros de paragrafos
        self.paragraph_pattern = r'\[(\d{1,4})\]'

    def is_french(self, text: str) -> bool:
        if not text or len(text.strip()) < 10:
            return False
        try:
            lang = detect(text)
            return lang == 'fr'
        except LangDetectException:
            return False

    def _normalize_text(self, document_text: str) -> str:
        processed = document_text.replace('\n', ' ').replace('•', ' ')
        processed = re.sub(r'\s+', ' ', processed).strip()
        return processed

    def extract_all_paragraphs(self, document_text: str) -> List[Tuple[str, str]]:
        processed = self._normalize_text(document_text)
        paragraph_matches = list(re.finditer(self.paragraph_pattern, processed))
        if not paragraph_matches:
            return []

        results: List[Tuple[str, str]] = []
        for i, match in enumerate(paragraph_matches):
            para_id = match.group(1)
            start = match.start()
            if i + 1 < len(paragraph_matches):
                end = paragraph_matches[i + 1].start()
            else:
                end = len(processed)
            para_text = processed[start:end].strip()
            results.append((para_id, para_text))
        return results

    def extract_paragraphs_with_markers(self, document_text: str) -> List[Tuple[str, str, str]]:
        processed = self._normalize_text(document_text)
        paragraph_matches = list(re.finditer(self.paragraph_pattern, processed))

        if not paragraph_matches:
            return []

        results = []
        markers_upper = [m.upper() for m in self.suppressed_markers]
        for i, match in enumerate(paragraph_matches):
            para_id = match.group(1)
            start = match.start()

            if i + 1 < len(paragraph_matches):
                end = paragraph_matches[i + 1].start()
            else:
                end = len(processed)

            para_text = processed[start:end].strip()
            para_upper = para_text.upper()
            has_marker = any(marker in para_upper for marker in markers_upper)

            if has_marker:
                subsequent_text = ""
                if i + 1 < len(paragraph_matches):
                    next_start = paragraph_matches[i + 1].start()
                    if i + 2 < len(paragraph_matches):
                        next_end = paragraph_matches[i + 2].start()
                    else:
                        next_end = len(processed)

                    subsequent_text = processed[next_start:next_end].strip()

                results.append((para_id, para_text, subsequent_text))

        return results

    def clean_paragraph(self, text: str) -> str:
        for marker in self.suppressed_markers:
            text = re.sub(rf'<{marker}>', ' ', text, flags=re.IGNORECASE)
            text = re.sub(marker, ' ', text, flags=re.IGNORECASE)

        text = re.sub(self.paragraph_pattern, ' ', text)
        text = re.sub(r'[<>]', ' ', text)
        text = re.sub(r'\[End of document\]', ' ', text, flags=re.IGNORECASE)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def tokenize_and_process(self, text: str) -> Tuple[List[str], List[str]]:
        if self.use_spacy:
            doc = nlp(text.lower())

            tokens = [
                token.text for token in doc
                if token.is_alpha and
                   token.text not in self.stop_words and
                   len(token.text) > 2
            ]

            lemmatized = [
                token.lemma_ for token in doc
                if token.is_alpha and
                   token.lemma_ not in self.stop_words and
                   len(token.lemma_) > 2
            ]

            return tokens, lemmatized
        else:
            tokens = word_tokenize(text.lower())
            tokens = [
                token for token in tokens
                if token.isalpha() and
                   token not in self.stop_words and
                   len(token) > 2
            ]
            stemmed = [self.stemmer.stem(token) for token in tokens]
            return tokens, stemmed

    def identify_markers(self, text: str) -> List[str]:
        found_markers = []
        upper_text = text.upper()
        for marker in self.suppressed_markers:
            if marker.upper() in upper_text or f'<{marker}>' in upper_text:
                found_markers.append(marker)
        return found_markers

    def process_document(self, doc_id: str, document_text: str, filter_french: bool = True, only_marked: bool = True) -> List[ProcessedParagraph]:
        if only_marked:
            paragraphs_with_markers = self.extract_paragraphs_with_markers(document_text)
        else:
            paragraphs_all = self.extract_all_paragraphs(document_text)
            paragraphs_with_markers = [(pid, ptext, "") for pid, ptext in paragraphs_all]

        processed_paragraphs = []

        for para_id, para_text, subsequent_text in paragraphs_with_markers:
            combined_text = f"{para_text} {subsequent_text}".strip()
            is_french_text = self.is_french(combined_text)
            if filter_french and is_french_text:
                continue

            cleaned = self.clean_paragraph(combined_text)
            if not cleaned or len(cleaned) < 10:
                continue

            tokens, processed_tokens = self.tokenize_and_process(cleaned)
            markers = self.identify_markers(para_text) if only_marked else []
            has_marker = len(markers) > 0
            processed_paragraphs.append(ProcessedParagraph(
                doc_id=doc_id, paragraph_id=para_id, original_text=combined_text,
                cleaned_text=cleaned, tokens=tokens, processed_tokens=processed_tokens,
                has_suppressed_marker=has_marker, marker_types=markers, is_french=is_french_text
            ))

        return processed_paragraphs


class PreprocessorPipeline:
    def __init__(self, preprocessor: LegalDocumentPreprocessor):
        self.preprocessor = preprocessor

    def _normalize_doc_id(self, doc_id: str) -> str:
        doc_id = doc_id.strip()
        if doc_id.lower().endswith('.txt'):
            return doc_id[:-4]
        return doc_id

    def process_and_save(self, documents_folder: str, labels_file: str, output_dir: str,
                         max_docs: Optional[int] = None, verbose: bool = True,
                         filter_french: bool = True):
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        query_cache = output_path / 'query_paragraphs_all.parquet'
        corpus_cache = output_path / 'corpus_paragraphs_all.parquet'

        if verbose:
            print(f"Lendo labels de: {labels_file}")
            
        with open(labels_file, 'r', encoding='utf-8') as f:
            labels = json.load(f)

        query_ids = {self._normalize_doc_id(k) for k in labels.keys()}
        noticed_ids = set()
        for _, values in labels.items():
            for value in values:
                noticed_ids.add(self._normalize_doc_id(value))

        if verbose:
            print(f"Query IDs: {len(query_ids)} | Corpus IDs: {len(noticed_ids)}")

        folder = Path(documents_folder)
        if not folder.exists():
            raise ValueError(f"Pasta '{documents_folder}' nao encontrada.")
            
        files = sorted(list(folder.glob('*.txt')))
        if max_docs:
            files = files[:max_docs]
            
        if verbose:
            print(f"Processando {len(files)} arquivos em '{documents_folder}'...")
            
        iterator = tqdm(files, desc="Processando documentos") if verbose else files

        query_paragraphs = []
        corpus_paragraphs = []
        
        for file_path in iterator:
            doc_id = file_path.stem
            doc_id_norm = self._normalize_doc_id(doc_id)
            
            if doc_id_norm not in query_ids and doc_id_norm not in noticed_ids:
                continue
                
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()

                # Process document (query)
                if doc_id_norm in query_ids:
                    paragraphs = self.preprocessor.process_document(
                        doc_id, content, filter_french=filter_french, only_marked=False
                    )
                    query_paragraphs.extend(paragraphs)

                # Process document (corpus)
                if doc_id_norm in noticed_ids:
                    paragraphs = self.preprocessor.process_document(
                        doc_id, content, filter_french=filter_french, only_marked=False
                    )
                    corpus_paragraphs.extend(paragraphs)
            except Exception as e:
                if verbose:
                    print(f"Falha ao processar {file_path}: {e}")

        if verbose:
            print(f"Salvando resultados em {output_dir} no formato Parquet...")
            
        if query_paragraphs:
            df_queries = pd.DataFrame([p.to_dict() for p in query_paragraphs])
            df_queries.to_parquet(query_cache, index=False)
            if verbose:
                print(f"Queries salvas: {len(df_queries)} paragrafos em {query_cache}")
        else:
            print("Nenhuma query encontrada.")
        
        if corpus_paragraphs:
            df_corpus = pd.DataFrame([p.to_dict() for p in corpus_paragraphs])
            df_corpus.to_parquet(corpus_cache, index=False)
            if verbose:
                print(f"Corpus salvo: {len(df_corpus)} paragrafos em {corpus_cache}")
        else:
            print("Nenhum corpus encontrado.")
            
        print("Pre-processamento concluido com sucesso!")


if __name__ == '__main__':
    # Updated paths based on user input
    DOCUMENTS_FOLDER = 'data/test-files/docs'
    LABELS_FILE = 'data/labels/task1_test_labels_2025.json'
    OUTPUT_DIR = 'data/test-files/processed'
    
    print(f"Pasta de documentos: {DOCUMENTS_FOLDER}")
    print(f"Arquivo de labels: {LABELS_FILE}")
    print(f"Pasta de saida: {OUTPUT_DIR}")
    
    preprocessor = LegalDocumentPreprocessor(use_spacy=False)
    pipeline = PreprocessorPipeline(preprocessor)
    
    pipeline.process_and_save(
        documents_folder=DOCUMENTS_FOLDER,
        labels_file=LABELS_FILE,
        output_dir=OUTPUT_DIR,
        max_docs=None,
        verbose=True,
        filter_french=True
    )
