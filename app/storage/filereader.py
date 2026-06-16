from pypdf import PdfReader
import csv
import io
from database import Database

class FileReader:
    def __init__(self, file_paths):
        self.file_paths = file_paths

    def _read_files(self):
        try:
            all_sources = {}
            for file_path in self.file_paths:
                all_sources[file_path] = self._read_supported_file(file_path)
            return all_sources
        except FileNotFoundError:
            print(f"File not found: {self.file_path}")
            return None
        except Exception as e:
            print(f"An error occurred while reading the file: {e}")
            return None
    
    def _read_supported_file(self, file_path):
        lower_path = file_path.lower()
        if lower_path.split(".")[-1] not in ("pdf", "txt", "csv"):
            raise ValueError(f"Unsupported file type: {file_path}")
        
        return self._read(file_path)
    
    def _chunk_text(self, text, filename, chunk_size=300):
        chunks = {"QA": [],
                  "Documents": []}
        # quest-answer csv pipeline
        try:
            reader = csv.DictReader(io.StringIO(text.strip()))
            chunks["QA"].extend(list(reader))
        except Exception:
            # Document pipeline
            try:
                words = text.split()
                chunks = []
                for i in range(0, len(words), chunk_size):
                    chunk = " ".join(words[i:i + chunk_size])
                    chunks["Documents"].extend([{"text": chunk, "filename": filename}])
            except Exception as e:
                print(f"An error occurred while chunking the text from {filename}: {e}")
                raise e
        return chunks

    def _read(self, file_path):
        methods = {"pdf": self._read_pdf, 
                   "txt": self._read_txt, 
                   "csv": self._read_txt}
        lower_path = file_path.lower()
        file_type = lower_path.split(".")[-1]
        return methods[file_type](file_path)
    
    def _read_pdf(self, file_path):
        reader = PdfReader(file_path)
        text = ""

        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"

        return text
    
    def _read_txt(self, file_path):
        with open(file_path, "r", encoding="utf-8") as f: return f.read()

    def put_chunks_into_database(self, database=Database()):
        all_sources = self._read_files()

        if all_sources is not None:
            for source, text in all_sources.items():
                database.put_chunks(self._chunk_text(text, source))
