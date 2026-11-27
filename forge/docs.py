"""Load a document folder and split it into chunks that remember their source."""
import csv, os, re
from dataclasses import dataclass


@dataclass
class Doc:
    name: str
    path: str
    kind: str   # pdf | text | csv
    text: str


@dataclass
class Chunk:
    id: str
    source: str
    text: str


def load_folder(folder):
    docs = []
    for fn in sorted(os.listdir(folder)):
        p = os.path.join(folder, fn)
        ext = fn.lower().rsplit(".", 1)[-1] if "." in fn else ""
        if ext in ("txt", "md"):
            docs.append(Doc(fn, p, "text", open(p, encoding="utf-8", errors="ignore").read()))
        elif ext == "pdf":
            from pypdf import PdfReader
            txt = "\n".join((pg.extract_text() or "") for pg in PdfReader(p).pages)
            docs.append(Doc(fn, p, "pdf", txt))
        elif ext == "csv":
            with open(p, newline="", encoding="utf-8", errors="ignore") as f:
                rows = list(csv.DictReader(f))
            # one line per row so each row stays retrievable as a unit
            txt = "\n".join("; ".join(f"{k}: {v}" for k, v in r.items()) for r in rows)
            docs.append(Doc(fn, p, "csv", txt))
    return docs


def chunk_doc(doc, size=200, overlap=40):
    """Word-based chunking. CSVs are chunked by row groups so rows are never cut in half."""
    chunks = []
    if doc.kind == "csv":
        lines = [l for l in doc.text.split("\n") if l.strip()]
        step = max(1, size // 40)
        for i in range(0, len(lines), step):
            chunks.append(Chunk(f"{doc.name}#{len(chunks)}", doc.name, "\n".join(lines[i:i + step])))
        return chunks
    words = re.sub(r"[ \t]+", " ", doc.text).split()
    step = max(1, size - overlap)
    for i in range(0, max(len(words), 1), step):
        piece = " ".join(words[i:i + size])
        if piece.strip():
            chunks.append(Chunk(f"{doc.name}#{len(chunks)}", doc.name, piece))
        if i + size >= len(words):
            break
    return chunks


def chunk_all(docs, size=200, overlap=40):
    out = []
    for d in docs:
        out.extend(chunk_doc(d, size, overlap))
    return out
