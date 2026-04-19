from fastapi import FastAPI, HTTPException, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, HttpUrl
from typing import List, Optional
import os
import logging
import math
import re
from collections import Counter
from dotenv import load_dotenv
import PyPDF2
import requests
import io
from langchain_groq import ChatGroq

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="DocLens",
    description="API for processing PDF documents and answering questions using RAG",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------- Pydantic Models ----------

class QueryRequest(BaseModel):
    documents: HttpUrl
    questions: List[str]

class QueryResponse(BaseModel):
    answers: List[str]

class ErrorResponse(BaseModel):
    error: str
    detail: Optional[str] = None


# ---------- Lightweight TF-IDF Retriever (replaces FAISS + sentence-transformers) ----------

class TFIDFRetriever:
    def __init__(self, text: str, chunk_size: int = 2000, chunk_overlap: int = 200):
        self.chunks = self._chunk_text(text, chunk_size, chunk_overlap)
        self.tfidf_matrix, self.vocab = self._build_tfidf()
        logger.info(f"Built TF-IDF index with {len(self.chunks)} chunks")

    def _chunk_text(self, text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
        if len(text) <= chunk_size:
            return [text]
        chunks = []
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]
            if end < len(text):
                boundary = max(chunk.rfind('.'), chunk.rfind('\n'))
                if boundary > start + chunk_size // 2:
                    chunk = text[start:start + boundary + 1]
                    end = start + boundary + 1
            chunks.append(chunk.strip())
            start = end - chunk_overlap
            if start >= len(text):
                break
        return chunks

    def _tokenize(self, text: str) -> List[str]:
        return re.findall(r'\b\w+\b', text.lower())

    def _build_tfidf(self):
        tokenized = [self._tokenize(c) for c in self.chunks]
        vocab = list({tok for doc in tokenized for tok in doc})
        vocab_index = {w: i for i, w in enumerate(vocab)}
        N = len(self.chunks)

        # Document frequency
        df = Counter()
        for doc in tokenized:
            for tok in set(doc):
                df[tok] += 1

        # TF-IDF matrix as list of dicts
        matrix = []
        for doc in tokenized:
            tf = Counter(doc)
            total = len(doc) or 1
            vec = {}
            for tok, count in tf.items():
                tfidf = (count / total) * math.log((N + 1) / (df[tok] + 1))
                vec[vocab_index[tok]] = tfidf
            matrix.append(vec)

        return matrix, vocab_index

    def _cosine(self, vec_a: dict, vec_b: dict) -> float:
        dot = sum(vec_a.get(k, 0) * v for k, v in vec_b.items())
        norm_a = math.sqrt(sum(v ** 2 for v in vec_a.values())) or 1
        norm_b = math.sqrt(sum(v ** 2 for v in vec_b.values())) or 1
        return dot / (norm_a * norm_b)

    def retrieve(self, query: str, top_k: int = 5) -> List[tuple]:
        tokenized_q = self._tokenize(query)
        query_vec = {}
        for tok in tokenized_q:
            if tok in self.vocab:
                idx = self.vocab[tok]
                query_vec[idx] = query_vec.get(idx, 0) + 1

        scores = [
            (self.chunks[i], self._cosine(doc_vec, query_vec))
            for i, doc_vec in enumerate(self.tfidf_matrix)
        ]
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]


# ---------- Helper Functions ----------

def extract_pdf_from_url(url: str) -> str:
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        pdf_file = io.BytesIO(response.content)
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
        return text
    except Exception as e:
        logger.error(f"Error extracting PDF: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Failed to extract PDF: {str(e)}")


def process_pdf_queries(pdf_url: str, questions: List[str], groq_api_key: str) -> List[str]:
    llm = ChatGroq(
        groq_api_key=groq_api_key,
        model_name="llama-3.1-8b-instant",
        temperature=0.1,
        max_tokens=300
    )

    result = extract_pdf_from_url(pdf_url)
    if not result:
        raise HTTPException(status_code=400, detail="Failed to extract PDF content")

    retriever = TFIDFRetriever(result)

    system_prompt = """You are an expert AI assistant specializing in intelligent document analysis for insurance, legal, HR, and compliance domains.

Response Guidelines:
1. Base all responses strictly on retrieved document content. Never hallucinate.
2. Provide a direct, concise answer to the query.
3. Mention specific conditions, limits, or exceptions when present.
4. If information is not found in the context, say "Information not found in document."
5. Give output in plain text, 1–3 sentences max unless complexity requires more.
"""

    examples = """Examples:
Q: What is the grace period for premium payment?
A: 30 days grace period after due date.

Q: What is the waiting period for pre-existing diseases?
A: 36 months continuous coverage required.

Q: Does this policy cover maternity expenses?
A: Yes, after 24 months continuous coverage. Limited to 2 deliveries per policy period.
"""

    answers = []
    for query in questions:
        relevant_chunks = retriever.retrieve(query, top_k=5)
        context = "\n\n".join([chunk for chunk, score in relevant_chunks])

        prompt = f"""{system_prompt}

{examples}

Context from document:
{context}

Q: {query}
A:"""

        response = llm.invoke(prompt)
        answer = response.content.strip()

        # Strip any Q:/A: prefixes the model might echo back
        if answer.startswith("Q:"):
            a_index = answer.find("A:")
            if a_index != -1:
                answer = answer[a_index + 2:].strip()
        elif answer.startswith("A:"):
            answer = answer[2:].strip()

        answers.append(answer)

    return answers


# ---------- Endpoints ----------

@app.get("/")
async def root():
    return {"status": "healthy", "message": "DocLens is running"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "message": "DocLens is running"}

@app.post("/DocLens", response_model=QueryResponse)
async def process_document_queries(request: QueryRequest):
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        raise HTTPException(status_code=500, detail="GROQ_API_KEY not configured")

    if not request.questions:
        raise HTTPException(status_code=400, detail="At least one question is required")

    if len(request.questions) > 20:
        raise HTTPException(status_code=400, detail="Maximum 20 questions per request")

    logger.info(f"Processing {len(request.questions)} questions for: {request.documents}")

    answers = process_pdf_queries(
        pdf_url=str(request.documents),
        questions=request.questions,
        groq_api_key=groq_api_key
    )

    return QueryResponse(answers=answers)


# ---------- Error Handlers ----------

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail, "status_code": exc.status_code}
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "detail": str(exc)}
    )


if __name__ == "__main__":
    import uvicorn
    if not os.getenv("GROQ_API_KEY"):
        logger.error("GROQ_API_KEY environment variable is required")
        exit(1)
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True, log_level="info")
