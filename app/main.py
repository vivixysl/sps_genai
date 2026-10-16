from typing import Union
from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel
from app.bigram_model import BigramModel
import spacy
from app.image_classifier import ImageClassifier

app = FastAPI()

# Sample corpus for the bigram model
corpus = [
    "The Count of Monte Cristo is a novel written by Alexandre Dumas. \
It tells the story of Edmond Dantès, who is falsely imprisoned and later seeks revenge.",
    "this is another example sentence",
    "we are generating text based on bigram probabilities",
    "bigram models are simple but effective"
]

bigram_model = BigramModel(corpus)

# Load the spaCy large English model (same as Module 2 Practical 3)
nlp = spacy.load("en_core_web_lg")

# Assignment 2: CNN image classifier trained on CIFAR-10 (loaded once at startup)
classifier = ImageClassifier()


def calculate_embedding(input_word):
    word = nlp(input_word)
    return word.vector


class TextGenerationRequest(BaseModel):
    start_word: str
    length: int


class EmbeddingRequest(BaseModel):
    word: str


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.post("/generate")
def generate_text(request: TextGenerationRequest):
    generated_text = bigram_model.generate_text(request.start_word, request.length)
    return {"generated_text": generated_text}


@app.post("/embedding")
def get_embedding(request: EmbeddingRequest):
    embedding = calculate_embedding(request.word)
    return {
        "word": request.word,
        "dimension": len(embedding),
        "embedding": embedding.tolist(),  # numpy array -> list so it can be returned as JSON
    }


@app.post("/classify_image")
async def classify_image(file: UploadFile = File(...)):
    """Upload an image (jpg/png); returns its predicted CIFAR-10 class."""
    if not (file.content_type or "").startswith("image/"):
        raise HTTPException(status_code=400, detail="Please upload an image file.")
    image_bytes = await file.read()
    return classifier.predict(image_bytes)
