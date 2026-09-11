
from src.serving.inference import get_recommendations, recommender

import os
import sys
from contextlib import asynccontextmanager
from typing import List, Optional

from fastapi import FastAPI, HTTPException
import gradio as gr
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from groq import Groq


# Load API key from .env
load_dotenv()

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
groq_client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else Non

@asynccontextmanager
async def lifespan(app: FastAPI):
    yield


app = FastAPI(
    title="Philosophy RAG Engine",
    version="1.0.0",
    lifespan=lifespan
)


# --- Pydantic Models ---

class RecommendRequest(BaseModel):
    query: str = Field(..., min_length=3)
    top_k: int = Field(default=3, ge=1, le=10)
    school: Optional[str] = Field(default=None)


class BookRecommendation(BaseModel):
    id: int
    title: str
    author: str
    school: str
    summary: str
    similarity_score: float


class RecommendResponse(BaseModel):
    query: str
    results_count: int
    recommendations: List[BookRecommendation]


# --- FastAPI Routes ---

@app.get("/", tags=["Health"])
def health_check():
    return {
        "status": "healthy",
        "service": "philosophy-rag"
    }


@app.post("/recommend", response_model=RecommendResponse, tags=["API"])
def recommend_books(payload: RecommendRequest):
    try:
        results = get_recommendations(
            query=payload.query,
            top_k=payload.top_k,
            school=payload.school,
        )

        return RecommendResponse(
            query=payload.query,
            results_count=len(results),
            recommendations=results,
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


# --- RAG Pipeline (Gradio UI) ---

def rag_chat(query: str, top_k: int, school: str):
    if not query.strip():
        return (
            "Please enter a valid philosophical question.",
            ""
        )

    selected_school = None if school == "All" else school

    # 1. RETRIEVAL: Retrieve the most relevant books using ONNX embeddings
    retrieved_books = get_recommendations(
        query=query,
        top_k=int(top_k),
        school=selected_school
    )

    if not retrieved_books:
        return (
            "No relevant books were found. Try a different search query.",
            "No sources available."
        )

    # 2. AUGMENT: Build the context from retrieved books
    context_text = ""
    sources_markdown = "### Sources used:\n"

    for idx, book in enumerate(retrieved_books, 1):
        context_text += (
            f"Title: {book['title']}, "
            f"Author: {book['author']}, "
            f"Summary: {book['summary']}\n\n"
        )

        sources_markdown += (
            f"- **{book['title']}** by {book['author']} "
            f"*(Similarity: {book['similarity_score'] * 100:.1f}%)*\n"
        )

    prompt = f"""
    You are an expert philosophy assistant.
    Answer the user's question STRICTLY using the information provided
    in the Context below.

    If the information is not available in the context, clearly state
    that you cannot answer based on the books available in the database.

    Do not invent information.
    Be concise, thoughtful, and explain philosophical concepts clearly.

    Context (summaries retrieved from the indexed philosophy database):
    {context_text}

    User's question: {query}
    """

    # 3. GENERATION: Send the augmented prompt to the LLM
    system_instruction = """
    You are a university professor of philosophy: empathetic, erudite, and deeply reflective.
    Do not speak like a search engine and do not limit your response to cold lists or tables.
    
    Writing instructions:
    1. Respond directly, warmly, and thoughtfully to the user's existential dilemma.
    2. Integrate the ideas from the provided books into a coherent essay, showing how the authors' perspectives connect and complement or challenge one another.
    3. Provide an applied synthesis: explain how the user can use these ideas to gain clarity and navigate everyday life.
    4. Base your arguments STRICTLY on the provided context. Do not invent or introduce external theories or information.
    5. Respond concretely, don't just send the user to read the books.
    6. Don't make a different section for each book.
    7. Have shorter responses.
    8. Remember that talking about their situation is central, incorporating the book in your response is second. 
    """

    user_payload = f"""
    Context from the indexed philosophy library:
    {context_text}
    
    User's dilemma:
    {query}
    """
    if not groq_client:
        return (
            "Config error: GROQ_API_KEY missing.",
            sources_markdown,
        )
    completion = groq_client.chat.completions.create(
        model="openai/gpt-oss-20b",  # or the model currently configured
        messages=[
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": user_payload},
        ],
        temperature=0.6,
        max_completion_tokens=2048,
    )
    ai_response = completion.choices[0].message.content
    return ai_response, sources_markdown


# --- Gradio UI Layout ---

schools = [
    "All",
    "Absurdism",
    "Existentialism",
    "Christian Existentialism",
    "Nietzscheanism",
    "Philosophical Pessimism",
    "Egoist Anarchism",
    "Stoicism",
    "Political Realism",
    "Social Contract",
    "Political Enlightenment",
    "Marxism",
    "Classical Liberalism",
    "Confucianism",
    "Taoism"
]


def create_gradio_ui():
    schools = [
        "All",
        "Absurdism",
        "Existentialism",
        "Christian Existentialism",
        "Nietzscheanism",
        "Philosophical Pessimism",
        "Egoist Anarchism",
        "Stoicism",
        "Political Realism",
        "Social Contract",
        "Political Enlightenment",
        "Marxism",
        "Classical Liberalism",
        "Confucianism",
        "Taoism",
    ]

    with gr.Blocks(title="Philosophy RAG Engine") as demo:
        gr.Markdown("# 🏛️ Philosophy RAG Assistant")
        gr.Markdown(
            "Ask any philosophical question."
        )

        with gr.Row():
            with gr.Column(scale=2):
                query_input = gr.Textbox(
                    lines=2,
                    placeholder="e.g., How to handle grief and things beyond control?",
                    label="Your quesiton",
                )
                with gr.Row():
                    k_slider = gr.Slider(
                        minimum=1,
                        maximum=5,
                        value=3,
                        step=1,
                        label="Analized books (Top K)",
                    )
                    school_dropdown = gr.Dropdown(
                        choices=schools, value="All", label="Philosophical school"
                    )
                submit_btn = gr.Button(
                    "Ask the AI philosopher", variant="primary"
                )

            with gr.Column(scale=3):
                ai_output = gr.Markdown(label="AI response")
                gr.Markdown("---")
                sources_output = gr.Markdown(label="Cited sources")

        submit_btn.click(
            fn=rag_chat,
            inputs=[query_input, k_slider, school_dropdown],
            outputs=[ai_output, sources_output],
        )

    return demo

if "pytest" not in sys.modules:
    gradio_app = create_gradio_ui()
    app = gr.mount_gradio_app(app, gradio_app, path="/ui")