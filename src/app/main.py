print("MAIN START")

from src.serving.inference import get_recommendations

from typing import List, Optional
from fastapi import FastAPI, HTTPException
import gradio as gr
from pydantic import BaseModel, Field


print("INFERENCE IMPORTED")

app = FastAPI(
    title="Philosophy Semantic Search & Recommendation Engine",
    description="Vector retrieval system for philosophical works using FastEmbed ONNX",
    version="1.0.0",
)


# --- Pydantic Schemas ---
class RecommendRequest(BaseModel):
    query: str = Field(
        ...,
        min_length=3,
        description="Search query or theme to find books for",
    )
    top_k: int = Field(
        default=3, ge=1, le=10, description="Number of recommendations"
    )
    school: Optional[str] = Field(
        default=None, description="Optional philosophical school filter"
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "query": "overcoming gods, individualism, and creating personal values",
                "top_k": 3,
                "school": None,
            }
        }
    }


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


# --- Health Check (Required for AWS ALB) ---
@app.get("/", tags=["Health"])
def health_check():
    return {
        "status": "healthy",
        "service": "philosophy-semantic-engine",
        "version": "1.0.0",
    }


# --- REST API Recommendation Endpoint ---
@app.post(
    "/recommend", response_model=RecommendResponse, tags=["Recommendation"]
)
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
            status_code=500, detail=f"Inference error: {str(e)}"
        )


# --- Gradio UI Wrapper ---
def search_ui(query: str, top_k: int, school: str):
    if not query.strip():
        return "Please enter a valid philosophical query."

    selected_school = None if school == "All" else school
    results = get_recommendations(
        query=query, top_k=int(top_k), school=selected_school
    )

    if not results:
        return "No matches found."

    output_cards = []
    for idx, item in enumerate(results, 1):
        score_percent = f"{item['similarity_score'] * 100:.1f}%"
        card = (
            f"### {idx}. {item['title']} by **{item['author']}**\n"
            f"- **School:** {item['school']}\n"
            f"- **Similarity Match:** `{score_percent}`\n"
            f"- **Summary:** {item['summary']}\n"
        )
        output_cards.append(card)

    return "\n---\n".join(output_cards)


# --- Gradio Interface Layout ---
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

gradio_interface = gr.Interface(
    fn=search_ui,
    inputs=[
        gr.Textbox(
            lines=2,
            placeholder="e.g., how to accept fate and maintain emotional peace...",
            label="What concept or dilemma are you exploring?",
        ),
        gr.Slider(
            minimum=1, maximum=5, value=3, step=1, label="Top Recommendations"
        ),
        gr.Dropdown(
            choices=schools, value="All", label="Filter by Philosophical School"
        ),
    ],
    outputs=gr.Markdown(label="Recommended Books"),
    title="Philosophy Semantic Recommendation Engine",
    description="Vector-based semantic search powered by FastEmbed embeddings.",
    examples=[
        ["a book about defying gods and overcoming conventional morality", 3, "All"],
        ["dealing with loss, anxiety, and things outside my control", 2, "Stoicism"],
        ["human nature is fundamentally good and like a sprouting plant", 2, "Confucianism"],
    ],
)

# Mount Gradio app at /ui
app = gr.mount_gradio_app(app, gradio_interface, path="/ui")

print("MAIN FINISHED")