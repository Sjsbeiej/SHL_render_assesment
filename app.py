"""
FastAPI service for the SHL Assessment Recommendation Engine.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from core.graph import recommend, warmup

app = FastAPI(
    title="SHL Assessment Recommendation Engine",
    description="GenAI-powered recommendation system for SHL assessments",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class RecommendRequest(BaseModel):
    query: str


class AssessmentResponse(BaseModel):
    url: str
    adaptive_support: str
    description: str
    duration: int | None
    name: str
    remote_support: str
    test_type: list[str]


class RecommendResponse(BaseModel):
    recommended_assessments: list[AssessmentResponse]


@app.on_event("startup")
async def startup():
    """Pre-load FAISS index and models on startup."""
    warmup()


@app.get("/health")
async def health():
    return {"status": "healthy"}


@app.post("/recommend", response_model=RecommendResponse)
async def recommend_assessments(request: RecommendRequest):
    if not request.query or not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    try:
        results = recommend(request.query)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Recommendation failed: {str(e)}")

    assessments = []
    for r in results:
        assessments.append(AssessmentResponse(
            url=r["url"],
            adaptive_support=r.get("adaptive_support", "No"),
            description=r.get("description", ""),
            duration=r.get("duration"),
            name=r["name"],
            remote_support=r.get("remote_support", "No"),
            test_type=r.get("test_type", []),
        ))

    return RecommendResponse(recommended_assessments=assessments)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
