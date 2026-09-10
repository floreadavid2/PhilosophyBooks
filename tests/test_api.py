import pytest
from httpx import ASGITransport, AsyncClient
from src.app.main import app


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture
async def client():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


@pytest.mark.anyio
async def test_health_check(client):
    """Verify health check returns 200 for ALB monitoring."""
    response = await client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


@pytest.mark.anyio
async def test_recommendation_endpoint_valid_payload(client):
    """Test vector retrieval through standard POST request."""
    payload = {
        "query": "overcoming despair and accepting the absurd",
        "top_k": 2,
    }
    response = await client.post("/recommend", json=payload)
    assert response.status_code == 200

    data = response.json()
    assert data["results_count"] == 2
    assert len(data["recommendations"]) == 2
    assert "title" in data["recommendations"][0]
    assert "similarity_score" in data["recommendations"][0]


@pytest.mark.anyio
async def test_recommendation_filtering(client):
    """Test metadata filtering by philosophical school."""
    payload = {
        "query": "duty and accepting mortality",
        "top_k": 3,
        "school": "Stoicism",
    }
    response = await client.post("/recommend", json=payload)
    assert response.status_code == 200

    recs = response.json()["recommendations"]
    for rec in recs:
        assert rec["school"] == "Stoicism"


@pytest.mark.anyio
async def test_invalid_payload_too_short(client):
    """Ensure validation error (422) when query is too short."""
    payload = {"query": "a", "top_k": 2}
    response = await client.post("/recommend", json=payload)
    assert response.status_code == 422