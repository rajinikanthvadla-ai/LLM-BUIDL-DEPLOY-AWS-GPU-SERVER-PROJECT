from app.main import app
from fastapi.testclient import TestClient

client = TestClient(app)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_infer_no_gpu_returns_503():
    r = client.post("/v1/infer", json={"prompt": "hello"})
    assert r.status_code == 503
