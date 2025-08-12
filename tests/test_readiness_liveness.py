import json

def test_liveness_endpoint(client):
    res = client.get('/live')
    assert res.status_code == 200
    body = res.get_json()
    assert body["status"] == "ok"


def test_readiness_endpoint(client):
    res = client.get('/ready')
    # Readiness may be 200 or 503 depending on whether the model is present
    assert res.status_code in (200, 503)
    body = res.get_json()
    assert body["status"] in {"ready", "not_ready"}

