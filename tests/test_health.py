import json

def test_health_endpoint(client):
    res = client.get('/health')
    # Health should always respond 200 with JSON. Model may be loaded or not.
    assert res.status_code == 200
    body = res.get_json()
    assert body["status"] in {"healthy", "error"}
    assert "model_info" in body
    assert "timestamp" in body

