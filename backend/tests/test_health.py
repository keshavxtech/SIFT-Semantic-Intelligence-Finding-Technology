def test_root_endpoint(client):
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "operational"
    assert "SIFT" in data["service"]


def test_health_endpoint(client):
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "SIFT" in data["service"]
    assert "version" in data
    assert "storage" in data
    assert isinstance(data["documents_indexed"], int)
