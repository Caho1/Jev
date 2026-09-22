from unittest.mock import Mock

import pytest

from jev_ultrafast import model
from jev_ultrafast.model import decision_endpoint


def test_default_provider_keeps_cloud_contract(monkeypatch):
    monkeypatch.delenv("TYPESAFE_BASE_URL", raising=False)
    monkeypatch.setenv("TYPESAFE_API_KEY", "cloud-secret")
    assert decision_endpoint() == ("https://api.typesafe.ai/v1/systemone", "cloud-secret")


@pytest.mark.parametrize("host", ["127.0.0.1", "localhost", "[::1]"])
def test_local_endpoint_never_receives_cloud_key(monkeypatch, host):
    monkeypatch.setenv("TYPESAFE_BASE_URL", f"http://{host}:8767/v1/")
    monkeypatch.setenv("TYPESAFE_API_KEY", "cloud-secret")
    assert decision_endpoint() == (f"http://{host}:8767/v1/systemone", "")
    monkeypatch.delenv("TYPESAFE_API_KEY")
    assert decision_endpoint()[1] == ""


@pytest.mark.parametrize("url", ["http://example.com/v1", "file:///tmp/a", "https://user:pw@example.com"])
def test_unsafe_endpoint_rejected(monkeypatch, url):
    monkeypatch.setenv("TYPESAFE_BASE_URL", url)
    with pytest.raises(ValueError):
        decision_endpoint()


def test_local_post_omits_empty_bearer_header(monkeypatch):
    client = Mock()
    client.post.return_value.status_code = 200
    client.post.return_value.is_error = False
    client.post.return_value.json.return_value = {"answers": {}}
    monkeypatch.setattr(model, "CLIENT", client)
    assert model.post_json("http://127.0.0.1:8767/v1/systemone", "", {}) == {"answers": {}}
    assert client.post.call_args.kwargs["headers"] == {}
