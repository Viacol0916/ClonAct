"""Keep service-starting integration checks explicit; no model calls in offline tests."""
import pytest


def pytest_addoption(parser):
    parser.addoption("--run-integration", action="store_true", default=False,
                     help="Run tests that start benchmark services and require downloaded data")


    parser.addoption("--run-model-tests", action="store_true", default=False,
                     help="Explicitly allow tests that submit real model requests and incur API cost")


def pytest_configure(config):
    config.addinivalue_line("markers", "model_api: invokes paid model APIs; requires --run-model-tests")
    config.addinivalue_line("markers", "integration: starts local benchmark services; requires upstream environments")


def pytest_collection_modifyitems(config, items):
    for item in items:
        if "model_api" in item.keywords and not config.getoption("--run-model-tests"):
            item.add_marker(pytest.mark.skip(reason="Use --run-model-tests to explicitly enable paid model calls"))
    if not config.getoption("--run-integration"):
        for item in items:
            if "integration" in item.keywords:
                item.add_marker(pytest.mark.skip(reason="Use --run-integration with provisioned upstream environments"))


@pytest.fixture(autouse=True)
def forbid_unrequested_model_access(request, monkeypatch):
    """Offline tests cannot boot DSH or issue model HTTP requests, even with local credentials."""
    if request.node.get_closest_marker("model_api") and request.config.getoption("--run-model-tests"):
        return
    from dsh_runner.synaptic_worker_controller import SynapticDSHWorkerHost
    import httpx
    async def no_model_host(*args, **kwargs):
        raise AssertionError("Offline tests must use FakeDSHHost; model access requires --run-model-tests")
    monkeypatch.setattr(SynapticDSHWorkerHost, "start", no_model_host)
    original_sync = httpx.Client.send
    original_async = httpx.AsyncClient.send
    def local_only(url):
        if url.host not in {"127.0.0.1", "localhost", "testserver"}:
            raise AssertionError(f"External HTTP forbidden in offline tests: {url.host}")
    def guarded_sync(self, request, *args, **kwargs):
        local_only(request.url)
        return original_sync(self, request, *args, **kwargs)
    async def guarded_async(self, request, *args, **kwargs):
        local_only(request.url)
        return await original_async(self, request, *args, **kwargs)
    monkeypatch.setattr(httpx.Client, "send", guarded_sync)
    monkeypatch.setattr(httpx.AsyncClient, "send", guarded_async)
