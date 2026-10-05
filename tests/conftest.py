import pytest

from opsagent import world as W
from opsagent.store import Store


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("OPSAGENT_WORLD", str(tmp_path / "world.json"))
    W.reset()
    store = Store(tmp_path / "ops.db")
    yield store
    store.close()
