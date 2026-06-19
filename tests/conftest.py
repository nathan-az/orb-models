from pathlib import Path

import pytest
import torch

from orb_models.common import utils


def pytest_addoption(parser):
    """Add `--run-integration` to opt into slow/heavyweight tests (real-scale
    architectures and tests that download released checkpoints), which are skipped
    by default so the unit suite stays fast and offline."""
    parser.addoption(
        "--run-integration",
        action="store_true",
        default=False,
        help="run integration tests (real-scale models, checkpoint downloads)",
    )


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "integration: slow/heavyweight test (real-scale model or checkpoint "
        "download); only runs with --run-integration",
    )


def pytest_collection_modifyitems(config, items):
    if config.getoption("--run-integration"):
        return
    skip = pytest.mark.skip(reason="needs --run-integration")
    for item in items:
        if "integration" in item.keywords:
            item.add_marker(skip)


@pytest.fixture(autouse=True, scope="function")
def default_test_setup():
    """
    Ensure all tests by default use float32, are deterministic and have the same seed.
    Deviations should explicitly be made within each test.
    """
    torch.set_default_dtype(torch.float32)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    utils.seed_everything(42, 0)
    yield


@pytest.fixture(scope="module")
def fixtures_path(request):
    """Return the file fixtures path for any script."""
    return Path(request.fspath).parent / "fixtures"


@pytest.fixture(scope="module")
def shared_fixtures_path(request):
    """Return the top-level fixtures path for any script."""
    return Path(request.fspath).parent.parent / "fixtures"
