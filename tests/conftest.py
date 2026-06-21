from pathlib import Path

import pytest
import torch

from orb_models.common import utils


def pytest_addoption(parser):
    """Opt-in flags for test categories skipped by default:

    * ``--run-integration``: slow/heavyweight tests (real-scale architectures and
      tests that download released checkpoints).
    * ``--run-equivalence``: JAX<->PyTorch numeric equivalence tests, which require
      the torch reference and verify the JAX port matches torch (as opposed to the
      fundamental unit tests that check the JAX code's behaviour on its own).
    """
    parser.addoption(
        "--run-integration",
        action="store_true",
        default=False,
        help="run integration tests (real-scale models, checkpoint downloads)",
    )
    parser.addoption(
        "--run-equivalence",
        action="store_true",
        default=False,
        help="run JAX<->PyTorch numeric equivalence tests",
    )


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "integration: slow/heavyweight test (real-scale model or checkpoint "
        "download); only runs with --run-integration",
    )
    config.addinivalue_line(
        "markers",
        "equivalence: JAX<->PyTorch numeric equivalence test; only runs with "
        "--run-equivalence",
    )


def pytest_collection_modifyitems(config, items):
    for opt, mark in (
        ("--run-integration", "integration"),
        ("--run-equivalence", "equivalence"),
    ):
        if config.getoption(opt):
            continue
        skip = pytest.mark.skip(reason=f"needs {opt}")
        for item in items:
            if mark in item.keywords:
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
