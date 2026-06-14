# -*- coding: utf-8 -*-
"""Shared pytest fixtures for the `virtRTG` plugin test suite."""

from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np
import pytest


def _ensure_repo_root_on_syspath():
	"""Ensure the project root is importable when tests run from the plugin subtree."""
	repo_root = Path(__file__).resolve().parents[3]
	repo_root_str = str(repo_root)
	if repo_root_str not in sys.path:
		sys.path.insert(0, repo_root_str)


def _install_plugin_package_stubs():
	"""Install namespace-like package stubs to avoid host-side import side effects.

	The repository-level `plugins/__init__.py` currently performs application
	bootstrapping during import. Backend unit tests must bypass that side effect
	and import `plugins.virtRTG.*` as a plain package tree.
	"""
	repo_root = Path(__file__).resolve().parents[3]
	plugins_dir = repo_root / "plugins"
	virt_rtg_dir = plugins_dir / "virtRTG"
	xray_dir = virt_rtg_dir / "xray"
	gui_dir = virt_rtg_dir / "gui"

	if "plugins" not in sys.modules:
		plugins_pkg = types.ModuleType("plugins")
		plugins_pkg.__path__ = [str(plugins_dir)]
		sys.modules["plugins"] = plugins_pkg

	if "plugins.virtRTG" not in sys.modules:
		virt_rtg_pkg = types.ModuleType("plugins.virtRTG")
		virt_rtg_pkg.__path__ = [str(virt_rtg_dir)]
		sys.modules["plugins.virtRTG"] = virt_rtg_pkg

	if "plugins.virtRTG.xray" not in sys.modules:
		xray_pkg = types.ModuleType("plugins.virtRTG.xray")
		xray_pkg.__path__ = [str(xray_dir)]
		sys.modules["plugins.virtRTG.xray"] = xray_pkg

	if "plugins.virtRTG.gui" not in sys.modules:
		gui_pkg = types.ModuleType("plugins.virtRTG.gui")
		gui_pkg.__path__ = [str(gui_dir)]
		sys.modules["plugins.virtRTG.gui"] = gui_pkg


_ensure_repo_root_on_syspath()
_install_plugin_package_stubs()


@pytest.fixture
def sample_projection_image():
	"""Return one small deterministic projection image for presentation-model tests."""
	return np.array(
		[
			[0.0, 1.0, 2.0],
			[3.0, 4.0, 5.0],
		],
		dtype=np.float32,
	)
