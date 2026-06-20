# -*- coding: utf-8 -*-
"""Shared flow-layout helpers for plugin property panels."""

from __future__ import annotations

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
	QGroupBox,
	QHBoxLayout,
	QLabel,
	QPushButton,
	QSizePolicy,
	QVBoxLayout,
	QWidget,
)

from dpVision.gui.flowLayout import FlowLayout


class CollapsibleGroup(QWidget):
	"""Simple collapsible section with a relayout-aware body widget."""

	def __init__(self, title, collapsed=True, parent=None):
		"""Create one collapsible group with a body that participates in relayout."""
		super().__init__(parent)
		self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
		outer = QVBoxLayout(self)
		outer.setContentsMargins(0, 2, 0, 2)
		outer.setSpacing(0)
		outer.setAlignment(Qt.AlignTop | Qt.AlignLeft)

		self._title = title
		self._btn = QPushButton()
		self._btn.setCheckable(True)
		self._btn.setChecked(not collapsed)
		self._btn.setFlat(True)
		self._btn.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Fixed)
		self._btn.setStyleSheet(
			"QPushButton { text-align: left; padding: 3px 6px; font-weight: bold; }"
		)
		self._update_text(not collapsed)
		self._btn.setMaximumWidth(self._btn.sizeHint().width())
		outer.addWidget(self._btn)

		self._body = QWidget()
		self._body.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
		self._body.setVisible(not collapsed)
		outer.addWidget(self._body)

		self._btn.toggled.connect(self._on_toggle)

	def _update_text(self, expanded):
		"""Refresh the disclosure marker according to the expanded state."""
		self._btn.setText(("\u25BC  " if expanded else "\u25B6  ") + self._title)
		self._btn.setMaximumWidth(self._btn.sizeHint().width())

	def _on_toggle(self, checked):
		"""Show or hide the body and notify parent layouts about the new size hint."""
		self._update_text(checked)
		self._body.setVisible(checked)
		self.updateGeometry()
		parent = self.parentWidget()
		while parent is not None:
			parent.updateGeometry()
			parent = parent.parentWidget()

	def body(self):
		"""Return the inner widget to which a content layout should be assigned."""
		return self._body


class FlowGroupBox(QGroupBox):
	"""Group box that forwards height-for-width to its wrapping child layout."""

	def hasHeightForWidth(self):
		"""Report height-for-width when the installed layout supports it."""
		layout = self.layout()
		return bool(layout is not None and layout.hasHeightForWidth())

	def heightForWidth(self, width):
		"""Return the group height needed for the given content width."""
		layout = self.layout()
		if layout is None or not layout.hasHeightForWidth():
			return super().heightForWidth(width)
		return layout.totalHeightForWidth(max(0, width))

	def minimumSizeHint(self):
		"""Keep the minimum size consistent with the flow layout's wrapped content."""
		layout = self.layout()
		if layout is None:
			return super().minimumSizeHint()
		return layout.minimumSize()

	def sizeHint(self):
		"""Prefer the current width and let the height follow the wrapped content."""
		size_hint = super().sizeHint()
		if self.hasHeightForWidth():
			size_hint.setHeight(self.heightForWidth(size_hint.width()))
		return size_hint


class FlowPanelMixin:
	"""Shared helpers for flow-based property panels."""

	def _create_flow_group(self, title):
		"""Create one width-aware group box with wrapping controls."""
		group = FlowGroupBox(title)
		group.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
		layout = FlowLayout(group)
		layout.setContentsMargins(6, 6, 6, 6)
		layout.setSpacing(4)
		layout.setAlignment(Qt.AlignTop | Qt.AlignLeft)
		return group, layout

	def _create_collapsible_flow_group(self, title, collapsed=True):
		"""Create one collapsible section whose body uses a wrapping flow layout."""
		group = CollapsibleGroup(title, collapsed=collapsed)
		layout = FlowLayout(group.body())
		layout.setContentsMargins(6, 6, 6, 6)
		layout.setSpacing(4)
		layout.setAlignment(Qt.AlignTop | Qt.AlignLeft)
		return group, layout

	def _wrap_flow_widget(self, control):
		"""Wrap one standalone control so it behaves like a flow item."""
		widget = QWidget()
		widget.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Preferred)
		layout = QVBoxLayout(widget)
		layout.setContentsMargins(6, 6, 6, 6)
		layout.setSpacing(4)
		layout.addWidget(control)
		return widget

	def _vcontrol(self, label_text, control):
		"""Return one labeled control block suitable for flow-based groups."""
		widget = QWidget()
		widget.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Preferred)
		
		# widget.setStyleSheet("border: 1px solid transparent; border-radius: 4px;")
		layout = QVBoxLayout(widget)
		layout.setContentsMargins(6, 6, 6, 6)
		layout.setSpacing(4)
		
		label = QLabel(f"{label_text}:")
		label.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Preferred)
		layout.addWidget(label)
		layout.addWidget(control)
		return widget

	def _add_flow_control(self, layout, label_text, control):
		"""Add one wrapped control to a flow layout, with or without label."""
		if label_text:
			layout.addWidget(self._vcontrol(label_text, control))
		else:
			layout.addWidget(self._wrap_flow_widget(control))

	def _set_compact_field(self, widget):
		"""Prefer size-hint width for editor widgets inside flow groups."""
		widget.setSizePolicy(QSizePolicy.Maximum, widget.sizePolicy().verticalPolicy())

	def _set_compact_group(self, group):
		"""Keep flow-based group boxes width-aware and height-driven by contents."""
		group.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)

	def _create_inline_row(self, *widgets):
		"""Create one compact inline row widget for use inside flow controls."""
		row = QWidget()
		row.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Preferred)
		layout = QHBoxLayout(row)
		layout.setContentsMargins(0, 0, 0, 0)
		layout.setSpacing(4)
		for widget in widgets:
			layout.addWidget(widget)
		return row
