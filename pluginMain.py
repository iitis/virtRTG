# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 11:08:03 2023

@author: pojdulos
"""

from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *

from dpVision import AP, Object, Transform, PluginInterface
from dpVision.annotationPath import AnnotationPath
from dpVision.annotationPoint import AnnotationPoint
from .virtualXRay import VirtualXRay
from .gui.propVirtualXRay import PropVirtualXRay

import numpy as np
import os

class VirtualRTG(PluginInterface):
	def __init__(self):
		self.plugin_name = 'Virtual RTG'
		self.panel = None
		self.setup = None
		AP.mainWin.dock["properties"].properties_map[Object]['VirtualXRay'] = PropVirtualXRay
		#AP.mainWin.dock["properties"].properties_map[BaseObject]['Graph'] = PropGraph


	def on_load(self):
		print( "plugin "+self.plugin_name+" loaded.")
		
		# self.mainWindow.helpAbout()
		self.add_menu()
		# self.create_panel(AP.mainWin)

	def create_panel(self, parent):
		self.panel = QDockWidget(parent)
		self.panel.setWindowTitle(self.plugin_name)
		AP.mainWin.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.panel)

	def on_unload(self):
		self.remove_menu()
		if self.panel:
			AP.mainWin.removeDockWidget(self.panel);    
		print( "Bye bye")


	def add_menu(self):
		nadrzedne_menu = self.add_plugins_menu()

		self.menu = QMenu(self.plugin_name, AP.mainWin)
		
		nadrzedne_menu.addMenu(self.menu)
		
		action = QAction("Create RTG", AP.mainWin)
		self.menu.addAction(action)
		action.triggered.connect(self.onAction_Create_RTG)

		action = QAction("Create Demo", AP.mainWin)
		self.menu.addAction(action)
		action.triggered.connect(self.onAction_Create_Demo)

		self.menu.addSeparator()

		action = QAction("Benchmark", AP.mainWin)
		self.menu.addAction(action)
		action.triggered.connect(self.onAction_Benchmark)

		self.menu.addSeparator()

		action = QAction("UnLoad", AP.mainWin)
		self.menu.addAction(action)
		action.triggered.connect(self.onAction_UnLoad)
		
	def remove_menu(self):
		# Znajdź "NadrzędneMenu"
		nadrzedne_menu = None
		for action in AP.mainWin.menuBar.actions():
			if action.text() == "Plugins":
				nadrzedne_menu = action.menu()
				break
		
		if nadrzedne_menu:
			# Znajdź i usuń "Plugin01"
			plugin_menu_action = None
			for action in nadrzedne_menu.actions():
				if action.text() == self.plugin_name and action.menu():
					plugin_menu_action = action
					break
			
			if plugin_menu_action:
				nadrzedne_menu.removeAction(plugin_menu_action)
		
		self.remove_plugins_menu()
		

	def onAction_UnLoad(self):
		print("Akcja menu: Wyładuj plugin")
		AP.mainApp.unload_plugin(self)

	def onAction_Create_RTG(self):
		print("Akcja menu: Create RTG")

		if self.setup is None:
			self.setup = VirtualXRay()

		if self.setup.parent is None and self.setup not in AP.mainWin.workspace.m_data:
			AP.addObject(self.setup)

		self.setup.apply_geometry_preset("orthoralix_PA")


	def onAction_Create_Demo(self):
		print("Akcja menu: Create Demo")
		
		self.onAction_Create_RTG()

		def on_success(skull):
			if skull is None:
				return

			AP.removeObject(child=skull.parent)

			# if isinstance(skull, Mesh):
			# 	report_mesh_xray_topology(skull)

				# skull, result = clean_mesh_for_xray(skull, 
				# 			drop_nonmanifold_faces=True,
				# 			drop_boundary_faces=False,)
				# print(result)
				# print(result["cleaned_report"])


			skull_transform = Transform()
			#skull_transform.translate(-100, 45, 0)
			skull_transform.rotate(-90, [1,0,0])
			skull_transform.rotate(90, [0,1,0])
			skull_transform.addChild(skull)
			self.setup.addChild(skull_transform)

			p = [-78.18, -74.30, -21.59]
			s = [147.456, 147.456, 86.976]

			points = {
				'A': [p[0],     p[1],     p[2]],
				'B': [p[0],     p[1],     p[2]+s[2]],
				'C': [p[0],     p[1]+s[1],p[2]],
				'D': [p[0],     p[1]+s[1],p[2]+s[2]],
				'E': [p[0]+s[0],p[1],     p[2]],
				'F': [p[0]+s[0],p[1],     p[2]+s[2]],
				'G': [p[0]+s[0],p[1]+s[1],p[2]],
				'H': [p[0]+s[0],p[1]+s[1],p[2]+s[2]],
			}

			for label, pos in points.items():
				point = AnnotationPoint(point = pos)
				point.label = label
				#point.setColor(r=255, g=0, b=0, a=255)
				skull.addChild(point)

			path = AnnotationPath()
			path.addPoint([v+30 for v in points['A']])
			path.addPoint([v-30 for v in points['H']])
			path.addPoint([0,0,0])
			
			path.setColor(name="red")
			path.label = "random line"
			skull.addChild(path)

			AP.updateAllViews()
			AP.mainWin.dock["workspace"].rebuildTree()

			skull.xray_mesh_backend = "projected_intersection_list"
			# skull.xray_debug_export_dir = r"d:\temp\xray_debug"
			# skull.xray_debug_compare_analytic = True
			# skull.xray_projected_min_abs_cos = 0.25

			#setup.debug_run_simulation_stop_after = "display"
			#setup.debug_run_simulation_stop_after = "update_views"
			#result = run_virtual_xray_headless(setup, r"d:/temp/vxray_test.png")

		# path1 = "c:/Users/darek/Desktop/praca/dane/20210312_142843/DCT0000.dcm"
		# path2 = "d:/praca0/dpVisionProject/dane/20210312_142843/DCT0000.dcm"
		# if os.path.isfile(path1):
		# 	load_path = path1
		# elif os.path.isfile(path2):
		# 	load_path = path2
		# else:
		# 	print("Nie można znaleźć pliku DICOM do testu X-ray demo.")
		# 	return
		
		# AP.load(load_path, on_success=on_success)

		# pathG = "d:/praca/dane/masks/gora/slice_000.dcm"
		# pathD = "d:/praca/dane/masks/dol/slice_000.dcm"

		pathG = "d:/praca/dane/vols/gora1/filtered_194.dcm"
		pathD = "d:/praca/dane/vols/dol/filtered_080.dcm"
		# pathD = "d:/praca/dane/vols/jaw_poisson.ply"
		# pathA = "d:/praca0/dpVisionProject/dane/20160501/filt/NDecom0000.dcm"
		# pathA = "d:/praca/dane/vols/20140521/0000.dcm"

		if os.path.isfile(pathG):
			AP.load(pathG, on_success=on_success)
		
		# if os.path.isfile(pathD):
		# 	AP.load(pathD, on_success=on_success)
		# # 	# AP.load(pathD, on_success=on_success)

		# if os.path.isfile(pathA):
		# 	AP.load(pathA, on_success=on_success)


	def onAction_Benchmark(self):
		print("Akcja menu: Benchmark")
		from .benchmark import benchmark_xray_performance, generate_latex_benchmark_report
		_bm_results, _diff_stats = benchmark_xray_performance()
		generate_latex_benchmark_report(_bm_results, diff_stats=_diff_stats)


	def perform_action(self):
		print("Akcja wykonana przez "+self.plugin_name)

	def on_button1(self):
		#QMessageBox.information(self.mainWindow, 'Komunikat', 'Akcja wykonana przez '+self.plugin_name)
		pass
