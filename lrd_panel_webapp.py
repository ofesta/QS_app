# -*- coding: utf-8 -*-
"""
Created on Wed May 31 02:06:05 2023

@author: ogf1n20
"""
import panel as pn
from panel.pane import Matplotlib
import param
from lrd_module import LrdDesign  

class LrdPanel(param.Parameterized):
    lrd_type = param.Selector(default="DO", objects=["DO", "TFI"])
    lrd_length = param.Number(default=12)
    lrd_dia = param.Number(default=4)
    lrd_o = param.Number(default=1.5)
    lrd_v = param.Number(default=5)
    theta = param.Number(default=30)
    lrd_fg = param.Number(default=5e6)
    # here you can add more parameters for TFI type if needed

    def __init__(self, **params):
        super().__init__(**params)
        self.update_lrd()

    @param.depends('lrd_type', 'lrd_length', 'lrd_dia', 'lrd_o', 'lrd_v', 'theta', 'lrd_fg', watch=True)
    def update_lrd(self):
        self.lrd = LrdDesign(
            self.lrd_type,
            lrd_length=self.lrd_length,
            lrd_dia=self.lrd_dia,
            lrd_o=self.lrd_o,
            lrd_v=self.lrd_v,
            theta=self.theta,
            lrd_fg=self.lrd_fg
            # add more parameters for TFI here if needed
        )

    @param.depends('lrd_type', 'lrd_length', 'lrd_dia', 'lrd_o', 'lrd_v', 'theta', 'lrd_fg')
    def draw_and_plot_lrd(self):
        return Matplotlib(self.lrd.draw_plot_lrd())

    def panel(self):
        return pn.Row(
            pn.Column(self.param, width=300),
            self.draw_and_plot_lrd,
        )

app = LrdPanel()
app.panel().servable()
pn.serve(app.panel(), port=5098)