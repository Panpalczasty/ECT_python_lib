import ect
import cv2
import numpy as np

from numpy import ndarray

import matplotlib.pyplot as plt

from abc import ABC, abstractmethod
from dataclasses import dataclass

from .matcher import Matcher, BasicMatcher, RotShiftMatcher
from .loader import Loader, FilepathLoader
from .transformation import Transformation, UnfilteredTransformation
from .center import max_index, center_of_mass

@dataclass
class Tracker(ABC):

    loader: Loader
    matcher: Matcher

    def match(self, image: ndarray): 

        self.matcher.clear()

        ect_image = self.matcher.prepare(image)
        self.result = self.matcher.match_point(ect_image, self.ect_template)
        self.match_result = self.matcher.match_numeric(ect_image, self.ect_template)

        return self.result


    def match_file(self, path: str): 
        
        image = self.loader.load(path)
        return self.match(image)
  

    @abstractmethod
    def setup(self, **params): ...


    @abstractmethod
    def show_result(self, label: ndarray, *args, **kwargs): ...


@dataclass
class BasicTracker(Tracker):

    template_path: str | None = None

    loader: Loader | None = None
    matcher: Matcher | None = None
    
    radius: float = 200
    offset: float = 10
    ect_offset: float = 10

    result: ndarray | None = None
    ect_template: ndarray | None = None

    def setup(self):

        if self.loader is None:
            self.loader = FilepathLoader(
                filepath = self.template_path,
                radius   = self.radius,
                offset   = self.offset
            )
        
        if self.matcher is None:
            self.matcher = BasicMatcher(
                transformer = UnfilteredTransformation(
                    self.img_offset, 
                    self.ect_offset),
                center_fnc = max_index,
                bp_thresh = 0.1
            )

        self.ect_template = self.loader.load(self.template_path)
        self.ect_template = self.matcher.prepare(self.ect_template)
        

    def show_result(self, label: str, image: ndarray): 
        
        x, y = self.match_result
        print(f"{label}: {x=}, {y=}")

        plt.figure(figsize=(10, 10))
        plt.imshow(self.result)
        plt.title(label)

@dataclass
class CalcRSTracker(Tracker):

    template_path: str | None = None

    loader: Loader | None = None
    matcher: Matcher | None = None
    
    radius: float = 200
    offset: float = 10
    ect_offset: float = 10

    result: ndarray | None = None
    ect_template: ndarray | None = None

    def setup(self, **params):

        if self.loader is None:
            self.loader = FilepathLoader(
                filepath = self.template_path,
                radius   = self.radius,
                offset   = self.offset
            )
        
        if self.matcher is None:
            self.matcher = RotShiftMatcher(
                transformer = UnfilteredTransformation(
                    self.img_offset, 
                    self.ect_offset),
                center_fnc = center_of_mass,
                bp_thresh = 0.1
            )

        self.ect_template = self.loader.load(self.template_path)
        self.ect_template = self.matcher.prepare(self.ect_template)

    def _calculate(self, x: float, y: float):

        R = self.radius
        P = int(self.radius/100*np.pi)

        angle = (y - int(self.radius*np.pi)/2)*1.8/np.pi
        scale_factor = 2.4*(x+1) - 141.6 

        return scale_factor, angle
    
    def show_result(self, label: str, *args, **kwargs):

        scale_raw, angle_raw = self.match_result
        scale, angle = self._calculate(scale_raw, angle_raw)
        print(f"{label}: {scale=:.2f}%, {angle=:.2f} deg")

        plt.figure(figsize=(10, 10))
        plt.imshow(self.result)
        plt.title(label)