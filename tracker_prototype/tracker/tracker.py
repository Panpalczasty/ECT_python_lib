import ect
import cv2
import numpy as np

import matplotlib.pyplot as plt

from abc import ABC, abstractmethod
from dataclasses import dataclass

from .matcher import Matcher, BasicMatcher
from .loader import Loader, ImageLoader
from .transformation import Transformation, FilteredTransformation
 
class Tracker(ABC):

    @abstractmethod
    def match(self, image: np.ndarray): ... 


    @abstractmethod
    def match(self, path: str): ...


    @abstractmethod
    def setup(self, **params): ...


    @abstractmethod
    def show_result(self, label: str): ...


@dataclass
class BasicTracker(Tracker):

    template_path: str

    transformer: Transformation | None = None
    loader: Loader | None = None
    matcher: Matcher | None = None
    
    radius: float = 200
    offset: float = 10
    ect_offset: float = 20

    result: np.ndarray | None = None
    ect_template: np.ndarray | None = None


    def setup(self):

        if self.transformer is None:
            self.transformer = FilteredTransformation(
                self.offset,
                self.ect_offset,
                self.radius
            )

        if self.loader is None:
            self.loader = ImageLoader(
                filepath = self.template_path,
                radius   = self.radius,
                offset   = self.offset
            )
        
        if self.matcher is None:
            self.matcher = BasicMatcher(
                bp_thresh = 0.1
            )

        self.ect_template = self.loader.load()
        self.ect_template = self.transformer.transform(self.ect_template)
        
        
    def match(self, image: np.ndarray):

        ect_image = self.transformer.transform(image)

        self.result = self.matcher.match_image(ect_image, self.ect_template)

        self.match_result = self.matcher.match_numeric(ect_image, self.ect_template)

        self.result = self.transformer.invert(self.result)

        return self.result
    

    def show_result(self, label: str): 
        
        radius, shift, value = self.match_result
        print(f"{radius=}, {shift=}, {value=:2f}")

        result_hsv = ect.complex_to_hsv(self.result)
        result_hsv = ect.ilogpolar(result_hsv, offset=self.offset)
        plt.figure(figsize=(10, 10))
        plt.imshow(result_hsv)
        plt.title(label)