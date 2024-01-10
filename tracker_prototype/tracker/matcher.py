import cv2
import ect
import numpy as np

from numpy import ndarray

from abc import ABC, abstractmethod
from dataclasses import dataclass

from .transformation import Transformation
from typing import Callable

@dataclass
class Matcher(ABC):

    transformer: Transformation
    center_fnc: Callable[[ndarray], tuple[int, int]]

    @abstractmethod
    def prepare(
        self,
        image: ndarray
    ) -> ndarray: ...

    @abstractmethod
    def match_image(
        self, 
        image: ndarray, 
        template: ndarray
        ) -> ndarray: ...


    def match_numeric(
            self, 
            image: ndarray,
            template: ndarray
            ) -> tuple[float, float]:

        if self.match_result is None:
            match_result = self.match_image(image, template)
        else:
            match_result = self.match_result

        # print(self.center_fnc.__name__)
        return self.center_fnc(match_result)


    def match_point(self, image:ndarray, template:ndarray) -> ndarray:

        x, y = self.match_numeric(image, template)

        return cv2.circle(
            img = self.match_result,
            center = (int(x), int(y)),
            radius = 10,
            color = (255, 0, 0),
            thickness = 3  
        )
    
    def clear(self):
        self.match_result = None
    

class DummyMatcher(Matcher):

    def match_image(self, image: ndarray, template: ndarray) -> ndarray:
        return np.zeros_like(image)
       
    
@dataclass    
class BasicMatcher(Matcher):

    transformer: Transformation
    bp_thresh: float = 0.2
    match_result: ndarray | None = None

    def prepare(self, image: ndarray) -> ndarray:
        return self.transformer.transform(image)

    def match_image(self, image: ndarray, template: ndarray) -> ndarray:
        
        if self.match_result is not None:
            return self.match_result

        xcorr = template * np.conj(image)
        xcorr_norm = xcorr/(xcorr + 1e-12)

        template_abs = ect.norm_minmax(np.abs(template), 0, 1, dtype=np.float64)
        bp_filter = np.zeros_like(template_abs)
        bp_filter[template_abs > self.bp_thresh] = 1

        self.match_result = xcorr_norm * bp_filter
        self.match_result = self.transformer.invert(self.match_result)
        self.match_result = np.abs(self.match_result)
        self.match_result = ect.norm_minmax(self.match_result, 0, 255, dtype=np.uint8)
        self.match_result = ect.ilogpolar(self.match_result, offset=0)
        
        return self.match_result


@dataclass
class CorrelationMatcher(Matcher):

    transformer: Transformation

    def match_image(self, image: ndarray, template: ndarray) -> ndarray:
        
        xcorr = image * np.conjugate(template)
        xcorr /= np.abs(xcorr)

        return xcorr
        # return self.transformer.invert(xcorr)
    
@dataclass
class RotShiftMatcher(Matcher):

    transformer: Transformation
    bp_thresh: float = 0.2
    match_result: ndarray | None = None

    
    def prepare(self, image: ndarray) -> ndarray:
        image = self.transformer.transform(image)
        # # image = np.abs(image)
        # pad = np.zeros_like(image)
        # image = np.hstack((pad, image))
        image = np.fft.fft2(image, axes=(0, 1))
        return image

    def match_image(self, image: ndarray, template: ndarray) -> ndarray:
        
        if self.match_result is not None:
            return self.match_result

        # calculate phase shift
        xcorr = template * np.conj(image)
        xcorr_norm = xcorr/(np.abs(xcorr) + 1e-12)

        template_abs = ect.norm_minmax(np.abs(template), 0, 1, dtype=np.float64)
        bp_filter = np.zeros_like(template_abs)
        bp_filter[template_abs > self.bp_thresh] = 1

        P, R = template.shape[:2]
        # return bp_filter
        # return np.exp(1j*np.arctan(phase)) * bp_filter
        self.match_result = xcorr_norm * bp_filter
        self.match_result = np.fft.ifft2(self.match_result, axes=(0, 1))

        left_half, right_half = np.hsplit(self.match_result, 2)
        self.match_result = np.hstack((right_half, left_half))

        upper_half, lower_half = np.vsplit(self.match_result, 2)
        self.match_result = np.vstack((lower_half, upper_half))

        # self.match_result = self.transformer.invert(self.match_result)
        self.match_result = np.abs(self.match_result)
        self.match_result = ect.norm_minmax(self.match_result, 0, 255, dtype=np.uint8)
        self.match_result = cv2.cvtColor(self.match_result, cv2.COLOR_GRAY2BGR)

        return self.match_result
