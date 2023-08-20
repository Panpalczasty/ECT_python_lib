import cv2
import ect
import numpy as np

from numpy import ndarray

from abc import ABC, abstractmethod
from dataclasses import dataclass


class Matcher(ABC):

    @abstractmethod
    def match_image(
        self, 
        image_ect: ndarray, 
        template_ect: ndarray
        ) -> ndarray: ...


    @abstractmethod
    def match_numeric(
        self, 
        image_ect: ndarray, 
        template_ect: ndarray
        ) -> tuple[float, float, float]: ...


    @abstractmethod
    def match_bbox(
        self, 
        image_ect: ndarray, 
        template_ect: ndarray
        ) -> ndarray: ...
    

class DummyMatcher(Matcher):

    def match_image(self, image_ect: ndarray, template_ect: ndarray) -> ndarray:
        return np.zeros_like(image_ect)
    
    def match_numeric(self, image_ect: ndarray, template_ect: ndarray) -> tuple[float, float, float]:
        return (0, 0, 0)
    
    def match_bbox(self, image_ect: ndarray, template_ect: ndarray) -> ndarray:
        return np.zeros_like(image_ect)
    
    
class BasicMatcher(Matcher):

    def match_image(self, image_ect: ndarray, template_ect: ndarray) -> ndarray:
        
        # calculate phase shift
        ir = np.real(image_ect)
        ii = np.imag(image_ect)
        tr = np.real(template_ect)
        ti = np.imag(template_ect)

        phase = (ir*ti - ii*tr)/(ii*ti + tr*ir + 10e-6)
        # phase = (ii*ti + tr*ir)

        thresh = .6
        # apply band-bass filter (from ect.filt tho)
        template_abs = ect.norm_minmax(np.abs(template_ect), 0, 1, dtype=np.float64)

        bp_filter = np.zeros_like(template_abs)
        bp_filter[template_abs > thresh] = 1

        return np.arctan(phase) * bp_filter

    def match_bbox(self, image_ect:ndarray, template_ect:ndarray) -> ndarray:
        return super().match_bbox(image, template)
    
    def match_numeric(self, image_ect:ndarray, template_ect:ndarray) -> tuple[float, float, float]:
        
        match_result = self.match_image(image_ect, template_ect)

        max_idx = np.argmax(abs(match_result))
        max_phase, max_radius = np.unravel_index(max_idx, match_result.shape[:2])
        max_value = match_result[max_phase, max_radius, 0]

        R = match_result.shape[1]

        max_phase /= 314
        max_phase *= np.pi
        max_phase -= np.pi/2

        return max_radius, max_phase, max_value
