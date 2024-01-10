from numpy import ndarray
import ect
import cv2
import math
import numpy as np
import matplotlib.pyplot as plt

from .tracker import Tracker
from .loader import FilepathLoader, Loader
from .matcher import RotShiftMatcher, Matcher
from .transformation import UnfilteredTransformation
from .center import center_of_mass

from dataclasses import dataclass, field

@dataclass
class TrajectoryLPTracker(Tracker):

    template_path: str | None = None

    loader: Loader | None = None
    matcher: Matcher | None = None
    
    radius: float = 200
    offset: float = 10
    ect_offset: float = 10

    result: ndarray | None = None
    ect_template: ndarray | None = None

    trajectory: list[tuple[float]] = field(default_factory=list)

    def setup(self, **params):
        
        if self.loader is None:
            self.loader = FilepathLoader(
                radius = self.radius,
                offset = self.offset
            )

        if self.matcher is None:
            self.matcher = RotShiftMatcher(
                transformer = UnfilteredTransformation(
                    self.offset,
                    self.ect_offset
                ),
                center_fnc = center_of_mass,
                bp_thresh = 0.1
            )
        
        self.ect_template = self.loader.load(self.template_path)
        self.ect_template = self.matcher.prepare(self.ect_template)
        self.trajectory.append((100, 0))

    def match(self, image: ndarray):
        result = super().match(image)
        scale, angle = self.match_result
        print(f"{scale=}, {angle=}")
        self.trajectory.append(self._calc_shift(scale, angle))
        return result

    def _calc_shift(self, scale, angle) -> tuple[float, float]:
        # fovea

        angle = (angle - int(self.radius*np.pi)/2)/100
        scale = (2.4*(scale+1) - 141.6)/100

        cx, cy = (0, 0)

        # get last position
        prev_x, prev_y = self.trajectory[0]

        prev_x -= cx
        prev_y -= cy

        prev_phi = math.atan2(prev_y, prev_x)
        prev_rho = 0.5*math.log(prev_x**2 + prev_y**2)

        phi = prev_phi - angle
        rho = prev_rho + math.log(scale)

        curr_x = math.exp(rho)*math.cos(phi)
        curr_y = math.exp(rho)*math.sin(phi)

        return curr_x, curr_y

    def show_result(self, label: str):
        
        trajectory = np.array(self.trajectory)

        plt.figure(figsize=(10, 10))
        plt.xlim((-200, 200))
        plt.ylim((-200, 200))
        plt.grid(True)
        plt.plot(trajectory[:, 0], trajectory[:, 1])
        plt.xlabel("Relative scale factor")
        plt.ylabel("Relative rotation")
        plt.title(label)

@dataclass
class DynamicTemplateTracker(Tracker):

    template_path: str | None = None

    loader: Loader | None = None
    matcher: Matcher | None = None
    
    radius: float = 200
    offset: float = 10
    ect_offset: float = 10

    result: ndarray | None = None
    ect_template: ndarray | None = None

    trajectory: list[tuple[float]] = field(default_factory=list)

    def setup(self, **params):
        
        if self.loader is None:
            self.loader = FilepathLoader(
                radius = self.radius,
                offset = self.offset
            )

        if self.matcher is None:
            self.matcher = RotShiftMatcher(
                transformer = UnfilteredTransformation(
                    self.offset,
                    self.ect_offset
                ),
                center_fnc = center_of_mass,
                bp_thresh = 0.1
            )
        
        self.ect_template = self.loader.load(self.template_path)
        self.ect_template = self.matcher.prepare(self.ect_template)

    def match(self, image: ndarray):
        result = super().match(image)
        scale, angle = self.match_result
        print(f"{scale=}, {angle=}")
        self.trajectory.append((scale, angle))
        return result


    def show_result(self, label: str):
        
        trajectory = np.array(self.trajectory)

        plt.figure(figsize=(10, 10))
        plt.plot(trajectory[:, 0], trajectory[:, 1])
        plt.xlabel("Relative scale factor")
        plt.ylabel("Relative rotation")
        plt.title(label)


    def dump_result(self, filepath: str):
        pass

