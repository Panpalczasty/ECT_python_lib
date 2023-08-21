import cv2
import numpy as np
import ect

from numpy.random import rand
import scipy.optimize as optim
import matplotlib.pyplot as plt
from typing import Callable

from abc import ABC, abstractmethod

class ECTOptimizer(ABC):

    def __init__(
        self,
        image_path: str,
        radius: int,
        img_offset: float,
        loss_fcn: Callable[[np.ndarray, np.ndarray], float]
    ):        
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        self.loss_fcn = loss_fcn
        self.radius = radius
        self.img_offset = img_offset

        self.image = ect.logpolar(image, radius, offset=img_offset)
        self.filt = ect.sidelobe(self.image.shape[:2], offset=img_offset)

        self.image = self.image * self.filt

    
    def start(self, N): 
        return rand(N)


    def loss(self, params: list[float]) -> float:
        return np.real(self.loss_fcn(self.transform(params), self.image))


    def show_result(
        self, 
        norm_fcn: Callable[[np.ndarray], np.ndarray],
        *params
        ):

        if self.inv is None:
            self.inv = self.transform(params)

        log_img = norm_fcn(self.image)
        inv_img = norm_fcn(self.inv)

        diff = log_img - inv_img

        plt.figure(figsize = (20, 10))
        plt.subplot(1, 3, 1)
        plt.imshow(ect.complex_to_hsv(self.image))

        plt.subplot(1, 3, 2)
        plt.imshow(ect.complex_to_hsv(inv_img))

        plt.subplot(1, 3, 3)
        plt.imshow(ect.complex_to_hsv(diff))

    def callback(self, params: list[float]):
        loss = self.loss(params)
        print(f"Current {loss=:.3f}")


    @abstractmethod
    def optim(self): ...


    @abstractmethod
    def transform(self, params: list[float]) -> np.ndarray: ...    


class ECTOffsetOptimizer(ECTOptimizer):

    def transform(self, params: list[float]) -> np.ndarray:

        ect_offset = params[0]

        ect_img = ect.fect(self.image, self.img_offset, ect_offset)
        self.inv = ect.ifect(ect_img, self.img_offset, ect_offset)

        return self.inv


    def loss_sidelobe(self, x) -> float:

        self.inv = self.transform(x)
        self.inv = self.inv * self.filt

        return np.real(self.loss_fcn(self.inv, self.image))


    def callback(self, x):
        ect_offset = x[0]
        loss = self.loss(x)
        print(f"{ect_offset=:.2f}, {loss=:.2f}")


    def start(self, N):
        return rand(1)*self.radius/4
    

    def optim(self, sidelobe: bool= False, **kwargs):

        fnc = self.loss_sidelobe if sidelobe else self.loss

        result = optim.minimize(
            fun = fnc,
            x0 = [self.start(2)],
            bounds = [[0, self.radius]],
            callback=self.callback,
            **kwargs
        )

        return result
    

class ECTFilterOptimizer(ECTOptimizer):

    def __init__(
        self, 
        image_path: str, 
        radius: int, 
        img_offset: float,
        ect_offset: float, 
        n_knots: int,
        loss_fcn: Callable[[np.ndarray, np.ndarray], float],
        do_fnf: bool = True,
        do_snf: bool = True):

        self.params = 0

        self.do_fnf = do_fnf
        self.do_snf = do_snf

        if do_fnf:
            self.params += n_knots

        if do_snf:
            self.params += n_knots
        
        self.ect_offset = ect_offset
        super().__init__(image_path, radius, img_offset, loss_fcn)


    def transform(self, params: list[float]) -> np.ndarray:

        if self.do_fnf and self.do_snf:    
            fnf_values = params[:self.params//2]
            snf_values = params[self.params//2:]    
        elif self.do_fnf:
            fnf_values = params
            snf_values = None
        elif self.do_snf:
            snf_values = params
            fnf_values = None
        else:
            raise Exception("At least one of two must be True.")

        dsize = self.image.shape[:2]
        fnf = ect.freqnorm(dsize, self.radius, fnf_values)
        snf = ect.spacenorm(dsize, self.radius, snf_values)
        ang = ect.angular_filter(dsize)

        self.ect_img = ect.fect(self.image, self.img_offset, self.ect_offset)
        self.ect_img = self.ect_img * fnf
        
        self.inv = ect.ifect(self.ect_img, self.img_offset, self.ect_offset)
        self.inv = self.inv * snf * ang

        return self.inv


    def optim(self, **kwargs) -> optim.OptimizeResult:

        result = optim.minimize(
            fun = self.loss,
            x0 = self.start(self.params),
            callback = self.callback,
            **kwargs
        )

        return result
    

class ECTFilterTuner(ECTOptimizer):

    def __init__(
        self, 
        image_path: str, 
        radius: int, 
        img_offset: float,
        ect_offset: float, 
        loss_fcn: Callable[[np.ndarray, np.ndarray], float],
        n_knots: int = 20):

        self.params = 2*n_knots
        self.ect_offset = ect_offset
        super().__init__(image_path, radius, img_offset, loss_fcn)


    def start(self, N):

        aa_params = np.array([0.27, 0.15, 0.27, 0.15, 0.25, 0.25])

        return np.r_[ect.DEFAULT_FNF, ect.DEFAULT_SNF, aa_params]


    def transform(self, params: list[float]) -> np.ndarray:

        fnf_values = params[:self.params//2]
        snf_values = params[self.params//2:self.params]
        aa_params = params[self.params:]
        ect_aa = aa_params[:2]
        iect_aa = aa_params[2:4]
        ect_slope = aa_params[4]
        iect_slope = aa_params[5]

        dsize = self.image.shape[:2]
        fnf = ect.freqnorm(dsize, self.radius, fnf_values)
        snf = ect.spacenorm(dsize, self.radius, snf_values)
        # ang = ect.angular_filter(dsize)

        self.ect_img = ect.fect(
            self.image, 
            self.img_offset, 
            self.ect_offset,
            aa_factors = ect_aa,
            aa_slope = ect_slope)
        
        self.ect_img = self.ect_img * fnf
        
        self.inv = ect.ifect(
            self.ect_img, 
            self.img_offset, 
            self.ect_offset,
            aa_factors = iect_aa,
            aa_slope = iect_slope)
        
        self.inv = self.inv * snf

        return self.inv
    

    def optim(self, **kwargs) -> optim.OptimizeResult:

        result = optim.minimize(
            fun = self.loss,
            x0 = self.start(self.params),
            callback = self.callback,
            **kwargs
        )

        return result


class ECTFullOptimizer(ECTOptimizer):

    def __init__(
        self, 
        image_path: str, 
        radius: int, 
        img_offset: float,
        ect_offset: float, 
        n_knots: int,
        loss_fcn: Callable[[np.ndarray, np.ndarray], float]):

        self.ect_offset = ect_offset
        self.num_knots = n_knots
        super().__init__(image_path, radius, img_offset, loss_fcn)


    def transform(self, params: list[float]) -> np.ndarray:
        
        fnf_values = params[:self.num_knots]
        snf_values = params[self.num_knots:2*self.num_knots]
        aa_params = params[2*self.num_knots:]
        ect_aa_factors = aa_params[0:2]
        # ect_aa_thresholds = aa_params[2:4]
        iect_aa_factors = aa_params[2:4]
        # iect_aa_thresholds = aa_params[6:8]
        # ect_aa_thresholds = params[22:24]
        # ect_aa

        dsize = self.image.shape[:2]

        self.ect_img = ect.fect(
            self.image, self.img_offset, self.ect_offset,
            aa_factors = ect_aa_factors)
            # aa_thresholds = ect_aa_thresholds)
        fnf = ect.freqnorm(dsize, self.radius, fnf_values)
        self.ect_img = self.ect_img * fnf
        
        self.inv = ect.ifect(
            self.ect_img, self.img_offset, self.ect_offset,
            aa_factors=iect_aa_factors)
        snf = ect.spacenorm(dsize, self.radius, snf_values)
        self.inv = self.inv * snf

        return self.inv


    def optim(self, **kwargs) -> optim.OptimizeResult:

        result = optim.minimize(
            fun = self.loss,
            x0 = self.start(self.num_knots*2+4),
            callback = self.callback,
            **kwargs
        )

        return result
    

class ECTAntialiasOptimizer(ECTOptimizer):

    def __init__(
        self, 
        image_path: str, 
        radius: int, 
        img_offset: float,
        ect_offset: float, 
        loss_fcn: Callable[[np.ndarray, np.ndarray], float]):

        self.ect_offset = ect_offset
        super().__init__(image_path, radius, img_offset, loss_fcn)

    def transform(self, params: list[float]) -> np.ndarray:

        ect_aa = params[:2]
        iect_aa = params[2:4]
        ect_slope = params[4]
        iect_slope = params[5]
        
        dsize = self.image.shape[:2]

        self.ect_img = ect.fect(
            self.image, self.img_offset, self.ect_offset,
            aa_factors = ect_aa,
            aa_slope = ect_slope)
            # aa_thresholds = ect_aa_thresholds)
        fnf = ect.freqnorm(dsize, self.radius)
        self.ect_img = self.ect_img * fnf
        
        self.inv = ect.ifect(
            self.ect_img, self.img_offset, self.ect_offset,
            aa_factors = iect_aa,
            aa_slope = iect_slope)
        snf = ect.spacenorm(dsize, self.radius)
        self.inv = self.inv * snf

        return self.inv
    
    def start(self, N):
        return [0.27, 0.13, 0.27, 0.13, 0.25, 0.25]

    def optim(self, **kwargs) -> optim.OptimizeResult:

        result = optim.minimize(
            fun = self.loss,
            x0 = self.start(4),
            callback = self.callback,
            **kwargs
        )

        return result
    

class ECTAngularSNF(ECTOptimizer):

    def __init__(
        self, 
        image_path: str, 
        radius: int, 
        img_offset: float,
        ect_offset: float, 
        n_knots: float,
        loss_fcn: Callable[[np.ndarray, np.ndarray], float]):

        self.params = n_knots
        self.ect_offset = ect_offset
        super().__init__(image_path, radius, img_offset, loss_fcn)


    def start(self, N):
        return rand(N)
    

    def transform(self, params: list[float]) -> np.ndarray:

        dsize = self.image.shape[:2]
        fnf = ect.freqnorm(dsize, self.radius)
        snf = ect.spacenorm(dsize, self.radius)

        self.ect_img = ect.fect(self.image, self.img_offset, self.ect_offset)
        self.ect_img = self.ect_img * fnf
        
        self.inv = ect.ifect(self.ect_img, self.img_offset, self.ect_offset)

        ang = ect.angular_filter(dsize, params)

        self.inv = self.inv * snf * ang

        return self.inv


    def optim(self, **kwargs) -> optim.OptimizeResult:

        result = optim.minimize(
            fun = self.loss,
            x0 = self.start(self.params),
            callback = self.callback,
            **kwargs
        )

        return result
    

class ECTAngularFNF(ECTOptimizer):

    def __init__(
        self, 
        image_path: str, 
        radius: int, 
        img_offset: float,
        ect_offset: float, 
        n_knots: float,
        loss_fcn: Callable[[np.ndarray, np.ndarray], float]):

        self.params = n_knots
        self.ect_offset = ect_offset
        super().__init__(image_path, radius, img_offset, loss_fcn)


    def start(self, N):
        return np.ones((N,))
    

    def transform(self, params: list[float]) -> np.ndarray:

        dsize = self.image.shape[:2]
        fnf = ect.freqnorm(dsize, self.radius)
        snf = ect.spacenorm(dsize, self.radius)
        ang = ect.freqnorm_phi(dsize, params)


        self.ect_img = ect.fect(self.image, self.img_offset, self.ect_offset)
        self.ect_img = self.ect_img * fnf * ang
        
        self.inv = ect.ifect(self.ect_img, self.img_offset, self.ect_offset)

        self.inv = self.inv * snf

        return self.inv


    def optim(self, **kwargs) -> optim.OptimizeResult:

        result = optim.minimize(
            fun = self.loss,
            x0 = self.start(self.params),
            callback = self.callback,
            **kwargs
        )

        return result