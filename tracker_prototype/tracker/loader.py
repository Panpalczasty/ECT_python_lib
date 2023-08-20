import cv2
import ect

from numpy import ndarray

from abc import ABC, abstractmethod
from dataclasses import dataclass

class Loader(ABC):

    @abstractmethod
    def load(self, *args, **kwargs) -> ndarray: ...


@dataclass
class ImageLoader(Loader):

    filepath: str
    image: ndarray | None = None
    radius: int = 200
    offset: float = 20

    def load(self) -> ndarray:
        
        if self.image is not None:
            return self.image
        
        self.image = cv2.imread(self.filepath)
        self.image = cv2.cvtColor(self.image, code=cv2.COLOR_BGR2GRAY)
        self.image = ect.logpolar(self.image, radius=self.radius, offset=self.offset)

        sidelobe = ect.sidelobe(self.image.shape[:2], offset=self.offset)

        self.image = self.image * sidelobe

        return self.image


@dataclass
class FilepathLoader(Loader):

    filepath: str | None = None
    image: ndarray | None = None
    radius: int = 200
    offset: float = 20

    def load(self, filepath: str) -> ndarray:

        self.image = cv2.imread(filepath)
        self.image = cv2.cvtColor(self.image, code=cv2.COLOR_BGR2GRAY)
        self.image = ect.logpolar(self.image, radius=self.radius, offset=self.offset)

        sidelobe = ect.sidelobe(self.image.shape[:2], offset=self.offset)

        self.image = self.image * sidelobe

        return self.image

@dataclass
class LogpolarLoader(Loader):

    radius: int = 200
    offset: int = 10

    def load(self, image: ndarray) -> ndarray:
        
        image = cv2.cvtColor(image, code=cv2.COLOR_BGR2GRAY)
        image = ect.logpolar(image, radius=self.radius, offset=self.offset)

        sidelobe = ect.sidelobe(image.shape[:2], offset=self.offset)

        return image * sidelobe    

# @dataclass
# class SequenceLoader(ImageLoader):

#     dirpath: str
#     root_name: str
#     ftype: str
#     index: int = -1
#     filepath: str | None = None
#     image: ndarray | None = None

#     def _iterate(self):
#         self.index += 1
#         self.filepath = f"{self.dirpath}/{self.root_name}{self.index}.{self.ftype}"


#     def load(self) -> ndarray:
#         self._iterate()
#         return super().load()

@dataclass
class CartesianLoader(Loader):

    filepath: str
    image: ndarray | None = None
    
    def load(self) -> ndarray:
        
        if self.image is not None:
            return self.image
        
        self.image = cv2.imread(self.filepath)
        self.image = cv2.cvtColor(self.image, code=cv2.COLOR_BGR2GRAY)
        return self.image

@dataclass
class CartesianFilepathLoader(CartesianLoader):

    filepath: str | None = None

    def load(self, filepath: str) -> ndarray:
        self.filepath = filepath
        return super().load()