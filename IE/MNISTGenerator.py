import cv2
import sys
from .NetworkService import NetworkService

class MNISTGenerator(NetworkService):

    def __init__(self, delay=-1):
        super(MNISTGenerator, self).__init__()
        self.delay = delay
