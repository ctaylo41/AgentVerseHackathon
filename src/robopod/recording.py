"""Clean, fixed-rate local MP4 exports; no overlays, audio or recovery transitions."""
from pathlib import Path
import cv2


class Recorder:
    def __init__(self, path, width, height, fps=10):
        self.path = Path(path)
        self.frames = 0
        self.writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'mp4v'), fps, (width,height))
        if not self.writer.isOpened():
            raise OSError('MP4 encoder could not open the output file')

    def write(self, obs):
        self.writer.write(cv2.cvtColor(obs.rgb, cv2.COLOR_RGB2BGR))
        self.frames += 1

    def close(self):
        self.writer.release()


class Stopped(Exception):
    """User stopped a workflow; preserve its plan and partial artifacts."""
