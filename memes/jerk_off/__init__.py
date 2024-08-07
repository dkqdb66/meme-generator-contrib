from pathlib import Path
import cv2
import numpy as np
from meme_generator import add_meme
from meme_generator.utils import FrameAlignPolicy, Maker, make_gif_or_combined_gif
from pil_utils import BuildImage

img_dir = Path(__file__).parent / "images"


def jerk_off(images: list[BuildImage], texts, args):
    image = cv2.cvtColor(np.array(images[0].image), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    cascade = cv2.CascadeClassifier(Path(__file__).parent /"lbpcascade_animeface.xml")
    faces = cascade.detectMultiScale(gray,
                                     scaleFactor = 1.1,
                                     minNeighbors = 10,
                                     minSize = (24, 24))
    if len(faces) == 0:
        img_w, img_h = images[0].size
        crop_pos = (0, 0, img_w, img_h)
    else:
        (x, y, w, h) = faces[0]
        img_w, img_h = w,h
        left = round(w*0.1)
        up = round(h*0.12)
        crop_pos = (max(0,x-left),max(0,y-up), x+w-left, y+h-up)
    jerk_w, jerk_h = BuildImage.open(img_dir / "0.png").size
    if img_w / img_h > jerk_w / jerk_h:
        frame_h = jerk_h
        frame_w = round(frame_h * img_w / img_h)
    else:
        frame_w = jerk_w
        frame_h = round(frame_w * img_h / img_w)

    def maker(i: int) -> Maker:
        def make(img: BuildImage) -> BuildImage:
            frame = img.convert("RGBA").crop(crop_pos).resize((frame_w, frame_h), keep_ratio=True)
            jerk = BuildImage.open(img_dir / f"{i}.png")
            frame.paste(jerk, ((frame_w - jerk_w) // 2, frame_h - jerk_h), alpha=True)
            return frame

        return make

    return make_gif_or_combined_gif(
        images[0], maker, 8, 0.1, FrameAlignPolicy.extend_loop
    )


add_meme("jerk_off", jerk_off, min_images=1, max_images=1, keywords=["打胶"])
