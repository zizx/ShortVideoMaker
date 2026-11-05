import numpy as np
from moviepy import Clip, ColorClip, CompositeVideoClip, vfx
from PIL import Image


# FadeIn
def fadein_transition(clip: Clip, t: float) -> Clip:
    return clip.with_effects([vfx.FadeIn(t)])


# FadeOut
def fadeout_transition(clip: Clip, t: float) -> Clip:
    return clip.with_effects([vfx.FadeOut(t)])


# SlideIn
def slidein_transition(clip: Clip, t: float, side: str) -> Clip:
    width, height = clip.size

    # MoviePy's built-in SlideIn is unstable for full-screen material in this
    # processing chain: the transition is logically applied, but the picture
    # barely changes. Use an explicit black background plus a position
    # animation instead, keeping the effect visible and the behavior
    # predictable.
    def position(current_time: float):
        progress = min(max(current_time / max(t, 0.001), 0), 1)

        if side == "left":
            return (-width + width * progress, 0)
        if side == "right":
            return (width - width * progress, 0)
        if side == "top":
            return (0, -height + height * progress)
        if side == "bottom":
            return (0, height - height * progress)
        return (0, 0)

    background = ColorClip(size=(width, height), color=(0, 0, 0)).with_duration(
        clip.duration
    )
    moving_clip = clip.with_position(position)
    return CompositeVideoClip([background, moving_clip], size=(width, height)).with_duration(
        clip.duration
    )


# SlideOut
def slideout_transition(clip: Clip, t: float, side: str) -> Clip:
    width, height = clip.size
    transition_duration = min(t, clip.duration)
    transition_start = max(clip.duration - transition_duration, 0)

    # SlideOut likewise uses an explicit position animation so the clip
    # reliably slides out of frame at the end.
    def position(current_time: float):
        if current_time <= transition_start:
            return (0, 0)

        progress = min(
            max((current_time - transition_start) / max(transition_duration, 0.001), 0), 1
        )

        if side == "left":
            return (-width * progress, 0)
        if side == "right":
            return (width * progress, 0)
        if side == "top":
            return (0, -height * progress)
        if side == "bottom":
            return (0, height * progress)
        return (0, 0)

    background = ColorClip(size=(width, height), color=(0, 0, 0)).with_duration(
        clip.duration
    )
    moving_clip = clip.with_position(position)
    return CompositeVideoClip([background, moving_clip], size=(width, height)).with_duration(
        clip.duration
    )


# Keep the original design's 20% zoom range so even a short three-second clip
# shows a clearly visible Ken Burns motion. Zoom stability comes from the
# sub-pixel center sampling below rather than hiding source-video encoding
# flicker by weakening the effect.
_ZOOM_MAX_SCALE = 1.2


def _zoom_frame(frame: np.ndarray, scale_factor: float) -> np.ndarray:
    """Stable, border-free zoom via sub-pixel center cropping.

    Do not convert the crop width/height to integers first: as the scale
    changes continuously, integer boundaries jump by different step sizes and
    shift the half-pixel sampling phase when switching between even and odd
    sizes, which shows up as frame jitter. Pillow's EXTENT transform accepts
    float boundaries directly and samples sub-pixel positions on a fixed output
    canvas; the left/right and top/bottom edges always stay symmetric around
    the same float center, which suits continuous slow zoom across a whole
    video.
    """
    if scale_factor <= 0:
        raise ValueError("scale_factor must be greater than zero")

    # A 1x zoom returns the original frame, avoiding pointless resampling that
    # would slightly blur the first frame.
    if abs(scale_factor - 1.0) < 1e-9:
        return frame

    height, width = frame.shape[:2]
    crop_width = width / scale_factor
    crop_height = height / scale_factor
    left = (width - crop_width) / 2
    top = (height - crop_height) / 2
    right = left + crop_width
    bottom = top + crop_height

    image = Image.fromarray(frame)
    transformed = image.transform(
        (width, height),
        Image.Transform.EXTENT,
        (left, top, right, bottom),
        # Continuous video zoom cares more about consistency between adjacent
        # frames. BICUBIC/LANCZOS are sharper per frame, but high-frequency
        # texture tends to ring and flicker in brightness when it crosses the
        # sampling grid; BILINEAR is softer and trades a little sharpness for a
        # more stable look in motion.
        resample=Image.Resampling.BILINEAR,
    )
    return np.asarray(transformed)


def zoomin_transition(clip: Clip, t: float) -> Clip:
    """Smoothly zoom from the original picture up to 1.2x over the whole clip."""
    # t is kept for a uniform call signature with the other transition
    # functions; the zoom must span the whole clip, otherwise the picture
    # suddenly freezes after a brief zoom, which does not suit static or
    # low-motion material.
    _ = t
    duration = max(clip.duration, 0.001)

    def scale_effect(get_frame, current_time: float):
        progress = min(max(current_time / duration, 0), 1)
        scale_factor = 1 + (_ZOOM_MAX_SCALE - 1) * progress
        return _zoom_frame(get_frame(current_time), scale_factor)

    return clip.transform(scale_effect, apply_to=["mask"])


def zoomout_transition(clip: Clip, t: float) -> Clip:
    """Smoothly zoom from 1.2x back down to the original picture over the whole
    clip."""
    # As in zoomin_transition, t only keeps the uniform transition call
    # interface.
    _ = t
    duration = max(clip.duration, 0.001)

    def scale_effect(get_frame, current_time: float):
        progress = min(max(current_time / duration, 0), 1)
        scale_factor = _ZOOM_MAX_SCALE - (_ZOOM_MAX_SCALE - 1) * progress
        return _zoom_frame(get_frame(current_time), scale_factor)

    return clip.transform(scale_effect, apply_to=["mask"])
