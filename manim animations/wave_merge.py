"""
This is 3blue1brown's code for making animations
need to do pip install manim
"""

import argparse
import os
import sys

from manim import * # type: ignore


def parse_custom_frequencies():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--f1", type=float, default=None)
    parser.add_argument("--f2", type=float, default=None)
    parser.add_argument("--freq1", type=float, default=None)
    parser.add_argument("--freq2", type=float, default=None)
    args, remaining = parser.parse_known_args()

    f1 = args.f1 if args.f1 is not None else args.freq1
    f2 = args.f2 if args.f2 is not None else args.freq2

    if f1 is None:
        f1_env = os.getenv("WAVE_MERGE_F1")
        if f1_env is not None:
            f1 = float(f1_env)
    if f2 is None:
        f2_env = os.getenv("WAVE_MERGE_F2")
        if f2_env is not None:
            f2 = float(f2_env)

    # Strip custom flags from sys.argv so Manim does not treat them as script tokens.
    cleaned_argv = [sys.argv[0]]
    skip_next = False
    for arg in sys.argv[1:]:
        if skip_next:
            skip_next = False
            continue
        if arg.startswith("--f1=") or arg.startswith("--f2=") or arg.startswith("--freq1=") or arg.startswith("--freq2="):
            continue
        if arg in {"--f1", "--f2", "--freq1", "--freq2"}:
            skip_next = True
            continue
        cleaned_argv.append(arg)

    sys.argv[:] = cleaned_argv
    return float(f1) if f1 is not None else 2.0, float(f2) if f2 is not None else 3.0


F1, F2 = parse_custom_frequencies()


class _BaseWaveScene(Scene):
    def setup_data(self):
        self.f1, self.f2 = F1, F2
        self.t_max = 2

        self.w1_func = lambda t: np.sin(2 * np.pi * self.f1 * t)
        self.w2_func = lambda t: np.sin(2 * np.pi * self.f2 * t)
        self.combined_func = lambda t: self.w1_func(t) + self.w2_func(t)

        # Fixed top and bottom axes (already in position)
        self.axes_top = Axes(
            x_range=[0, self.t_max, 0.5], y_range=[-1.5, 1.5, 1],
            x_length=9, y_length=2, axis_config={"color": GREY_B}
        ).shift(UP * 2)

        self.axes_bottom = Axes(
            x_range=[0, self.t_max, 0.5], y_range=[-1.5, 1.5, 1],
            x_length=9, y_length=2, axis_config={"color": GREY_B}
        ).shift(DOWN * 2)

        # Center axes
        self.axes_mid = Axes(
            x_range=[0, self.t_max, 0.5], y_range=[-2.5, 2.5, 1],
            x_length=9, y_length=4.5, axis_config={"color": GREY_B}
        ).move_to(ORIGIN)

        # Waves on their respective axes
        self.wave1 = self.axes_top.plot(self.w1_func, color=BLUE, x_range=[0, self.t_max])
        self.wave2 = self.axes_bottom.plot(self.w2_func, color=RED, x_range=[0, self.t_max])

        self.label1 = Text(f"{self.f1:g} Hz Signal", font_size=24, color=BLUE).next_to(self.axes_top, UP, buff=0.1)
        self.label2 = Text(f"{self.f2:g} Hz Signal", font_size=24, color=RED).next_to(self.axes_bottom, UP, buff=0.1)

        self.combined_wave = self.axes_mid.plot(self.combined_func, color=PURPLE, x_range=[0, self.t_max])
        self.label_combined = Text(f"{self.f1:g} Hz + {self.f2:g} Hz Combined Signal (Superposition)", font_size=26, color=PURPLE).to_edge(UP, buff=0.5)

    def smooth_unmerge(self, combined_wave, target_axes_top, target_axes_bottom, target_wave1, target_wave2, label1, label2):
        combined_top = combined_wave.copy()
        combined_bottom = combined_wave.copy()
        self.remove(combined_wave)

        self.play(
            FadeOut(self.axes_mid),
            FadeOut(self.label_combined),
            FadeIn(target_axes_top),
            FadeIn(target_axes_bottom),
            ReplacementTransform(combined_top, target_wave1),
            ReplacementTransform(combined_bottom, target_wave2),
            Write(label1),
            Write(label2),
            run_time=3,
            rate_func=smooth,
        )


# -------------------------------------------------------------
# 1. MERGE ONLY
# -------------------------------------------------------------
class MergeOnlyScene(_BaseWaveScene):
    def construct(self):
        self.setup_data()

        # Display top and bottom graphs
        self.play(Create(self.axes_top), Create(self.axes_bottom))
        self.play(Create(self.wave1), Write(self.label1), Create(self.wave2), Write(self.label2))
        self.wait(1)

        wave1_mid = self.axes_mid.plot(self.w1_func, color=BLUE, x_range=[0, self.t_max])
        wave2_mid = self.axes_mid.plot(self.w2_func, color=RED, x_range=[0, self.t_max])

        # Slide vertically into center grid
        self.play(
            ReplacementTransform(self.axes_top, self.axes_mid),
            ReplacementTransform(self.axes_bottom, self.axes_mid),
            ReplacementTransform(self.wave1, wave1_mid),
            ReplacementTransform(self.wave2, wave2_mid),
            FadeOut(self.label1),
            FadeOut(self.label2),
            run_time=2,
            rate_func=smooth,
        )
        self.wait(0.5)

        # Merge two separate curves into combined superposition wave
        self.play(
            ReplacementTransform(VGroup(wave1_mid, wave2_mid), self.combined_wave),
            Write(self.label_combined),
            run_time=2,
            rate_func=smooth,
        )
        self.wait(2)


# -------------------------------------------------------------
# REVERSAL ONLY (Fading in fixed axes while wave splits)
# -------------------------------------------------------------
class SmoothUnmergeScene(_BaseWaveScene):
    def construct(self):
        self.setup_data()

        # Start with center axes and combined wave
        self.add(self.axes_mid, self.combined_wave, self.label_combined)
        self.wait(1)

        self.smooth_unmerge(
            self.combined_wave,
            self.axes_top,
            self.axes_bottom,
            self.wave1,
            self.wave2,
            self.label1,
            self.label2,
        )

        self.wait(2)


# -------------------------------------------------------------
# FULL ANIMATION (Merge + Smooth Unmerge)
# -------------------------------------------------------------
class FullAnimationScene(_BaseWaveScene):
    def construct(self):
        self.setup_data()

        # --- PHASE 1: MERGE ---
        self.play(Create(self.axes_top), Create(self.axes_bottom))
        self.play(Create(self.wave1), Write(self.label1), Create(self.wave2), Write(self.label2))
        self.wait(1)

        wave1_mid = self.axes_mid.plot(self.w1_func, color=BLUE, x_range=[0, self.t_max])
        wave2_mid = self.axes_mid.plot(self.w2_func, color=RED, x_range=[0, self.t_max])

        self.play(
            ReplacementTransform(self.axes_top, self.axes_mid),
            ReplacementTransform(self.axes_bottom, self.axes_mid),
            ReplacementTransform(self.wave1, wave1_mid),
            ReplacementTransform(self.wave2, wave2_mid),
            FadeOut(self.label1),
            FadeOut(self.label2),
            run_time=2,
            rate_func=smooth,
        )
        self.wait(0.5)

        self.play(
            ReplacementTransform(VGroup(wave1_mid, wave2_mid), self.combined_wave),
            Write(self.label_combined),
            run_time=2,
            rate_func=smooth,
        )
        self.wait(2)

        # --- PHASE 2: SMOOTH UNMERGE ---
        axes_top_target = Axes(
            x_range=[0, self.t_max, 0.5], y_range=[-1.5, 1.5, 1],
            x_length=9, y_length=2, axis_config={"color": GREY_B}
        ).shift(UP * 2)
        axes_bottom_target = Axes(
            x_range=[0, self.t_max, 0.5], y_range=[-1.5, 1.5, 1],
            x_length=9, y_length=2, axis_config={"color": GREY_B}
        ).shift(DOWN * 2)
        wave1_target = axes_top_target.plot(self.w1_func, color=BLUE, x_range=[0, self.t_max])
        wave2_target = axes_bottom_target.plot(self.w2_func, color=RED, x_range=[0, self.t_max])
        label1_target = Text("2 Hz Signal", font_size=24, color=BLUE).next_to(axes_top_target, UP, buff=0.1)
        label2_target = Text("3 Hz Signal", font_size=24, color=RED).next_to(axes_bottom_target, UP, buff=0.1)

        self.smooth_unmerge(
            self.combined_wave,
            axes_top_target,
            axes_bottom_target,
            wave1_target,
            wave2_target,
            label1_target,
            label2_target,
        )
        self.wait(2)