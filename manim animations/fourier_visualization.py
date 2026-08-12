# MAKE SURE TO RUN THE FFT DATA BRIDGE FIRST!!!!
# INSTRUCTIONS ON HOW TO DO THAT IN ITS FILE
# to actually generate the video do this:
"""
ctrl + `
cd manim_animations
python fft_data_bridge.py
manim -pql fourier_visualization.py FFTDecomposition
"""
# if you want a higher quality video do either -pqm or -pqh, those are 720p 30 fps and 1080p 60fps
# -pql is 480p at 15 fps
# technically there is 4k but ur not doing that





"""
Manim COMMUNITY EDITION scene.

Loads fft_viz_data.npz (produced by fft_data_bridge.py) and animates:
  1. The original signal drawing itself in, top-left.
  2. Empty, labeled spectrum axes appear on the right (vertically
     centered, sized to fill essentially all the remaining vertical
     room in that column).
  3. The signal splits apart into its dominant frequency components
     ONE AT A TIME, in ascending frequency order, stacked in a column
     under the original signal. Each time a component peels off:
       - its own small axes + "X.X Hz" label fade in (label sits ABOVE
         its axes, outside the plotted area, so it can never overlap
         the wave itself)
       - a copy of the signal transforms into that pure sine wave
       - AT THE SAME TIME, the next slice of the spectrum curve (from
         the previous cutoff frequency up through this component's
         frequency) draws itself in with Create() -- the same
         progressive stroke-by-stroke build the original signal uses,
         not a transform/morph from anything. So the spectrum only
         ever shows as much of itself as has actually been "explained"
         by components peeled off so far, and it never draws past the
         last component you've split off.
       - a small frequency label appears at that peak on the spectrum
         itself (no dot marker).

HARD FIT: all vertical space is computed from the actual frame size
(config.frame_height / config.frame_width) up front. The component
stack (axes + label, as one unit) is measured against the space
actually left on screen and scaled down together if needed, so labels
and axes shrink in proportion and neither can run off the frame.

Render with:
    manim -pql fourier_visualization.py FFTDecomposition   # quick draft
    manim -pqh fourier_visualization.py FFTDecomposition   # final quality
"""

from manim import * # type: ignore
import numpy as np

DATA_PATH = "fft_viz_data.npz"
COMPONENT_COLORS = [RED, GREEN, ORANGE, TEAL, PINK, MAROON, GOLD, PURPLE_B]
SIGNAL_COLOR = BLUE
SPECTRUM_COLOR = YELLOW
AXIS_LABEL_COLOR = GREY_B


class FFTDecomposition(Scene):
    def construct(self):
        data = np.load(DATA_PATH)
        t = data["t"]
        signal_real = data["signal_real"]
        comp_freqs = data["component_freqs"]
        comp_amps = data["component_amps"]
        comp_phases = data["component_phases"]
        freqs = data["freqs"]
        magnitude = data["magnitude"]
        duration = float(data["duration"])
        sample_rate = float(data["sample_rate"])
        n_fft = len(freqs)
        n_components = len(comp_freqs)

        # ---------- One-sided, amplitude-normalized spectrum ----------
        nyquist = sample_rate / 2
        pos_mask = (freqs >= 0) & (freqs <= nyquist)
        freqs_pos = freqs[pos_mask]
        mag_pos = magnitude[pos_mask] / n_fft
        interior = (freqs_pos > 0) & (freqs_pos < nyquist)
        mag_pos[interior] *= 2

        # Process components in ascending frequency order so the spectrum
        # builds left-to-right in sync with each one peeling off.
        order = np.argsort(comp_freqs)
        sorted_freqs = comp_freqs[order]
        sorted_amps = comp_amps[order]
        sorted_phases = comp_phases[order]
        sorted_colors = [COMPONENT_COLORS[i % len(COMPONENT_COLORS)] for i in range(n_components)]

        # ==================================================================
        # HARD LAYOUT GEOMETRY -- derived from actual frame dimensions, not
        # guessed constants, so it holds for any resolution/aspect ratio
        # and any number of components.
        # ==================================================================
        frame_w = config.frame_width
        frame_h = config.frame_height

        outer_margin = 0.4
        col_gap = 0.5
        title_h = 0.35

        left_col_left = -frame_w / 2 + outer_margin
        left_col_right = -col_gap / 2
        left_x = (left_col_left + left_col_right) / 2
        col_width = left_col_right - left_col_left

        right_col_left = col_gap / 2
        right_col_right = frame_w / 2 - outer_margin
        right_x = (right_col_left + right_col_right) / 2

        top_y = frame_h / 2 - outer_margin
        bottom_y = -frame_h / 2 + outer_margin

        # ---------- Original signal: top of the left column ----------
        signal_h = 1.7
        y_pad = (signal_real.max() - signal_real.min()) * 0.15 + 1e-9
        signal_axes = Axes(
            x_range=[0, duration, duration / 5],
            y_range=[signal_real.min() - y_pad, signal_real.max() + y_pad,
                      (signal_real.max() - signal_real.min()) / 4 + 1e-9],
            x_length=col_width, y_length=signal_h,
            tips=True,
        )
        signal_axes.move_to(np.array([left_x, top_y - title_h - signal_h / 2, 0]))
        signal_title = Text("Original Signal", font_size=22, color=SIGNAL_COLOR).next_to(
            signal_axes, UP, buff=0.08
        )

        signal_line = signal_axes.plot_line_graph(
            x_values=t, y_values=signal_real,
            add_vertex_dots=False,
            line_color=SIGNAL_COLOR,
            stroke_width=3,
        )["line_graph"]

        self.play(Create(signal_axes), Write(signal_title), run_time=1.5)
        self.play(Create(signal_line), run_time=3.5, rate_func=smooth)
        self.wait(0.5)

        # ---------- Spectrum: vertically centered in the right column ----------
        # Fill essentially all remaining vertical room in this column --
        # only a title above and an x-axis label below are reserved.
        x_label_h = 0.3
        available_h_right = frame_h - 2 * outer_margin
        spectrum_h = available_h_right - title_h - x_label_h
        spectrum_x_length = col_width - 0.35  # leave a sliver on the left for the y-axis label

        spectrum_axes = Axes(
            x_range=[0, nyquist, nyquist / 5],
            y_range=[0, mag_pos.max() * 1.25, mag_pos.max() / 4],
            x_length=spectrum_x_length, y_length=spectrum_h,
            tips=True,
        )
        spectrum_title = Text(
            "Frequency Spectrum |FFT|", font_size=22, color=SPECTRUM_COLOR
        ).next_to(spectrum_axes, UP, buff=0.08)
        spectrum_x_label = Text(
            "Frequency (Hz)", font_size=16, color=AXIS_LABEL_COLOR
        ).next_to(spectrum_axes, DOWN, buff=0.12)
        spectrum_y_label = Text(
            "Magnitude", font_size=16, color=AXIS_LABEL_COLOR
        ).rotate(PI / 2).next_to(spectrum_axes, LEFT, buff=0.12)

        spectrum_group = VGroup(spectrum_title, spectrum_axes, spectrum_x_label, spectrum_y_label)
        spectrum_group.move_to(np.array([right_x, 0, 0]))

        self.play(
            Create(spectrum_axes),
            Write(spectrum_title),
            Write(spectrum_x_label),
            Write(spectrum_y_label),
            run_time=1.8,
        )
        self.wait(0.3)

        # ---------- Component waves: stacked below the signal, hard-fit ----------
        # Each unit = (its own mini axes) + (its "X.X Hz" label ABOVE the
        # axes, outside the plotted area -- so the label geometrically
        # cannot overlap the sine curve drawn inside the axes below it).
        signal_bottom_y = signal_axes.get_bottom()[1]
        gap_after_signal = 0.15
        available_stack_h = (signal_bottom_y - gap_after_signal) - bottom_y

        component_units = []
        for i in range(n_components):
            freq_i = float(sorted_freqs[i])
            amp_span = max(float(sorted_amps[i]), 1e-6)
            color = sorted_colors[i]

            ax = Axes(
                x_range=[0, duration, duration / 5],
                y_range=[-amp_span * 1.3, amp_span * 1.3, amp_span / 2],
                x_length=col_width, y_length=0.75,
                tips=False,
            )
            label = Text(f"{freq_i:.1f} Hz", font_size=13, color=color)
            label.next_to(ax, UP, buff=0.08).align_to(ax, LEFT)
            component_units.append(VGroup(ax, label))

        stack = VGroup(*component_units)
        stack.arrange(DOWN, buff=0.22)
        natural_stack_h = stack.height

        # Hard fit: scale axes+label together, as one unit, so labels
        # shrink right along with their axes and neither can spill past
        # the frame no matter how many components there are.
        if natural_stack_h > available_stack_h > 0:
            stack.scale(available_stack_h / natural_stack_h)

        group_top_y = signal_bottom_y - gap_after_signal
        stack.move_to(np.array([left_x, group_top_y - stack.height / 2, 0]))

        # ---------- Peel off components one at a time, building the ----------
        # ---------- spectrum curve in sync, segment by segment.       ----------
        prev_idx = 0  # index into freqs_pos / mag_pos of the last cutoff
        for i in range(n_components):
            freq = float(sorted_freqs[i])
            amp = float(sorted_amps[i])
            phase = float(sorted_phases[i])
            color = sorted_colors[i]

            ax, label = component_units[i]
            amp_span = max(amp, 1e-6)

            wave_graph = ax.plot(
                lambda x, f=freq, a=amp, p=phase: a * np.cos(2 * np.pi * f * x + p),
                x_range=[0, duration, duration / 500],
                color=color,
            )

            # Slice of the spectrum from the previous cutoff up through
            # this component's frequency -- this is the only part of the
            # transform graph that gets drawn this step.
            bin_idx = int(np.abs(freqs_pos - freq).argmin())
            bin_idx = max(bin_idx, prev_idx)
            seg_x = freqs_pos[prev_idx:bin_idx + 1]
            seg_y = mag_pos[prev_idx:bin_idx + 1]
            if len(seg_x) < 2:
                # Guard against a degenerate 1-point (or empty) slice.
                seg_x = freqs_pos[max(prev_idx - 1, 0):bin_idx + 1]
                seg_y = mag_pos[max(prev_idx - 1, 0):bin_idx + 1]
            spectrum_segment = spectrum_axes.plot_line_graph(
                x_values=seg_x, y_values=seg_y,
                add_vertex_dots=False,
                line_color=SPECTRUM_COLOR,
                stroke_width=3,
            )["line_graph"]

            # Peeling the wave off the signal and drawing the matching
            # spectrum slice happen together -- the spectrum slice draws
            # itself in with Create(), the same progressive stroke-by-
            # stroke build as the original signal, rather than morphing
            # in from a copy of anything.
            self.play(
                FadeIn(ax), FadeIn(label),
                TransformFromCopy(signal_line, wave_graph),
                Create(spectrum_segment),
                run_time=3.0,
                rate_func=smooth,
            )

            # Hold at the peak: label writes, then linger a bit longer
            # before moving on.
            peak_point = spectrum_axes.c2p(freqs_pos[bin_idx], mag_pos[bin_idx])
            peak_label = Text(f"{freq:.1f} Hz", font_size=14, color=color)
            peak_label.next_to(peak_point, UP, buff=0.1)
            self.play(Write(peak_label), run_time=0.9)
            self.wait(0.9)

            # Keep drawing a little past the peak -- enough that the curve
            # visibly comes back down off the peak instead of just
            # stopping dead on it, but stopping short of the NEXT
            # component's peak so we haven't already given that one away
            # before it actually splits off. For the last component there
            # is no "next" to hold back for, so it draws all the way out
            # to Nyquist instead of stopping flat on the final peak.
            if i < n_components - 1:
                next_freq = float(sorted_freqs[i + 1])
                next_bin_idx = max(int(np.abs(freqs_pos - next_freq).argmin()), bin_idx)
                gap = next_bin_idx - bin_idx
                continue_idx = bin_idx + max(1, gap // 2) if gap > 1 else bin_idx
                continue_idx = min(continue_idx, next_bin_idx - 1)
            else:
                continue_idx = len(freqs_pos) - 1

            if continue_idx > bin_idx:
                cont_x = freqs_pos[bin_idx:continue_idx + 1]
                cont_y = mag_pos[bin_idx:continue_idx + 1]
                continuation = spectrum_axes.plot_line_graph(
                    x_values=cont_x, y_values=cont_y,
                    add_vertex_dots=False,
                    line_color=SPECTRUM_COLOR,
                    stroke_width=3,
                )["line_graph"]
                self.play(Create(continuation), run_time=1.5, rate_func=smooth)
                prev_idx = continue_idx
            else:
                prev_idx = bin_idx

            self.wait(0.3)

        self.wait(2)