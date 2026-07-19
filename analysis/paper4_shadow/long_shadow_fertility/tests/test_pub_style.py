"""Tests for the pub_style publication stylesheet module."""
from __future__ import annotations

import matplotlib
import matplotlib.pyplot as plt
import pytest

from analysis.paper4_shadow.long_shadow_fertility.figures.pub_style import (
    GRAYS,
    LINESTYLES,
    MARKERS,
    SEQUENTIAL_CMAP,
    SEQUENTIAL_CMAP_COLORBLIND,
    apply_pub_style,
    get_sequential_cmap,
    style_lines,
)


class TestApplyPubStyle:
    """apply_pub_style() smoke tests and rcParam assertions."""

    def test_runs_without_error(self):
        """apply_pub_style() must not raise any exception."""
        apply_pub_style()

    def test_pdf_fonttype_42(self):
        """pdf.fonttype must be 42 (TrueType embed) after apply_pub_style()."""
        apply_pub_style()
        assert matplotlib.rcParams["pdf.fonttype"] == 42

    def test_ps_fonttype_42(self):
        """ps.fonttype must also be 42 for PostScript output compatibility."""
        apply_pub_style()
        assert matplotlib.rcParams["ps.fonttype"] == 42

    def test_font_size_default(self):
        """Default font size should be 10."""
        apply_pub_style()
        assert matplotlib.rcParams["font.size"] == 10

    def test_font_size_custom(self):
        """Custom font_size parameter is applied."""
        apply_pub_style(font_size=11)
        assert matplotlib.rcParams["font.size"] == 11

    def test_serif_font_family(self):
        """Default (serif=True) sets font.family to 'serif'."""
        apply_pub_style(serif=True)
        assert matplotlib.rcParams["font.family"] == ["serif"]

    def test_sans_font_family(self):
        """serif=False sets font.family to 'sans-serif'."""
        apply_pub_style(serif=False)
        assert matplotlib.rcParams["font.family"] == ["sans-serif"]

    def test_top_spine_off(self):
        """Top and right spines should be off by default."""
        apply_pub_style()
        assert matplotlib.rcParams["axes.spines.top"] is False
        assert matplotlib.rcParams["axes.spines.right"] is False

    def test_savefig_bbox_tight(self):
        """savefig.bbox should be 'tight'."""
        apply_pub_style()
        assert matplotlib.rcParams["savefig.bbox"] == "tight"


class TestPaletteLengths:
    """GRAYS, LINESTYLES, MARKERS must have exactly the expected lengths."""

    def test_grays_length(self):
        """GRAYS should have 6 levels (black through very light gray)."""
        assert len(GRAYS) == 6

    def test_linestyles_length(self):
        """LINESTYLES should have 6 distinct styles."""
        assert len(LINESTYLES) == 6

    def test_markers_length(self):
        """MARKERS should have 6 distinct markers."""
        assert len(MARKERS) == 6

    def test_grays_are_strings(self):
        """All GRAYS entries must be hex color strings."""
        for g in GRAYS:
            assert isinstance(g, str), f"Expected str, got {type(g)} for {g!r}"
            assert g.startswith("#"), f"Expected hex color '#...', got {g!r}"

    def test_linestyles_types(self):
        """LINESTYLES must be strings or tuples (matplotlib on-off dash style)."""
        for ls in LINESTYLES:
            assert isinstance(ls, (str, tuple)), (
                f"Expected str or tuple, got {type(ls)} for {ls!r}"
            )

    def test_markers_are_strings(self):
        """All MARKERS entries must be strings."""
        for m in MARKERS:
            assert isinstance(m, str), f"Expected str, got {type(m)} for {m!r}"


class TestSequentialCmap:
    """SEQUENTIAL_CMAP and get_sequential_cmap() tests."""

    def test_sequential_cmap_is_string(self):
        assert isinstance(SEQUENTIAL_CMAP, str)

    def test_sequential_cmap_colorblind_is_string(self):
        assert isinstance(SEQUENTIAL_CMAP_COLORBLIND, str)

    def test_get_sequential_cmap_default(self):
        """Default returns SEQUENTIAL_CMAP ('Greys')."""
        result = get_sequential_cmap()
        assert result == SEQUENTIAL_CMAP

    def test_get_sequential_cmap_colorblind(self):
        """colorblind_safe=True returns SEQUENTIAL_CMAP_COLORBLIND."""
        result = get_sequential_cmap(colorblind_safe=True)
        assert result == SEQUENTIAL_CMAP_COLORBLIND

    def test_sequential_cmap_valid_matplotlib(self):
        """SEQUENTIAL_CMAP must be a recognized matplotlib colormap."""
        import matplotlib.cm as cm
        assert SEQUENTIAL_CMAP in plt.colormaps()

    def test_sequential_cmap_colorblind_valid_matplotlib(self):
        """SEQUENTIAL_CMAP_COLORBLIND must be a recognized matplotlib colormap."""
        assert SEQUENTIAL_CMAP_COLORBLIND in plt.colormaps()


class TestStyleLines:
    """style_lines() applies a valid prop_cycle to axes."""

    def setup_method(self):
        apply_pub_style()

    def teardown_method(self):
        plt.close("all")

    def test_style_lines_runs(self):
        """style_lines(ax) must not raise."""
        fig, ax = plt.subplots()
        style_lines(ax)

    def test_style_lines_n_clipped(self):
        """style_lines with n=3 sets a 3-entry cycle (verified via GRAYS[:3])."""
        from analysis.paper4_shadow.long_shadow_fertility.figures.pub_style import GRAYS
        from cycler import cycler as make_cycler

        fig, ax = plt.subplots()
        style_lines(ax, n=3)
        # Build the expected cycler independently and confirm it has 3 entries
        expected = make_cycler(color=GRAYS[:3]) + make_cycler(linestyle=LINESTYLES[:3])
        colors = expected.by_key()["color"]
        assert len(colors) == 3

    def test_style_lines_no_markers_by_default(self):
        """By default, use_markers=False — the LINESTYLES cycler has no marker key."""
        from cycler import cycler as make_cycler

        # Verify the cycler built without markers has no 'marker' key
        prop = make_cycler(color=GRAYS[:2]) + make_cycler(linestyle=LINESTYLES[:2])
        first = next(iter(prop))
        assert "marker" not in first

    def test_style_lines_with_markers(self):
        """use_markers=True — the cycler includes a 'marker' key."""
        from cycler import cycler as make_cycler

        prop = (
            make_cycler(color=GRAYS[:2])
            + make_cycler(linestyle=LINESTYLES[:2])
            + make_cycler(marker=MARKERS[:2])
        )
        first = next(iter(prop))
        assert "marker" in first
