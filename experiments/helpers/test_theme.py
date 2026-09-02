import matplotlib.pyplot as plt
import pytest
from experiments.helpers import theme


def test_label_panels_uses_uppercase_row_major_labels():
    fig, axes = plt.subplots(2, 2)
    artists = theme.label_panels(axes.flat)

    assert [artist.get_text() for artist in artists] == ["A", "B", "C", "D"]
    assert all(artist.get_fontweight() == "bold" for artist in artists)
    assert all(artist.get_fontsize() == theme.SIZE_PANEL for artist in artists)
    assert all(not artist.get_clip_on() for artist in artists)
    plt.close(fig)


def test_label_panels_rejects_mismatched_labels():
    fig, axes = plt.subplots(1, 2)
    with pytest.raises(ValueError, match="same length"):
        theme.label_panels(axes, ["A"])
    plt.close(fig)


def test_label_panels_continues_after_z():
    fig, axes = plt.subplots(3, 9)
    artists = theme.label_panels(axes.flat)

    assert [artist.get_text() for artist in artists[-2:]] == ["Z", "AA"]
    plt.close(fig)
