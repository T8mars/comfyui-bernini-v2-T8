from bernini_v2.media import fit_media_size, ordered_renderer_sources


def test_fit_media_size_caps_long_edge_and_preserves_aspect():
    assert fit_media_size(1080, 1920, max_size=848) == (480, 848)


def test_fit_media_size_preserves_portrait_orientation():
    assert fit_media_size(1920, 1080, max_size=848) == (848, 480)


def test_fit_media_size_upscales_short_edge_to_240():
    assert fit_media_size(100, 200, max_size=848) == (240, 480)


def test_fit_media_size_caps_extreme_aspect_after_short_edge_rule():
    height, width = fit_media_size(100, 1000, max_size=848)
    assert max(height, width) <= 848
    assert (height, width) == (80, 848)


def test_renderer_sources_follow_official_image_then_video_order():
    assert ordered_renderer_sources(
        image_sources=["image-0", "image-1"],
        video_sources=["video-0"],
    ) == ["image-0", "image-1", "video-0"]
