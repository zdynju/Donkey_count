from src.counter import FrameCount, summarize_counts


def test_summarize_empty_counts() -> None:
    summary = summarize_counts([])

    assert summary["frame_count"] == 0
    assert summary["avg_count"] == 0
    assert summary["max_count"] == 0
    assert summary["min_count"] == 0
    assert summary["count_by_time"] == []


def test_summarize_frame_counts() -> None:
    frame_counts = [
        FrameCount(frame_index=1, timestamp_sec=0.5, count=2, detections=[]),
        FrameCount(frame_index=2, timestamp_sec=1.0, count=4, detections=[]),
    ]

    summary = summarize_counts(frame_counts)

    assert summary["frame_count"] == 2
    assert summary["avg_count"] == 3
    assert summary["max_count"] == 4
    assert summary["min_count"] == 2
    assert summary["count_by_time"][0]["count"] == 2

