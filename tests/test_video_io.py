from src.video_io import parse_time


def test_parse_time_seconds() -> None:
    assert parse_time("12.5") == 12.5


def test_parse_time_minutes_seconds() -> None:
    assert parse_time("01:30") == 90


def test_parse_time_hours_minutes_seconds() -> None:
    assert parse_time("01:02:03") == 3723

