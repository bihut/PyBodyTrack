import pytest
from pyBodyTrack.tracking import track_motion

def test_track_motion():
    result = track_motion("sample.mp4", "optical_flow")
    assert "total_movement" in result
    assert result["frames_analyzed"] >= 0



#ejecutar las pruebas --> pytest tests/
