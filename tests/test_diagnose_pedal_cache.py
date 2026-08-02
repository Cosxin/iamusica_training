from diagnose_pedal_cache import validated_cache_geometry


def test_cache_geometry_derives_and_checks_stride():
    identity = {
        "checkpoint": "/runs/model.torch",
        "split": "validation",
        "limit": 32,
        "chunk_secs": 8.0,
        "overlap_secs": 4.0,
    }
    assert validated_cache_geometry(identity, "validation", 32) == 4.0
    assert validated_cache_geometry(identity, "validation", 32, 4.0) == 4.0


def test_cache_geometry_rejects_mismatched_selection_or_stride():
    identity = {
        "checkpoint": "/runs/model.torch",
        "split": "validation",
        "limit": 32,
        "chunk_secs": 5.0,
        "overlap_secs": 2.0,
    }
    for split, limit, stride in (("test", 32, None),
                                 ("validation", None, None),
                                 ("validation", 32, 4.0)):
        try:
            validated_cache_geometry(identity, split, limit, stride)
        except ValueError:
            pass
        else:
            raise AssertionError("mismatched cache identity was accepted")
