from cardiep import run_reference_validation


def test_reference_validation_passes() -> None:
    report = run_reference_validation()
    assert report["passed"] is True
    assert all(report["checks"].values())
