"""A var written into a .env but left empty (`PIPELINE_DEVICE=`) is read as
"" rather than unset, so the None default never applies. Downstream that
empty string gets used as a real value - as a torch device it failed deep
inside torch.load, far from the cause - so Settings coerces blanks to None.
"""

from app.config import Settings


def _settings(**env: str) -> Settings:
    # _env_file=None so the developer's own .env can't affect the result.
    return Settings(_env_file=None, **env)


def test_blank_optionals_become_none():
    settings = _settings(
        pipeline_device="",
        court_weights_path="   ",
        s3_endpoint_url="",
        s3_public_endpoint_url="",
        invite_code="",
    )
    assert settings.pipeline_device is None
    assert settings.court_weights_path is None
    assert settings.s3_endpoint_url is None
    assert settings.s3_public_endpoint_url is None
    assert settings.invite_code is None


def test_real_values_are_untouched():
    settings = _settings(pipeline_device="cuda:0", invite_code="letmein")
    assert settings.pipeline_device == "cuda:0"
    assert settings.invite_code == "letmein"
