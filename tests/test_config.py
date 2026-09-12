"""Tests for the configuration layer (review session S4, F-12)."""

from pathlib import Path

import pytest

from localwhisper.config import (
    ConfigError,
    Settings,
    coerce_setting,
    default_config_path,
    load_settings,
    save_setting,
)

NO_ENV: dict[str, str] = {}


@pytest.fixture
def missing_config(tmp_path) -> Path:
    return tmp_path / "does-not-exist.toml"


class TestSettings:
    def test_defaults_match_model_decision(self):
        s = Settings()
        assert s.ollama_model == "gemma4:e2b-mlx"
        assert s.whisper_model == "base.en"
        assert s.language is None
        assert s.keep_alive == "5m"
        assert s.ollama_timeout == 120.0

    def test_english_only_model_rejects_other_language(self):
        with pytest.raises(ConfigError, match="English-only"):
            Settings(whisper_model="base.en", language="it")

    def test_english_only_model_accepts_en(self):
        assert Settings(whisper_model="base.en", language="en").language == "en"

    def test_multilingual_model_accepts_language(self):
        assert Settings(whisper_model="small", language="it").language == "it"

    def test_url_is_validated_and_normalised(self):
        assert Settings(ollama_url="http://x:1/").ollama_url == "http://x:1"
        with pytest.raises(ConfigError, match="ollama_url"):
            Settings(ollama_url="localhost:11434")

    def test_timeout_must_be_positive(self):
        with pytest.raises(ConfigError, match="ollama_timeout"):
            Settings(ollama_timeout=0)

    def test_empty_names_rejected(self):
        with pytest.raises(ConfigError):
            Settings(whisper_model="")
        with pytest.raises(ConfigError):
            Settings(ollama_model="")
        with pytest.raises(ConfigError):
            Settings(hotkey="")


class TestLoadSettings:
    def test_defaults_when_nothing_is_set(self, missing_config):
        assert load_settings([], env=NO_ENV, config_path=missing_config) == Settings()

    def test_config_file(self, tmp_path):
        cfg = tmp_path / "config.toml"
        cfg.write_text('whisper_model = "small"\nlanguage = "it"\nollama_timeout = 30\n')

        s = load_settings([], env=NO_ENV, config_path=cfg)

        assert s.whisper_model == "small"
        assert s.language == "it"
        assert s.ollama_timeout == 30.0

    def test_unknown_key_in_file_is_an_error(self, tmp_path):
        cfg = tmp_path / "config.toml"
        cfg.write_text('whisper_size = "small"\n')

        with pytest.raises(ConfigError, match=r"Unknown setting.*whisper_size"):
            load_settings([], env=NO_ENV, config_path=cfg)

    def test_malformed_file_is_an_error(self, tmp_path):
        cfg = tmp_path / "config.toml"
        cfg.write_text("this is not toml = = =\n")

        with pytest.raises(ConfigError, match="Cannot read config file"):
            load_settings([], env=NO_ENV, config_path=cfg)

    def test_env_overrides_file(self, tmp_path):
        cfg = tmp_path / "config.toml"
        cfg.write_text('ollama_model = "from-file"\n')
        env = {"LOCALWHISPER_OLLAMA_MODEL": "from-env", "LOCALWHISPER_VERBOSE": "true"}

        s = load_settings([], env=env, config_path=cfg)

        assert s.ollama_model == "from-env"
        assert s.verbose is True

    def test_cli_overrides_env(self, missing_config):
        env = {"LOCALWHISPER_OLLAMA_MODEL": "from-env", "LOCALWHISPER_HOTKEY": "<ctrl>+x"}

        s = load_settings(
            ["--ollama-model", "from-cli", "--ollama-timeout", "7.5", "-v"],
            env=env,
            config_path=missing_config,
        )

        assert s.ollama_model == "from-cli"
        assert s.hotkey == "<ctrl>+x"  # env value survives when the flag is absent
        assert s.ollama_timeout == 7.5
        assert s.verbose is True

    def test_empty_language_means_auto(self, missing_config):
        s = load_settings(["--language", ""], env=NO_ENV, config_path=missing_config)
        assert s.language is None

    def test_bad_number_from_env_is_an_error(self, missing_config):
        with pytest.raises(ConfigError, match="expected a number"):
            load_settings(
                [], env={"LOCALWHISPER_OLLAMA_TIMEOUT": "soon"}, config_path=missing_config
            )

    def test_invalid_combination_from_flags_is_an_error(self, missing_config):
        with pytest.raises(ConfigError, match="English-only"):
            load_settings(["--language", "it"], env=NO_ENV, config_path=missing_config)

    def test_config_flag_selects_file(self, tmp_path):
        cfg = tmp_path / "custom.toml"
        cfg.write_text('hotkey = "<alt>+r"\n')

        s = load_settings(["--config", str(cfg)], env=NO_ENV)

        assert s.hotkey == "<alt>+r"

    def test_version_flag_exits(self, capsys):
        with pytest.raises(SystemExit) as exc:
            load_settings(["--version"], env=NO_ENV)
        assert exc.value.code == 0
        assert "localwhisper" in capsys.readouterr().out


class TestDefaultConfigPath:
    def test_xdg_config_home(self):
        path = default_config_path({"XDG_CONFIG_HOME": "/tmp/xdg"})
        assert path == Path("/tmp/xdg/localwhisper/config.toml")

    def test_explicit_env_override(self):
        path = default_config_path({"LOCALWHISPER_CONFIG": "/etc/lw.toml"})
        assert path == Path("/etc/lw.toml")

    def test_home_fallback(self):
        path = default_config_path({})
        assert path == Path.home() / ".config" / "localwhisper" / "config.toml"


class TestSaveSetting:
    def test_creates_file_and_directory(self, tmp_path):
        path = tmp_path / "nested" / "config.toml"
        assert save_setting("ollama_model", "x:1b", path) == path
        assert path.read_text() == 'ollama_model = "x:1b"\n'
        assert load_settings([], env=NO_ENV, config_path=path).ollama_model == "x:1b"

    def test_replaces_in_place_and_keeps_other_lines(self, tmp_path):
        path = tmp_path / "config.toml"
        path.write_text('# my config\nwhisper_model = "small"  # trailing\nhotkey = "<cmd>+g"\n')

        save_setting("whisper_model", "medium", path)

        assert path.read_text() == '# my config\nwhisper_model = "medium"\nhotkey = "<cmd>+g"\n'

    def test_none_removes_the_key(self, tmp_path):
        path = tmp_path / "config.toml"
        path.write_text('language = "it"\nwhisper_model = "small"\n')
        save_setting("language", None, path)
        assert path.read_text() == 'whisper_model = "small"\n'
        assert load_settings([], env=NO_ENV, config_path=path).language is None

    def test_none_on_missing_key_is_a_no_op(self, tmp_path):
        path = tmp_path / "config.toml"
        path.write_text('whisper_model = "small"\n')
        save_setting("language", None, path)
        assert path.read_text() == 'whisper_model = "small"\n'

    def test_bool_float_and_quoted_strings_round_trip(self, tmp_path):
        path = tmp_path / "config.toml"
        save_setting("auto_paste", True, path)
        save_setting("ollama_timeout", 45.5, path)
        save_setting("hotkey", '<cmd>+"q"', path)

        s = load_settings([], env=NO_ENV, config_path=path)

        assert s.auto_paste is True
        assert s.ollama_timeout == 45.5
        assert s.hotkey == '<cmd>+"q"'

    def test_unknown_name_is_an_error(self, tmp_path):
        with pytest.raises(ConfigError, match="Unknown setting"):
            save_setting("nope", "x", tmp_path / "c.toml")

    def test_unwritable_path_is_a_config_error(self, tmp_path):
        blocker = tmp_path / "file"
        blocker.write_text("")
        with pytest.raises(ConfigError, match="Cannot write"):
            save_setting("hotkey", "<cmd>+g", blocker / "config.toml")


class TestAutoPaste:
    def test_default_off(self):
        assert Settings().auto_paste is False

    def test_flag_env_and_file(self, tmp_path, missing_config):
        assert load_settings(["--auto-paste"], env=NO_ENV, config_path=missing_config).auto_paste
        env = {**NO_ENV, "LOCALWHISPER_AUTO_PASTE": "yes"}
        assert load_settings([], env=env, config_path=missing_config).auto_paste
        cfg = tmp_path / "config.toml"
        cfg.write_text("auto_paste = true\n")
        assert load_settings([], env=NO_ENV, config_path=cfg).auto_paste

    def test_coerce_setting(self):
        assert coerce_setting("auto_paste", "off") is False
        assert coerce_setting("language", "") is None
        with pytest.raises(ConfigError, match="Unknown setting"):
            coerce_setting("nope", "x")
