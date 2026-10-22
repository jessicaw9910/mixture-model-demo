import json

import pytest
from omegaconf import OmegaConf
from typer.testing import CliRunner

from mixture_demo.cli import app
from mixture_demo.config import ConfigError, DemoConfig

cli = CliRunner()


def test_yaml_roundtrip(tmp_path):
    cfg = DemoConfig().with_overrides({"monte_carlo.reps": 7, "truth.means": [-1.0, 1.0]})
    path = cfg.dump(tmp_path / "c.yaml")
    assert DemoConfig.load(path) == cfg


def test_json_config_loads(tmp_path):
    path = tmp_path / "c.json"
    path.write_text(json.dumps({"monte_carlo": {"reps": 5}, "model": {"tol": "1e-6"}}))
    cfg = DemoConfig.load(path)
    assert cfg.monte_carlo.reps == 5 and cfg.model.tol == 1e-6


def test_dotlist_overrides():
    cfg = DemoConfig().with_overrides(["monte_carlo.sample_sizes=[10,20]", "prior.alpha=1.5"])
    assert cfg.monte_carlo.sample_sizes == [10, 20] and cfg.prior.alpha == 1.5


@pytest.mark.parametrize(
    "overrides",
    [["model.nope=1"], ["truth.weights=[0.5,0.6]"], ["figures=[bogus]"], ["monte_carlo.reps=1.5"], ["reps"]],
)
def test_invalid_config_rejected(overrides):
    with pytest.raises(ConfigError):
        DemoConfig().with_overrides(overrides)


def test_cli_overrides_beat_config_file(tmp_path):
    path = DemoConfig().with_overrides({"monte_carlo.reps": 3, "single_dataset.n": 30}).dump(tmp_path / "c.yaml")
    result = cli.invoke(app, ["show-config", "--config", str(path), "monte_carlo.reps=9", "-f", "monte-carlo"])
    assert result.exit_code == 0, result.output
    shown = OmegaConf.create(result.output)
    assert shown.monte_carlo.reps == 9  # override wins
    assert shown.single_dataset.n == 30  # file value kept
    assert list(shown.figures) == ["monte-carlo"]


def test_cli_rejects_bad_key():
    result = cli.invoke(app, ["show-config", "monte_carlo.nope=1"])
    assert result.exit_code != 0


def test_run_writes_reusable_record(tmp_path):
    args = [
        "-q", "run", f"output.images_dir={tmp_path}", "output.dpi=40",
        "singularity.n_grid=50",
        "single_dataset.max_search=3", "single_dataset.n_starts=3",
        "monte_carlo.sample_sizes=[15,30]", "monte_carlo.reps=2", "monte_carlo.n_starts=2",
    ]
    result = cli.invoke(app, args)
    assert result.exit_code == 0, result.output
    (run_dir,) = tmp_path.iterdir()
    names = {p.name for p in run_dir.iterdir()}
    assert {"config.yaml", "summary.json", "fig1_singularity.png",
            "fig2_single_dataset.png", "fig3_monte_carlo.png"} <= names

    saved = DemoConfig.load(run_dir / "config.yaml")
    assert saved.monte_carlo.sample_sizes == [15, 30]
    summary = json.loads((run_dir / "summary.json").read_text())
    assert set(summary["figures"]) == {"singularity", "single-dataset", "monte-carlo"}

    # The saved config reproduces the run.
    rerun = cli.invoke(app, ["-q", "show-config", "--config", str(run_dir / "config.yaml")])
    assert DemoConfig.from_dict(OmegaConf.to_container(OmegaConf.create(rerun.output))) == saved
