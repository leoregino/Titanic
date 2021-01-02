from click.testing import CliRunner

from titanic_ML.scripts.predict import main


def test_predict():
    runner = CliRunner()
    result = runner.invoke(main, [])
    assert result.exit_code == 0
