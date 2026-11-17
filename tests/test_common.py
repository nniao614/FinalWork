from src.common import parse_cli


def test_parse_cli_types_values():
    path, ov = parse_cli(["--config", "configs/x.yaml", "data.limit=8", "layoutlm.ckpt=null",
                          "infer.splits=[validation, test]", "layoutlm.bf16=false", "name=abc"])
    assert path == "configs/x.yaml"
    assert ov == {"data.limit": 8, "layoutlm.ckpt": None, "infer.splits": ["validation", "test"],
                  "layoutlm.bf16": False, "name": "abc"}
