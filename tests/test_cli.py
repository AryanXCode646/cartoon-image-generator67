"""
Unit tests for CLI parser and subcommands.
"""

from cartoonify.cli import build_parser


def test_cli_parser_commands():
    parser = build_parser()

    # Test process subcommand
    args_proc = parser.parse_args(["process", "input.jpg", "-o", "out.jpg", "--style", "ghibli_pro"])
    assert args_proc.command == "process"
    assert args_proc.input == "input.jpg"
    assert args_proc.output == "out.jpg"
    assert args_proc.style == "ghibli_pro"

    # Test batch subcommand
    args_batch = parser.parse_args(["batch", "input_folder", "-o", "out_folder", "--style", "comic_pop"])
    assert args_batch.command == "batch"
    assert args_batch.input_dir == "input_folder"
    assert args_batch.output_dir == "out_folder"
    assert args_batch.style == "comic_pop"

    # Test web subcommand
    args_web = parser.parse_args(["web", "--port", "8080"])
    assert args_web.command == "web"
    assert args_web.port == 8080

    # Test list-styles subcommand
    args_list = parser.parse_args(["list-styles"])
    assert args_list.command == "list-styles"
