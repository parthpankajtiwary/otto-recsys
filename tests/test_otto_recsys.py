#!/usr/bin/env python

"""Tests for `otto_recsys` package."""


import unittest
from click.testing import CliRunner

from otto_recsys import otto_recsys
from otto_recsys import cli


class TestOtto_recsys(unittest.TestCase):
    """Tests for `otto_recsys` package."""

    def setUp(self):
        """Set up test fixtures, if any."""

    def tearDown(self):
        """Tear down test fixtures, if any."""

    def test_000_something(self):
        """Test something."""

    def test_command_line_interface(self):
        """Test the CLI."""
        runner = CliRunner()
        result = runner.invoke(cli.main)
        assert result.exit_code == 0
        assert 'otto_recsys.cli.main' in result.output
        help_result = runner.invoke(cli.main, ['--help'])
        assert help_result.exit_code == 0
        assert '--help  Show this message and exit.' in help_result.output
