"""
Command Line Interface (CLI) for ΨC-AI SDK

This module provides command-line tools for interacting with the ΨC-AI SDK,
including schema management, memory operations, and monitoring tools.
"""

from psi_c_ai_sdk.cli.main import main
from psi_c_ai_sdk.cli.schema_diff import (
    create_parser as create_schema_diff_parser,
    handle_diff_command,
    handle_list_command as handle_schema_list_command,
    handle_snapshot_command,
    handle_drift_command
)
from psi_c_ai_sdk.cli.memory_commands import (
    create_memory_command_parser,
    memory_cli_main,
    handle_list_command as handle_memory_list_command,
    handle_add_command,
    handle_get_command,
    handle_delete_command,
    handle_archive_command,
    handle_restore_command,
    handle_pin_command,
    handle_unpin_command,
    handle_export_command,
    handle_import_command,
    handle_decay_command,
    handle_cull_command
)
from psi_c_ai_sdk.cli.psic_commands import (
    create_psic_command_parser,
    psic_cli_main,
    handle_status_command,
    handle_log_command,
    handle_health_command,
    handle_collapse_command,
    handle_configure_command
)

__all__ = [
    'main',
    'create_schema_diff_parser',
    'handle_diff_command',
    'handle_schema_list_command',
    'handle_snapshot_command',
    'handle_drift_command',
    'create_memory_command_parser',
    'memory_cli_main',
    'handle_memory_list_command',
    'handle_add_command',
    'handle_get_command',
    'handle_delete_command',
    'handle_archive_command',
    'handle_restore_command',
    'handle_pin_command',
    'handle_unpin_command',
    'handle_export_command',
    'handle_import_command',
    'handle_decay_command',
    'handle_cull_command',
    'create_psic_command_parser',
    'psic_cli_main',
    'handle_status_command',
    'handle_log_command',
    'handle_health_command',
    'handle_collapse_command',
    'handle_configure_command'
] 