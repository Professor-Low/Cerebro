#!/usr/bin/env python3
"""
Centralized logging for AI Memory system.
Logs to both file and optional external service.
"""
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional


class MemoryLogger:
    """Centralized logger for AI Memory operations."""

    def __init__(self, base_path: str = ""):
        if not base_path:
            from config import DATA_DIR
            base_path = str(DATA_DIR / "logs")
        self.log_path = Path(base_path)
        self.log_path.mkdir(parents=True, exist_ok=True)

        # Rotating log files (daily)
        today = datetime.now().strftime('%Y-%m-%d')
        self.log_file = self.log_path / f"memory_{today}.log"
        self.error_log_file = self.log_path / f"memory_errors_{today}.log"

        # Setup logging
        self.logger = logging.getLogger('AIMemory')
        self.logger.setLevel(logging.INFO)

        # Clear existing handlers to avoid duplicates
        if self.logger.hasHandlers():
            self.logger.handlers.clear()

        # File handler for all logs
        file_handler = logging.FileHandler(self.log_file, encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(file_formatter)

        # File handler for errors only
        error_handler = logging.FileHandler(self.error_log_file, encoding='utf-8')
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(file_formatter)

        # Console handler (less verbose)
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.WARNING)
        console_formatter = logging.Formatter('%(levelname)s: %(message)s')
        console_handler.setFormatter(console_formatter)

        self.logger.addHandler(file_handler)
        self.logger.addHandler(error_handler)
        self.logger.addHandler(console_handler)

    def _format_context(self, **context) -> str:
        """Format context dictionary as JSON string."""
        if not context:
            return ""
        try:
            return json.dumps(context, ensure_ascii=False, default=str)
        except Exception:
            return str(context)

    def info(self, message: str, **context):
        """Log info with context."""
        context_str = self._format_context(**context)
        if context_str:
            self.logger.info(f"{message} | {context_str}")
        else:
            self.logger.info(message)

    def error(self, message: str, exception: Optional[Exception] = None, **context):
        """Log error with exception details."""
        context_str = self._format_context(**context)
        log_message = f"{message} | {context_str}" if context_str else message

        if exception:
            self.logger.error(log_message, exc_info=exception)
        else:
            self.logger.error(log_message)

    def warning(self, message: str, **context):
        """Log warning."""
        context_str = self._format_context(**context)
        if context_str:
            self.logger.warning(f"{message} | {context_str}")
        else:
            self.logger.warning(message)

    def debug(self, message: str, **context):
        """Log debug information."""
        context_str = self._format_context(**context)
        if context_str:
            self.logger.debug(f"{message} | {context_str}")
        else:
            self.logger.debug(message)

    def performance(self, operation: str, duration_ms: float, **context):
        """Log performance metrics."""
        context_str = self._format_context(**context)
        log_message = f"PERF: {operation} took {duration_ms:.2f}ms"
        if context_str:
            log_message += f" | {context_str}"
        self.logger.info(log_message)

    def mcp_tool_call(self, tool_name: str, arguments: Dict[str, Any],
                      success: bool, duration_ms: float, error: Optional[str] = None):
        """Log MCP tool call with standardized format."""
        log_data = {
            'tool': tool_name,
            'args': arguments,
            'success': success,
            'duration_ms': duration_ms
        }

        if error:
            log_data['error'] = error
            self.error(f"MCP tool call failed: {tool_name}", **log_data)
        else:
            self.info(f"MCP tool call: {tool_name}", **log_data)

    def context_injection(self, entities_detected: list, context_items: int,
                          confidence: float, injection_success: bool):
        """Log auto-context injection events."""
        self.info(
            "Auto-context injection",
            entities=entities_detected,
            context_items=context_items,
            confidence=confidence,
            success=injection_success
        )

    def conversation_save(self, conversation_id: str, chunks_created: int,
                          embeddings_generated: int, duration_ms: float):
        """Log conversation save operations."""
        self.info(
            "Conversation saved",
            conversation_id=conversation_id,
            chunks=chunks_created,
            embeddings=embeddings_generated,
            duration_ms=duration_ms
        )

    def search_operation(self, query: str, results_count: int,
                        search_type: str, duration_ms: float):
        """Log search operations."""
        self.info(
            f"{search_type} search completed",
            query=query[:100],  # Truncate long queries
            results=results_count,
            duration_ms=duration_ms
        )

    def get_log_stats(self) -> Dict[str, Any]:
        """Get statistics about logs."""
        stats = {
            'log_file': str(self.log_file),
            'error_log_file': str(self.error_log_file),
            'log_file_exists': self.log_file.exists(),
            'error_log_exists': self.error_log_file.exists()
        }

        if self.log_file.exists():
            stats['log_file_size_kb'] = self.log_file.stat().st_size / 1024

        if self.error_log_file.exists():
            stats['error_log_size_kb'] = self.error_log_file.stat().st_size / 1024

        return stats


# Global logger instance
_global_logger: Optional[MemoryLogger] = None


def get_logger() -> MemoryLogger:
    """Get or create global logger instance."""
    global _global_logger
    if _global_logger is None:
        _global_logger = MemoryLogger()
    return _global_logger


# Convenience functions
def log_info(message: str, **context):
    """Convenience function for info logging."""
    get_logger().info(message, **context)


def log_error(message: str, exception: Optional[Exception] = None, **context):
    """Convenience function for error logging."""
    get_logger().error(message, exception=exception, **context)


def log_warning(message: str, **context):
    """Convenience function for warning logging."""
    get_logger().warning(message, **context)


def log_performance(operation: str, duration_ms: float, **context):
    """Convenience function for performance logging."""
    get_logger().performance(operation, duration_ms, **context)


if __name__ == "__main__":
    # Test the logger
    logger = MemoryLogger()

    logger.info("Logger initialized successfully")
    logger.performance("Test operation", 42.5, test_param="value")
    logger.warning("This is a test warning", component="test")

    try:
        raise ValueError("Test exception")
    except Exception as e:
        logger.error("Test error logging", exception=e, context="testing")

    print("\nLog statistics:")
    print(json.dumps(logger.get_log_stats(), indent=2))
    print(f"\nLogs written to: {logger.log_file}")
