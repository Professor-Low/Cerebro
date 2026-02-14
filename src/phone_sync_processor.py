#!/usr/bin/env python3
"""
Phone Sync Processor - Process Claude conversations from mobile
Watches data directory phone_inbox for new files and processes them into AI memory

Supports:
- JSON exports from Claude mobile
- Text file conversations
- Screenshot images (with basic text extraction)
"""
import json
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

# Add home directory to path for imports
sys.path.insert(0, str(Path.home()))

from ai_memory_enhanced import EnhancedMemoryService

from config import DATA_DIR

# Configuration
NAS_PATH = DATA_DIR
INBOX_PATH = NAS_PATH / "phone_inbox"
PROCESSED_PATH = INBOX_PATH / "processed"
LOG_FILE = NAS_PATH / "phone_sync.log"

def log(message):
    """Log messages to file and console"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_msg = f"[{timestamp}] {message}"
    print(log_msg)
    try:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(log_msg + "\n")
    except:
        pass

class PhoneConversationProcessor:
    def __init__(self):
        self.memory = EnhancedMemoryService(base_path=str(NAS_PATH))
        log("Phone Sync Processor initialized")

    def process_file(self, filepath: Path) -> bool:
        """Process a single file from the phone inbox"""
        log(f"Processing: {filepath.name}")

        try:
            suffix = filepath.suffix.lower()

            if suffix == '.json':
                return self._process_json(filepath)
            elif suffix == '.txt':
                return self._process_text(filepath)
            elif suffix in ['.png', '.jpg', '.jpeg']:
                return self._process_image(filepath)
            elif suffix == '.md':
                return self._process_markdown(filepath)
            else:
                log(f"  Unknown file type: {suffix}, treating as text")
                return self._process_text(filepath)

        except Exception as e:
            log(f"  ERROR processing {filepath.name}: {e}")
            return False

    def _process_json(self, filepath: Path) -> bool:
        """Process JSON conversation export"""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Handle different JSON formats
        messages = []

        if isinstance(data, list):
            # List of messages
            messages = data
        elif isinstance(data, dict):
            if 'messages' in data:
                messages = data['messages']
            elif 'conversation' in data:
                messages = data['conversation']
            else:
                # Single message or unknown format
                messages = [{"role": "user", "content": json.dumps(data)}]

        if not messages:
            log("  No messages found in JSON")
            return False

        # Normalize message format
        normalized = []
        for msg in messages:
            if isinstance(msg, dict):
                role = msg.get('role', msg.get('sender', 'user'))
                content = msg.get('content', msg.get('text', msg.get('message', '')))
                if content:
                    normalized.append({"role": role, "content": str(content)})
            elif isinstance(msg, str):
                normalized.append({"role": "user", "content": msg})

        return self._save_conversation(normalized, filepath, "json-export")

    def _process_text(self, filepath: Path) -> bool:
        """Process plain text conversation"""
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()

        # Try to parse as conversation (look for patterns like "User:", "Assistant:", etc.)
        messages = self._parse_text_conversation(content)

        if not messages:
            # Treat entire file as single user message
            messages = [{"role": "user", "content": content}]

        return self._save_conversation(messages, filepath, "text-export")

    def _process_markdown(self, filepath: Path) -> bool:
        """Process markdown conversation"""
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()

        messages = self._parse_text_conversation(content)

        if not messages:
            messages = [{"role": "user", "content": content}]

        return self._save_conversation(messages, filepath, "markdown-export")

    def _process_image(self, filepath: Path) -> bool:
        """Process screenshot image - basic handling"""
        # For now, just save metadata about the image
        # TODO: Add OCR with pytesseract if needed

        messages = [{
            "role": "user",
            "content": f"[Screenshot from phone: {filepath.name}]"
        }]

        # Copy image to a screenshots folder
        screenshots_path = NAS_PATH / "phone_screenshots"
        screenshots_path.mkdir(exist_ok=True)

        dest = screenshots_path / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{filepath.name}"
        shutil.copy2(filepath, dest)

        messages[0]["content"] += f"\nSaved to: {dest}"

        return self._save_conversation(messages, filepath, "screenshot")

    def _parse_text_conversation(self, text: str) -> list:
        """Try to parse text into conversation format"""
        messages = []

        # Common patterns for conversation turns

        # Try splitting by common markers
        lines = text.split('\n')
        current_role = None
        current_content = []

        for line in lines:
            line_lower = line.lower().strip()

            # Check for role markers
            if line_lower.startswith(('user:', 'human:', 'me:', 'you:')):
                if current_role and current_content:
                    messages.append({
                        "role": current_role,
                        "content": '\n'.join(current_content).strip()
                    })
                current_role = "user"
                current_content = [line.split(':', 1)[1].strip() if ':' in line else '']
            elif line_lower.startswith(('assistant:', 'claude:', 'ai:', 'bot:')):
                if current_role and current_content:
                    messages.append({
                        "role": current_role,
                        "content": '\n'.join(current_content).strip()
                    })
                current_role = "assistant"
                current_content = [line.split(':', 1)[1].strip() if ':' in line else '']
            elif current_role:
                current_content.append(line)

        # Don't forget the last message
        if current_role and current_content:
            content = '\n'.join(current_content).strip()
            if content:
                messages.append({
                    "role": current_role,
                    "content": content
                })

        return messages

    def _save_conversation(self, messages: list, source_file: Path, source_type: str) -> bool:
        """Save processed conversation to memory"""
        if not messages:
            log("  No messages to save")
            return False

        # Save using the enhanced memory service
        conv_id = self.memory.save_conversation(
            messages=messages,
            metadata={
                "source": "phone",
                "source_type": source_type,
                "source_file": source_file.name,
                "device": "Z Fold6",
                "imported_at": datetime.now().isoformat()
            }
        )

        log(f"  Saved as: {conv_id}")

        # Move processed file
        processed_file = PROCESSED_PATH / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{source_file.name}"
        shutil.move(str(source_file), str(processed_file))
        log(f"  Moved to: processed/{processed_file.name}")

        return True


class InboxWatcher(FileSystemEventHandler):
    """Watch for new files in the phone inbox"""

    def __init__(self, processor: PhoneConversationProcessor):
        self.processor = processor
        self.processing = set()  # Track files being processed

    def on_created(self, event):
        if event.is_directory:
            return

        filepath = Path(event.src_path)

        # Skip if already processing or in processed folder
        if filepath.parent.name == "processed":
            return
        if str(filepath) in self.processing:
            return

        # Wait a moment for file to finish writing
        time.sleep(1)

        self.processing.add(str(filepath))
        try:
            self.processor.process_file(filepath)
        finally:
            self.processing.discard(str(filepath))


def process_existing_files(processor: PhoneConversationProcessor):
    """Process any files already in the inbox"""
    log("Checking for existing files in inbox...")

    for filepath in INBOX_PATH.iterdir():
        if filepath.is_file() and filepath.name != ".gitkeep":
            processor.process_file(filepath)


def main():
    log("=" * 60)
    log("Phone Sync Processor Starting")
    log(f"Watching: {INBOX_PATH}")
    log("=" * 60)

    # Create processor
    processor = PhoneConversationProcessor()

    # Process existing files first
    process_existing_files(processor)

    # Set up file watcher
    event_handler = InboxWatcher(processor)
    observer = Observer()
    observer.schedule(event_handler, str(INBOX_PATH), recursive=False)
    observer.start()

    log("Watcher started. Press Ctrl+C to stop.")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        log("Stopping watcher...")
        observer.stop()

    observer.join()
    log("Phone Sync Processor stopped")


if __name__ == "__main__":
    main()
