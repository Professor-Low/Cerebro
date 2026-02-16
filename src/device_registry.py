"""
Device Registry - Multi-Device Awareness for AI Memory
Tracks which device each conversation comes from and enables device-specific filtering.
"""

import json
import platform
import socket
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List


class DeviceRegistry:
    """
    Manages device identification and tagging for multi-device AI Memory.

    Supports:
    - Auto-detection of current device
    - Device registration and profiles
    - Conversation tagging by device
    - Device-aware searching
    """

    def __init__(self, base_path: str = None):
        if base_path is None:

            from config import AI_MEMORY_BASE

            base_path = str(AI_MEMORY_BASE)

        self.base_path = Path(base_path)
        self.devices_path = self.base_path / "devices"
        self.devices_path.mkdir(parents=True, exist_ok=True)

        # Device registry file
        self.registry_file = self.devices_path / "device_registry.json"

        # Load or create registry
        self.registry = self._load_registry()

        # Detect current device on init
        self.current_device = self._detect_current_device()

    def _load_registry(self) -> Dict[str, Any]:
        """Load device registry from file."""
        if self.registry_file.exists():
            try:
                with open(self.registry_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except:
                pass

        # Default registry structure
        return {
            "devices": {},
            "created_at": datetime.now().isoformat(),
            "last_updated": datetime.now().isoformat()
        }

    def _save_registry(self):
        """Save device registry to file."""
        self.registry["last_updated"] = datetime.now().isoformat()
        with open(self.registry_file, 'w', encoding='utf-8') as f:
            json.dump(self.registry, f, indent=2, ensure_ascii=False)

    def _detect_current_device(self) -> Dict[str, Any]:
        """Auto-detect current device information."""
        hostname = socket.gethostname()

        # Detect OS and architecture
        system = platform.system()
        architecture = platform.machine()

        # Determine device type based on hostname and characteristics
        device_type = "unknown"
        device_name = hostname

        # Check for known device patterns
        if "spark" in hostname.lower():
            device_type = "dgx_spark"
            device_name = "DGX Spark"
        elif "gx10" in hostname.lower():
            device_type = "asus_gx10"
            device_name = "ASUS GX10"
        elif system == "Windows":
            device_type = "windows_pc"
            device_name = "Windows PC"
        elif "darwin" in system.lower() or "mac" in hostname.lower():
            device_type = "mac"
            device_name = "Mac"
        elif architecture == "aarch64" and "nvidia" in platform.release().lower():
            device_type = "dgx_spark"
            device_name = "DGX Spark"
        elif system == "Linux":
            device_type = "linux_pc"
            device_name = "Linux PC"

        device_info = {
            "hostname": hostname,
            "device_type": device_type,
            "device_name": device_name,
            "os": system,
            "os_version": platform.version(),
            "architecture": architecture,
            "platform_release": platform.release(),
            "detected_at": datetime.now().isoformat()
        }

        # Auto-register if not in registry
        if hostname not in self.registry.get("devices", {}):
            self.register_device(device_info)

        return device_info

    def register_device(self, device_info: Dict[str, Any] = None,
                       friendly_name: str = None,
                       description: str = None) -> Dict[str, Any]:
        """
        Register a device in the registry.

        Args:
            device_info: Device info dict (auto-detected if None)
            friendly_name: Human-friendly name (e.g., "DGX Spark", "Gaming PC")
            description: Description of the device

        Returns:
            Registered device info
        """
        if device_info is None:
            device_info = self._detect_current_device()

        hostname = device_info.get("hostname", socket.gethostname())

        # Merge with any existing registration
        existing = self.registry.get("devices", {}).get(hostname, {})

        device_record = {
            **existing,
            **device_info,
            "friendly_name": friendly_name or existing.get("friendly_name") or device_info.get("device_name", hostname),
            "description": description or existing.get("description", ""),
            "registered_at": existing.get("registered_at", datetime.now().isoformat()),
            "last_seen": datetime.now().isoformat(),
            "conversation_count": existing.get("conversation_count", 0)
        }

        if "devices" not in self.registry:
            self.registry["devices"] = {}

        self.registry["devices"][hostname] = device_record
        self._save_registry()

        return device_record

    def get_current_device(self) -> Dict[str, Any]:
        """Get current device info with registry data."""
        hostname = self.current_device.get("hostname")

        # Merge detected info with registered info
        registered = self.registry.get("devices", {}).get(hostname, {})

        return {
            **self.current_device,
            **registered,
            "is_current": True
        }

    def get_device_tag(self) -> str:
        """
        Get a simple device tag for the current device.
        Used for tagging conversations.

        Returns:
            Device tag like "dgx_spark" or "windows_pc"
        """
        device = self.get_current_device()
        return device.get("device_type", "unknown")

    def get_device_name(self) -> str:
        """Get friendly name for current device."""
        device = self.get_current_device()
        return device.get("friendly_name", device.get("device_name", "Unknown Device"))

    def get_all_devices(self) -> List[Dict[str, Any]]:
        """Get all registered devices."""
        devices = []
        current_hostname = self.current_device.get("hostname")

        for hostname, info in self.registry.get("devices", {}).items():
            devices.append({
                **info,
                "hostname": hostname,
                "is_current": hostname == current_hostname
            })

        return devices

    def increment_conversation_count(self, hostname: str = None):
        """Increment conversation count for a device."""
        if hostname is None:
            hostname = self.current_device.get("hostname")

        if hostname in self.registry.get("devices", {}):
            self.registry["devices"][hostname]["conversation_count"] = \
                self.registry["devices"][hostname].get("conversation_count", 0) + 1
            self.registry["devices"][hostname]["last_seen"] = datetime.now().isoformat()
            self._save_registry()

    def get_device_metadata_for_conversation(self) -> Dict[str, Any]:
        """
        Get device metadata to embed in a conversation.
        Returns minimal but sufficient info for device identification.
        """
        device = self.get_current_device()

        return {
            "device_tag": device.get("device_type", "unknown"),
            "device_name": device.get("friendly_name", device.get("device_name", "Unknown")),
            "hostname": device.get("hostname"),
            "os": device.get("os"),
            "architecture": device.get("architecture")
        }

    def should_tag_as_device_specific(self, content_type: str,
                                      keywords: List[str] = None) -> bool:
        """
        Determine if content should be tagged as device-specific.

        General information (facts, learnings) - no device tag needed
        Device-specific (projects, configs, paths) - needs device tag

        Args:
            content_type: Type of content (project, config, path, general, learning)
            keywords: Keywords in the content

        Returns:
            True if should be device-tagged
        """
        # Always tag these types
        device_specific_types = [
            'project', 'configuration', 'config', 'path', 'file',
            'installation', 'setup', 'environment', 'local'
        ]

        if content_type.lower() in device_specific_types:
            return True

        # Check keywords for device-specific content
        device_specific_keywords = [
            'installed', 'path', 'directory', 'folder', 'drive',
            'local', 'config', 'setup', 'environment', 'project',
            'repo', 'repository', 'working on', 'building'
        ]

        if keywords:
            for kw in keywords:
                if any(dsk in kw.lower() for dsk in device_specific_keywords):
                    return True

        return False

    def filter_by_device(self, items: List[Dict],
                        device_tag: str = None,
                        include_untagged: bool = True) -> List[Dict]:
        """
        Filter a list of items by device tag.

        Args:
            items: List of items with potential device_tag field
            device_tag: Device to filter for (current device if None)
            include_untagged: Include items without device tag

        Returns:
            Filtered list
        """
        if device_tag is None:
            device_tag = self.get_device_tag()

        filtered = []
        for item in items:
            item_tag = item.get("device_tag") or item.get("metadata", {}).get("device_tag")

            if item_tag == device_tag:
                filtered.append(item)
            elif item_tag is None and include_untagged:
                filtered.append(item)

        return filtered


# Singleton instance for easy access
_device_registry = None

def get_device_registry(base_path: str = None) -> DeviceRegistry:
    """Get or create the device registry singleton."""
    global _device_registry
    if _device_registry is None:
        if base_path:
            _device_registry = DeviceRegistry(base_path)
        else:
            _device_registry = DeviceRegistry()
    return _device_registry

def get_current_device_tag() -> str:
    """Quick function to get current device tag."""
    return get_device_registry().get_device_tag()

def get_current_device_name() -> str:
    """Quick function to get current device friendly name."""
    return get_device_registry().get_device_name()

def get_device_metadata() -> Dict[str, Any]:
    """Quick function to get device metadata for conversations."""
    return get_device_registry().get_device_metadata_for_conversation()


if __name__ == "__main__":
    # Test device detection
    registry = DeviceRegistry()

    print("=== Current Device ===")
    device = registry.get_current_device()
    print(json.dumps(device, indent=2))

    print("\n=== Device Tag ===")
    print(f"Tag: {registry.get_device_tag()}")
    print(f"Name: {registry.get_device_name()}")

    print("\n=== All Registered Devices ===")
    for d in registry.get_all_devices():
        print(f"  - {d.get('friendly_name')} ({d.get('hostname')}) - {d.get('device_type')}")

    print("\n=== Conversation Metadata ===")
    print(json.dumps(registry.get_device_metadata_for_conversation(), indent=2))
