"""
Image Processor for AI Memory System
Handles screenshot/image storage with optional OCR text extraction
"""
import base64
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from PIL import Image

# Optional: pytesseract for OCR
try:
    import pytesseract
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False
    print("[ImageProcessor] pytesseract not available - OCR disabled")


class ImageProcessor:
    """
    Process and store images/screenshots with optional OCR.
    """

    def __init__(self, base_path: str = None):
        if base_path is None:
            try:
                from .config import get_base_path
            except ImportError:
                from config import get_base_path
            base_path = get_base_path()
        self.base_path = Path(base_path)
        self.images_path = self.base_path / "images"
        self.index_file = self.images_path / "index.json"

        # Create directory
        self.images_path.mkdir(parents=True, exist_ok=True)

        # Load or create index
        self.index = self._load_index()

    def _load_index(self) -> Dict:
        """Load image index"""
        if self.index_file.exists():
            with open(self.index_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {'images': []}

    def _save_index(self):
        """Save image index"""
        with open(self.index_file, 'w', encoding='utf-8') as f:
            json.dump(self.index, f, indent=2, ensure_ascii=False)

    def save_screenshot(self, image_data: str, conversation_id: str,
                       description: str = "") -> Dict:
        """
        Save a screenshot.

        Args:
            image_data: Base64 encoded image or file path
            conversation_id: Associated conversation ID
            description: Optional description

        Returns:
            Image metadata
        """
        # Generate unique ID
        image_id = f"img_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # Determine if base64 or file path
        if image_data.startswith('data:image') or len(image_data) > 1000:
            # Base64 encoded
            image_path = self._save_base64_image(image_data, image_id)
        else:
            # File path
            image_path = self._copy_image_file(image_data, image_id)

        if not image_path:
            return {'error': 'Failed to save image'}

        # Extract text if OCR available
        ocr_text = ""
        if OCR_AVAILABLE:
            try:
                ocr_text = pytesseract.image_to_string(Image.open(image_path))
            except Exception as e:
                print(f"[ImageProcessor] OCR failed: {e}")

        # Create metadata
        metadata = {
            'image_id': image_id,
            'path': str(image_path),
            'conversation_id': conversation_id,
            'description': description,
            'ocr_text': ocr_text,
            'timestamp': datetime.now().isoformat()
        }

        # Add to index
        self.index['images'].append(metadata)
        self._save_index()

        return metadata

    def _save_base64_image(self, base64_data: str, image_id: str) -> Optional[Path]:
        """Save base64 encoded image"""
        try:
            # Remove data URL prefix if present
            if 'base64,' in base64_data:
                base64_data = base64_data.split('base64,')[1]

            # Decode
            image_bytes = base64.b64decode(base64_data)

            # Save as PNG
            image_path = self.images_path / f"{image_id}.png"
            with open(image_path, 'wb') as f:
                f.write(image_bytes)

            return image_path

        except Exception as e:
            print(f"[ImageProcessor] Error saving base64 image: {e}")
            return None

    def _copy_image_file(self, source_path: str, image_id: str) -> Optional[Path]:
        """Copy image file to images directory"""
        try:
            import shutil
            source = Path(source_path)

            if not source.exists():
                return None

            # Preserve extension
            ext = source.suffix or '.png'
            dest = self.images_path / f"{image_id}{ext}"

            shutil.copy2(source, dest)
            return dest

        except Exception as e:
            print(f"[ImageProcessor] Error copying image: {e}")
            return None

    def search_images(self, query: str, limit: int = 10) -> List[Dict]:
        """
        Search images by OCR text or description.

        Args:
            query: Search query
            limit: Max results

        Returns:
            List of matching images
        """
        query_lower = query.lower()
        results = []

        for img in self.index['images']:
            # Search in OCR text and description
            if (query_lower in img.get('ocr_text', '').lower() or
                query_lower in img.get('description', '').lower()):

                results.append({
                    'image_id': img['image_id'],
                    'path': img['path'],
                    'conversation_id': img['conversation_id'],
                    'description': img['description'][:200],
                    'ocr_preview': img.get('ocr_text', '')[:200],
                    'timestamp': img['timestamp']
                })

                if len(results) >= limit:
                    break

        return results

    def get_images_for_conversation(self, conversation_id: str) -> List[Dict]:
        """Get all images for a conversation"""
        return [
            img for img in self.index['images']
            if img.get('conversation_id') == conversation_id
        ]
