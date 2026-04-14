import json
import tempfile
import unittest
from pathlib import Path

from patent_parser.biblio_cache import BiblioMetadataProvider


class BiblioCacheTests(unittest.TestCase):
    def test_lookup_with_normalized_publication_key(self):
        payload = {
            "WO2025123456A1": {
                "metadata": {
                    "publication_number": "WO2025123456A1",
                    "title": "OLED Device",
                }
            }
        }
        with tempfile.TemporaryDirectory() as tmp:
            cache_path = Path(tmp) / "biblio.json"
            cache_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
            provider = BiblioMetadataProvider(cache_path)
            meta = provider.lookup("wo-2025/123456-a1")
            self.assertIsNotNone(meta)
            self.assertEqual(meta.publication_number, "WO2025123456A1")
            self.assertEqual(meta.title, "OLED Device")


if __name__ == "__main__":
    unittest.main()
