import json
import tempfile
import unittest
from pathlib import Path

from patent_parser.postprocess import _collect_table_entities_relations, build_structured_json


class PostprocessTableFixtureTests(unittest.TestCase):
    def _oled_table_rows(self):
        return [
            [
                {"text": "Material", "is_header": True, "rowspan": 2, "colspan": 1},
                {"text": "Performance", "is_header": True, "rowspan": 1, "colspan": 2},
                {"text": "Host", "is_header": True, "rowspan": 2, "colspan": 1},
            ],
            [
                {"text": "EQE (%)", "is_header": True, "rowspan": 1, "colspan": 1},
                {"text": "CE (cd/A)", "is_header": True, "rowspan": 1, "colspan": 1},
            ],
            [
                {"text": "CBP", "is_header": False, "rowspan": 1, "colspan": 1},
                {"text": "21.4", "is_header": False, "rowspan": 1, "colspan": 1},
                {"text": "36.2", "is_header": False, "rowspan": 1, "colspan": 1},
                {"text": "yes", "is_header": False, "rowspan": 1, "colspan": 1},
            ],
            [
                {"text": "mCP", "is_header": False, "rowspan": 1, "colspan": 1},
                {"text": "18.0", "is_header": False, "rowspan": 1, "colspan": 1},
                {"text": "30.5", "is_header": False, "rowspan": 1, "colspan": 1},
                {"text": "no", "is_header": False, "rowspan": 1, "colspan": 1},
            ],
        ]

    def test_multilevel_header_row_col_binding(self):
        rows = self._oled_table_rows()
        entities, relations = _collect_table_entities_relations(rows, table_id="tbl_fixture_01", material_alias_map={})

        row_metric = [r for r in relations if r.get("type") == "row_has_metric"]
        has_value = [r for r in relations if r.get("type") == "has_value"]
        has_role = [r for r in relations if r.get("type") == "has_role"]
        self.assertGreaterEqual(len(row_metric), 4)
        self.assertGreaterEqual(len(has_value), 4)
        self.assertGreaterEqual(len(has_role), 2)

        rules = {str(r.get("rule")) for r in relations}
        self.assertIn("table_col_binding", rules)
        self.assertNotIn("nearest_metric", rules)
        self.assertNotIn("nearest_role", rules)

        mats = [e for e in entities if e.get("type") == "material"]
        self.assertTrue(mats)
        self.assertTrue(all(str(m.get("canonical_id", "")).startswith("mat:") for m in mats))

    def test_build_structured_json_with_oled_table_html(self):
        table_html = """
        <table>
          <tr><th rowspan="2">Material</th><th colspan="2">Performance</th><th rowspan="2">Host</th></tr>
          <tr><th>EQE (%)</th><th>CE (cd/A)</th></tr>
          <tr><td>CBP</td><td>21.4</td><td>36.2</td><td>yes</td></tr>
          <tr><td>mCP</td><td>18.0</td><td>30.5</td><td>no</td></tr>
        </table>
        """

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pdf_path = root / "WO2026000001A1.pdf"
            output_root = root / "output"
            doc_dir = output_root / pdf_path.stem
            doc_dir.mkdir(parents=True, exist_ok=True)

            content_list_path = doc_dir / f"{pdf_path.stem}_content_list.json"
            content_list = [
                {
                    "type": "table",
                    "page_idx": 0,
                    "bbox": [0, 0, 1000, 1000],
                    "table_body": table_html,
                    "table_caption": ["Table 1 OLED device performance"],
                }
            ]
            content_list_path.write_text(json.dumps(content_list, ensure_ascii=False), encoding="utf-8")

            out_path = build_structured_json(pdf_path=pdf_path, output_dir=output_root)
            self.assertIsNotNone(out_path)
            payload = json.loads(Path(out_path).read_text(encoding="utf-8"))

            table_blocks = [b for b in payload.get("blocks", []) if b.get("type") == "table"]
            self.assertTrue(table_blocks)
            rels = table_blocks[0].get("relations", [])
            self.assertTrue(rels)
            rules = {str(r.get("rule")) for r in rels}
            self.assertIn("table_col_binding", rules)
            self.assertNotIn("nearest_metric", rules)

            table_entities = payload.get("tables", [])[0].get("entities", [])
            material_entities = [e for e in table_entities if e.get("type") == "material"]
            self.assertTrue(material_entities)
            self.assertTrue(all(str(e.get("canonical_id", "")).startswith("mat:") for e in material_entities))


if __name__ == "__main__":
    unittest.main()
