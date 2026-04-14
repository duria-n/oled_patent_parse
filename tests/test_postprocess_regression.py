import unittest

from patent_parser.postprocess import (
    _bind_material_roles,
    _collect_table_entities_relations,
    _extract_entities,
    _extract_example_id,
    _parse_claim_line,
)


class PostprocessRegressionTests(unittest.TestCase):
    def test_parse_claim_line_variants(self):
        no1, text1 = _parse_claim_line("1. An OLED device comprising ...")
        self.assertEqual(no1, 1)
        self.assertIn("An OLED device", text1 or "")

        no2, text2 = _parse_claim_line("Claim 1: The device of claim 1 ...")
        self.assertEqual(no2, 1)
        self.assertIn("The device", text2 or "")

        no3, text3 = _parse_claim_line("权利要求1 一种有机电致发光器件")
        self.assertEqual(no3, 1)
        self.assertIn("一种有机电致发光器件", text3 or "")

    def test_extract_example_id(self):
        self.assertEqual(_extract_example_id("实施例 1 制备方法"), "1")
        self.assertEqual(_extract_example_id("Comparative Example A"), "A")
        self.assertIsNone(_extract_example_id("example without number"))

    def test_pattern_aware_material_role_binding(self):
        text = "CBP was used as host, and NPB as dopant in EML."
        entities = _extract_entities(text)
        for i, ent in enumerate(entities, 1):
            ent["entity_id"] = f"e{i:03d}"
        rels = _bind_material_roles(text, entities, base_id="b001")
        rules = {r.get("rule") for r in rels}
        self.assertIn("pattern_material_as_role", rules)
        self.assertIn("pattern_material_in_layer", rules)

    def test_table_col_row_cell_binding(self):
        rows = [
            [
                {"text": "Material", "is_header": True, "rowspan": 1, "colspan": 1},
                {"text": "EQE (%)", "is_header": True, "rowspan": 1, "colspan": 1},
                {"text": "Host", "is_header": True, "rowspan": 1, "colspan": 1},
            ],
            [
                {"text": "CBP", "is_header": False, "rowspan": 1, "colspan": 1},
                {"text": "21.4", "is_header": False, "rowspan": 1, "colspan": 1},
                {"text": "yes", "is_header": False, "rowspan": 1, "colspan": 1},
            ],
        ]
        entities, relations = _collect_table_entities_relations(rows, table_id="tbl001", material_alias_map={})
        rel_types = {r.get("type") for r in relations}
        self.assertIn("row_has_metric", rel_types)
        self.assertIn("has_value", rel_types)
        material_entities = [e for e in entities if e.get("type") == "material"]
        self.assertTrue(material_entities)
        self.assertEqual(material_entities[0].get("canonical_id"), "mat:CBP")


if __name__ == "__main__":
    unittest.main()
