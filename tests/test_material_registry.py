import unittest

from patent_backend.postgres_store import PostgresPatentStore


class MaterialRegistryTests(unittest.TestCase):
    def test_prepare_registry_rows_aligns_existing_alias(self):
        entity_rows = [
            {
                "entity_id": "e1",
                "entity_type": "material",
                "value_text": "CBP",
                "normalized": "cbp",
                "canonical_id": "mat:LOCAL_CBP",
            },
            {
                "entity_id": "e2",
                "entity_type": "material",
                "value_text": "NPB",
                "normalized": "npb",
                "canonical_id": "mat:NPB",
            },
        ]
        existing = {"CBP": "mat:GLOBAL_CBP"}
        canonical_rows, alias_rows, usage_rows = PostgresPatentStore._prepare_material_registry_rows(
            doc_id="doc1",
            entity_rows=entity_rows,
            existing_alias_map=existing,
        )

        # e1 canonical 应被历史 alias 映射覆盖
        self.assertEqual(entity_rows[0]["canonical_id"], "mat:GLOBAL_CBP")
        canon_ids = {row["canonical_id"] for row in canonical_rows}
        self.assertIn("mat:GLOBAL_CBP", canon_ids)
        self.assertIn("mat:NPB", canon_ids)
        usage = {(row["doc_id"], row["canonical_id"]): row["mention_count"] for row in usage_rows}
        self.assertEqual(usage[("doc1", "mat:GLOBAL_CBP")], 1)
        self.assertEqual(usage[("doc1", "mat:NPB")], 1)
        alias_map = {row["alias_key"]: row["canonical_id"] for row in alias_rows}
        self.assertEqual(alias_map["CBP"], "mat:GLOBAL_CBP")

    def test_prepare_registry_rows_generates_fallback_canonical(self):
        entity_rows = [
            {
                "entity_id": "e3",
                "entity_type": "material",
                "value_text": "Unknown-Mat",
                "normalized": None,
                "canonical_id": None,
            }
        ]
        canonical_rows, alias_rows, usage_rows = PostgresPatentStore._prepare_material_registry_rows(
            doc_id="doc2",
            entity_rows=entity_rows,
            existing_alias_map={},
        )
        canonical_id = entity_rows[0]["canonical_id"]
        self.assertTrue(isinstance(canonical_id, str) and canonical_id.startswith("mat:"))
        self.assertEqual(canonical_rows[0]["canonical_id"], canonical_id)
        self.assertEqual(usage_rows[0]["canonical_id"], canonical_id)
        self.assertTrue(alias_rows)

    def test_alias_key_normalization(self):
        self.assertEqual(PostgresPatentStore._normalize_material_alias_key("Ir(ppy)3"), "IRPPY3")
        self.assertEqual(PostgresPatentStore._normalize_material_alias_key(" 4,4'-Bis "), "44BIS")


if __name__ == "__main__":
    unittest.main()
