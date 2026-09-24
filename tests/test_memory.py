from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from memory import SecretMemoryError, SQLiteMemoryStore


class MemoryStoreTests(unittest.TestCase):
    def test_crud_and_rule_retrieval(self) -> None:
        with TemporaryDirectory() as temporary:
            store = SQLiteMemoryStore(Path(temporary) / "memory.sqlite3")
            favorite = store.save("preference", "A Ana le gusta el té verde", "user", 0.9)
            store.save("fact", "El gato se llama Milo", "user", 0.8)
            matches = store.search("¿Qué té le gusta a Ana?", 1)
            self.assertEqual([match.record.id for match in matches], [favorite.id])
            self.assertIn("té", matches[0].reason)
            updated = store.update(favorite.id, content="A Ana le gusta el café", confidence=1.0)
            self.assertEqual(updated.content, "A Ana le gusta el café")
            self.assertTrue(store.delete(favorite.id))
            self.assertIsNone(store.get(favorite.id))

    def test_secrets_are_rejected_by_default(self) -> None:
        with TemporaryDirectory() as temporary:
            store = SQLiteMemoryStore(Path(temporary) / "memory.sqlite3")
            with self.assertRaises(SecretMemoryError):
                store.save("event", "password=hunter2", "user", 0.5)
