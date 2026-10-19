import gzip
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from experiments.omr.bounded_writer import BoundedWriter


class BoundedWriterTests(unittest.TestCase):
    def test_immutable_chunks_byte_bound_and_durable_barrier(self):
        with tempfile.TemporaryDirectory() as tmp:
            writer = BoundedWriter(tmp, 'collection', max_bytes=64, flush_bytes=40, flush_seconds=.05)
            for i in range(20):
                writer.enqueue((json.dumps({'i': i}) + '\n').encode())
                self.assertLessEqual(writer.buffered_bytes, 64)
            writer.drain()
            self.assertEqual(writer.enqueued, writer.durable)
            snapshots = {r['path']: Path(r['path']).read_bytes() for r in writer.chunks}
            writer.enqueue(b'{"i":20}\n')
            writer.close()
            self.assertEqual(writer.enqueued, 21)
            for path, expected in snapshots.items():
                self.assertEqual(Path(path).read_bytes(), expected)
            rows = [json.loads(line) for r in writer.chunks
                    for line in gzip.decompress(Path(r['path']).read_bytes()).decode().splitlines()]
            self.assertEqual([r['i'] for r in rows], list(range(21)))
            self.assertFalse(list(Path(tmp).glob('*.tmp-*')))

    def test_writer_failure_surfaces_and_records_are_never_silently_dropped(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(BoundedWriter, '_publish', side_effect=OSError('injected disk full')):
                writer = BoundedWriter(tmp, 'failure', max_bytes=16, flush_bytes=1)
                writer.enqueue(b'abc')
                with self.assertRaisesRegex(RuntimeError, 'disk full'):
                    writer.drain()
                with self.assertRaisesRegex(RuntimeError, 'disk full'):
                    writer.close()
            self.assertTrue((Path(tmp) / 'failure-writer-error.json').exists())
            writer = BoundedWriter(tmp, 'shape', max_bytes=4)
            with self.assertRaises(ValueError):
                writer.enqueue(b'12345')
            with self.assertRaises(TypeError):
                writer.enqueue(bytearray(b'a'))
            writer.close()


if __name__ == '__main__':
    unittest.main()
