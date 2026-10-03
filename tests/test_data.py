import tempfile
import unittest
from pathlib import Path
import numpy as np
from PIL import Image
from data_features import cats_records,cap_indices


class DataTests(unittest.TestCase):
    def test_corruption_and_duplicate_exclusion(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);rng=np.random.default_rng(123)
            for folder in ['Cat','Dog']:
                (root/folder).mkdir()
                for i in range(12):
                    Image.fromarray(rng.integers(0,255,(8,8,3),dtype=np.uint8)).save(root/folder/f'{i}.png')
            (root/'Cat'/'duplicate.png').write_bytes((root/'Cat'/'0.png').read_bytes())
            (root/'Dog'/'corrupt.jpg').write_bytes(b'not an image')
            rows,excluded=cats_records(root,download=False)
            self.assertEqual(len(rows),24);self.assertEqual(len(excluded),2)
            self.assertEqual(len(set(r['id'] for r in rows)),24)

    def test_all_conflicting_copies_excluded_before_splitting(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);rng=np.random.default_rng(321)
            for folder in ['Cat','Dog']:
                (root/folder).mkdir()
                for i in range(12):
                    Image.fromarray(rng.integers(0,255,(8,8,3),dtype=np.uint8)).save(root/folder/f'{i}.png')
            for value,locations in [(0,['Cat/a.png','Cat/b.png','Dog/c.png']),
                                    (255,['Cat/d.png','Dog/e.png','Dog/f.png'])]:
                for loc in locations:
                    Image.fromarray(np.full((8,8,3),value,dtype=np.uint8)).save(root/loc)
            before={p:p.read_bytes() for p in root.rglob('*.png')}
            rows,excluded=cats_records(root,download=False)
            self.assertEqual(len(rows),24);self.assertEqual(len(excluded),6)
            self.assertTrue(all(r['reason']=='conflicting labels for identical decoded image' for r in excluded))
            self.assertEqual(len({r['id'] for r in excluded}),2)
            self.assertFalse({r['id'] for r in rows}&{r['id'] for r in excluded})
            self.assertEqual([sum(r['label']==label for r in rows) for label in [0,1]],[12,12])
            self.assertEqual((rows,excluded),cats_records(root,download=False))
            self.assertEqual(before,{p:p.read_bytes() for p in root.rglob('*.png')})

    def test_cap_is_reproducible_and_stratified(self):
        y=np.repeat(np.arange(10),100);idx=np.arange(1000)
        a=cap_indices(idx,y,200,17);b=cap_indices(idx,y,200,17)
        np.testing.assert_array_equal(a,b)
        np.testing.assert_array_equal(np.bincount(y[a]),np.full(10,20))
