import unittest

import torch

from adwersbad import Adwersbad

n_workers = 4
limit = 4000


class TestMultiProcessedDataLoader(unittest.TestCase):

    def setUp(self) -> None:
        cols = {"weather": ["weather_uid"]}
        self.ds = Adwersbad(data=cols, limit=limit, shuffle=True)
        self.dataloader = torch.utils.data.DataLoader(self.ds, num_workers=n_workers)

    def test_default(self):
        uids = []
        for uid, *_ in self.dataloader:
            uids.append(uid.item())
        self.assertEqual(len(uids), limit)
        self.assertEqual(len(uids), len(set(uids)))

    def tearDown(self) -> None:
        self.ds.close()


class TestMultiProcessedLimit(unittest.TestCase):

    def setUp(self) -> None:
        cols = {"weather": ["weather_uid"]}
        self.ds = Adwersbad(data=cols, limit=limit, shuffle=True)
        self.dataloader = torch.utils.data.DataLoader(self.ds, num_workers=n_workers)

    def test_default(self):
        uids = []
        for uid, *_ in self.dataloader:
            uids.append(uid.item())
        self.assertEqual(len(uids), limit)
        self.assertEqual(len(uids), len(set(uids)))
        self.assertEqual(limit, self.ds.count)

    def tearDown(self) -> None:
        self.ds.close()


if __name__ == "__main__":
    unittest.main()
