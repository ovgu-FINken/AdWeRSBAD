import random
import unittest

import torch

from adwersbad import Adwersbad


class TestDatasetConstructionDict(unittest.TestCase):

    def setUp(self) -> None:
        cols = {"weather": ["weather"]}
        self.ds = Adwersbad(data=cols)

    def test_default(self):
        self.assertTrue(self.ds.fields)

    def tearDown(self) -> None:
        self.ds.close()


class TestReadImage(unittest.TestCase):

    def setUp(self) -> None:
        cols = {"camera": ["image"]}
        offset = random.randint(0, 1000)
        self.ds = Adwersbad(data=cols, offset=offset, limit=1)

    def test_read(self):
        for image, *_ in self.ds:
            self.assertIsInstance(image, torch.Tensor)
            self.assertEqual(image.dtype, torch.float32)
            self.assertEqual(image.ndim, 3)
            self.assertEqual(image.shape[0], 3)
            break

    def tearDown(self) -> None:
        self.ds.close()


class TestReadImageForEachDataset(unittest.TestCase):

    def setUp(self) -> None:
        datasets = ["nuimages", "nuscenes", "waymo"]
        cols = {"camera": ["image"]}
        self.datasets = []
        for ds in datasets:
            self.datasets.append(Adwersbad(data=cols, datasets=[ds], limit=1))

    def test_read(self):
        for dataset in self.datasets:
            for image, *_ in dataset:
                self.assertIsInstance(image, torch.Tensor)
                self.assertEqual(image.dtype, torch.float32)
                self.assertEqual(image.ndim, 3)
                self.assertEqual(image.shape[0], 3)
                break

    def tearDown(self) -> None:
        for ds in self.datasets:
            ds.close()


class TestReadImageLabel(unittest.TestCase):

    def setUp(self) -> None:
        cols = {"camera_segmentation": ["camera_segmentation"]}
        self.ds = Adwersbad(data=cols, limit=1)

    def test_read(self):
        for label, *_ in self.ds:
            self.assertIsInstance(label, torch.Tensor)
            self.assertEqual(label.dtype, torch.long)
            self.assertEqual(label.ndim, 2)
            break

    def tearDown(self) -> None:
        self.ds.close()


class TestReadImageLabelForEachDataset(unittest.TestCase):

    def setUp(self) -> None:
        datasets = ["nuimages", "nuscenes", "waymo"]
        cols = {"camera_segmentation": ["camera_segmentation"]}
        self.datasets = []
        for ds in datasets:
            self.datasets.append(Adwersbad(data=cols, datasets=[ds], limit=1))

    def test_read(self):
        for dataset in self.datasets:
            for label, *_ in dataset:
                self.assertIsInstance(label, torch.Tensor)
                self.assertEqual(label.dtype, torch.long)
                self.assertEqual(label.ndim, 2)
                break

    def tearDown(self) -> None:
        for ds in self.datasets:
            ds.close()


class TestReadImageSegm(unittest.TestCase):

    def setUp(self) -> None:
        cols = {"camera": ["image"], "camera_segmentation": ["camera_segmentation"]}
        self.ds = Adwersbad(data=cols, limit=1)

    def test_read(self):
        for image, label in self.ds:
            self.assertEqual(image.ndim, 3)
            self.assertEqual(label.ndim, 2)

            self.assertEqual(image.shape[0], 3)
            self.assertEqual(image.shape[1], label.shape[0])
            self.assertEqual(image.shape[2], label.shape[1])

            break

    def tearDown(self) -> None:
        self.ds.close()


class TestReadImageSegmForEachDataset(unittest.TestCase):

    def setUp(self) -> None:
        datasets = ["nuimages", "nuscenes", "waymo"]
        cols = {"camera": ["image"], "camera_segmentation": ["camera_segmentation"]}
        self.datasets = []
        for ds in datasets:
            self.datasets.append(Adwersbad(data=cols, datasets=[ds], limit=1))
        self.ds = Adwersbad(data=cols)

    def test_read(self):
        for ds in self.datasets:
            for image, label in ds:
                self.assertEqual(image.ndim, 3)
                self.assertEqual(label.ndim, 2)

                self.assertEqual(image.shape[0], 3)
                self.assertEqual(image.shape[1], label.shape[0])
                self.assertEqual(image.shape[2], label.shape[1])
                break

    def tearDown(self) -> None:
        for ds in self.datasets:
            ds.close()


class TestReadLidar(unittest.TestCase):

    def setUp(self) -> None:
        cols = {"lidar": ["points"]}
        self.ds = Adwersbad(data=cols, limit=1)

    def test_read(self):
        for points, *_ in self.ds:
            self.assertIsInstance(points, torch.Tensor)
            self.assertEqual(points.dtype, torch.float32)
            self.assertEqual(points.ndim, 2)
            break

    def tearDown(self) -> None:
        self.ds.close()


class TestReadLidarLabel(unittest.TestCase):

    def setUp(self) -> None:
        cols = {"lidar_segmentation": ["lidar_segmentation"]}
        self.ds = Adwersbad(data=cols, limit=1)

    def test_read(self):
        for label, *_ in self.ds:
            self.assertIsInstance(label, torch.Tensor)
            self.assertEqual(label.dtype, torch.long)
            self.assertEqual(label.ndim, 1)
            break

    def tearDown(self) -> None:
        self.ds.close()


class TestReadLidarLabelForEachDataset(unittest.TestCase):

    def setUp(self) -> None:
        datasets = ["nuimages", "nuscenes", "waymo"]
        cols = {"lidar_segmentation": ["lidar_segmentation"]}
        self.datasets = []
        for ds in datasets:
            self.datasets.append(Adwersbad(data=cols, datasets=[ds], limit=1))

    def test_read(self):
        for ds in self.datasets:
            for label, *_ in ds:
                self.assertIsInstance(
                    label, torch.Tensor, f"Faulty lidar label in {ds}"
                )
                self.assertEqual(label.dtype, torch.long, f"Faulty lidar label in {ds}")
                self.assertEqual(label.ndim, 1, f"Faulty lidar label in {ds}")
                break

    def tearDown(self) -> None:
        for ds in self.datasets:
            ds.close()


class TestReadLidarSegm(unittest.TestCase):

    def setUp(self) -> None:
        cols = {"lidar": ["points"], "lidar_segmentation": ["lidar_segmentation"]}
        self.ds = Adwersbad(data=cols, limit=1)

    def test_read(self):
        for lidar, label in self.ds:
            self.assertEqual(lidar.ndim, 2)
            self.assertEqual(label.ndim, 1)
            self.assertEqual(len(lidar), len(label))
            break

    def tearDown(self) -> None:
        self.ds.close()


# class TestReadProbability(unittest.TestCase):
#
#     def setUp(self) -> None:
#         cols = {"results_camera" : ["result"]}
#         self.ds = Adwersbad(data=cols)
#
#     def test_read(self):
#         for prob, *_ in self.ds:
#             self.assertIsInstance(prob, torch.Tensor)
#             self.assertEqual(prob.dtype, torch.float32)
#             self.assertEqual(prob.ndim, 3)
#             self.assertEqual(prob.shape[0], 28)
#             torch.testing.assert_close(prob.sum(axis=0), torch.ones_like(prob.sum(axis=0)))
#             break
#
#     def tearDown(self) -> None:
#         self.ds.close()

# class TestReadIoU(unittest.TestCase):
#
#     def setUp(self) -> None:
#         cols = {"results_camera" : ["iou"]}
#         self.ds = Adwersbad(data=cols)
#
#     def test_read(self):
#         for iou, *_ in self.ds:
#             self.assertIsInstance(iou, float)
#             self.assertTrue(0 <= iou <= 1)
#             break
#
#     def tearDown(self) -> None:
#         self.ds.close()

# class TestQueryGenAllTables(unittest.TestCase):
#     def setUp(self) -> None:
#         cols = {'lidar' : ['id_lidar'], 'camera': ['id_camera'], 'weather' : ['weather'], 'results_lidar': ['iou'], 'results_camera': ['iou']}
#         self.n_fields = len(cols.items())
#         self.ds = Adwersbad(data=cols)
#
#     def test_read(self):
#         for row in self.ds:
#             self.assertEqual(len(row), self.n_fields)
#             break
#
#     def tearDown(self) -> None:
#         self.ds.close()


if __name__ == "__main__":
    unittest.main()
