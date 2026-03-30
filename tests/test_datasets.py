import pytest
import torch
from unittest.mock import Mock, patch, call

from torchfits.datasets import FITSDataset, IterableFITSDataset, TableChunkDataset


class TestFITSDataset:
    def test_init_and_manifest(self):
        file_paths = ["file1.fits", "file2.fits"]
        dataset = FITSDataset(
            file_paths=file_paths,
            hdu=1,
            device="cuda",
            include_header=True,
            mmap=True,
            cache_capacity=5,
            handle_cache_capacity=10,
            scale_on_device=False,
            raw_scale=True,
        )

        assert dataset.file_paths == file_paths
        assert dataset.hdu == 1
        assert dataset.device == "cuda"
        assert dataset.include_header is True
        assert dataset.mmap is True
        assert dataset.cache_capacity == 5
        assert dataset.handle_cache_capacity == 10
        assert dataset.scale_on_device is False
        assert dataset.raw_scale is True

        assert len(dataset) == 2
        assert dataset.manifest == [
            {"file_path": "file1.fits", "hdu": 1, "sample_id": 0},
            {"file_path": "file2.fits", "hdu": 1, "sample_id": 1},
        ]

    def test_getitem_out_of_range(self):
        dataset = FITSDataset(["file1.fits"])
        with pytest.raises(IndexError, match="Index 1 out of range \\[0, 1\\)"):
            _ = dataset[1]
        with pytest.raises(IndexError, match="Index -1 out of range \\[0, 1\\)"):
            _ = dataset[-1]

    @patch("torchfits.read")
    def test_getitem_without_header(self, mock_read):
        dataset = FITSDataset(["file1.fits"], hdu=0, device="cpu")

        mock_read.return_value = torch.tensor([1.0, 2.0])

        data = dataset[0]

        mock_read.assert_called_once_with(
            "file1.fits",
            hdu=0,
            device="cpu",
            mmap="auto",
            cache_capacity=0,
            handle_cache_capacity=64,
            scale_on_device=True,
            raw_scale=False,
            return_header=False,
        )

        assert torch.equal(data, torch.tensor([1.0, 2.0]))

    @patch("torchfits.read")
    def test_getitem_with_header(self, mock_read):
        dataset = FITSDataset(["file1.fits"], hdu=1, include_header=True)

        mock_data = torch.tensor([1.0, 2.0])
        mock_header = {"KEY": "VALUE"}
        mock_read.return_value = (mock_data, mock_header)

        data, header = dataset[0]

        mock_read.assert_called_once()
        assert mock_read.call_args[1]["return_header"] is True

        assert torch.equal(data, mock_data)
        assert header == mock_header

    @patch("torchfits.read")
    def test_getitem_with_transform(self, mock_read):
        def transform(x):
            return x * 2

        dataset = FITSDataset(["file1.fits"], transform=transform)
        mock_read.return_value = torch.tensor([1.0, 2.0])

        data = dataset[0]

        assert torch.equal(data, torch.tensor([2.0, 4.0]))


class TestIterableFITSDataset:
    def test_init(self):
        dataset = IterableFITSDataset(
            index_url="http://example.com",
            hdu=2,
            device="cuda",
            shard_size=10,
        )

        assert dataset.index_url == "http://example.com"
        assert dataset.hdu == 2
        assert dataset.device == "cuda"
        assert dataset.shard_size == 10

    def test_get_assigned_shards(self):
        dataset = IterableFITSDataset("http://example.com")

        # Test 1 worker
        shards = dataset._get_assigned_shards(worker_id=0, num_workers=1)
        assert shards == list(range(10))

        # Test 2 workers
        shards_0 = dataset._get_assigned_shards(worker_id=0, num_workers=2)
        shards_1 = dataset._get_assigned_shards(worker_id=1, num_workers=2)

        assert shards_0 == list(range(5))
        assert set(shards_0).intersection(set(shards_1)) == set()
        assert len(shards_0) + len(shards_1) == 10

    @patch("torchfits.read")
    def test_process_shard_success(self, mock_read):
        dataset = IterableFITSDataset("http://example.com", shard_size=2)
        mock_read.side_effect = [torch.tensor([1]), torch.tensor([2])]

        samples = list(dataset._process_shard(shard_id=0))

        assert len(samples) == 2
        assert torch.equal(samples[0], torch.tensor([1]))
        assert torch.equal(samples[1], torch.tensor([2]))

        assert mock_read.call_count == 2
        mock_read.assert_any_call("file_0_0.fits", hdu=0, device="cpu")
        mock_read.assert_any_call("file_0_1.fits", hdu=0, device="cpu")

    @patch("torchfits.logging.logger")
    @patch("torchfits.read")
    def test_process_shard_error_handling(self, mock_read, mock_logger):
        dataset = IterableFITSDataset("http://example.com", shard_size=3)

        # First succeeds, second throws IOError, third throws ValueError
        mock_read.side_effect = [
            torch.tensor([1]),
            IOError("Mock IO Error"),
            ValueError("Mock Value Error")
        ]

        samples = list(dataset._process_shard(shard_id=0))

        assert len(samples) == 1
        assert torch.equal(samples[0], torch.tensor([1]))

        assert mock_logger.error.call_count == 2

    @patch("torchfits.logging.logger")
    @patch("torchfits.read")
    def test_process_shard_critical_error(self, mock_read, mock_logger):
        dataset = IterableFITSDataset("http://example.com", shard_size=1)
        mock_read.side_effect = TypeError("Mock Type Error")

        with pytest.raises(TypeError, match="Mock Type Error"):
            list(dataset._process_shard(shard_id=0))

        mock_logger.critical.assert_called_once()

    @patch("torchfits.read")
    def test_process_shard_with_transform(self, mock_read):
        dataset = IterableFITSDataset(
            "http://example.com",
            shard_size=1,
            transform=lambda x: x + 1
        )
        mock_read.return_value = torch.tensor([1])

        samples = list(dataset._process_shard(shard_id=0))
        assert torch.equal(samples[0], torch.tensor([2]))

    @patch("torch.utils.data.get_worker_info")
    @patch.object(IterableFITSDataset, "_get_assigned_shards")
    @patch.object(IterableFITSDataset, "_process_shard")
    def test_iter_single_process(self, mock_process_shard, mock_get_shards, mock_worker_info):
        mock_worker_info.return_value = None
        mock_get_shards.return_value = [0]
        mock_process_shard.return_value = iter([torch.tensor([1]), torch.tensor([2])])

        dataset = IterableFITSDataset("http://example.com")
        samples = list(dataset)

        assert len(samples) == 2
        mock_get_shards.assert_called_once_with(0, 1)

    @patch("torch.utils.data.get_worker_info")
    @patch.object(IterableFITSDataset, "_get_assigned_shards")
    @patch.object(IterableFITSDataset, "_process_shard")
    def test_iter_multi_process(self, mock_process_shard, mock_get_shards, mock_worker_info):
        worker_info = Mock()
        worker_info.id = 1
        worker_info.num_workers = 4
        mock_worker_info.return_value = worker_info

        mock_get_shards.return_value = [2, 3]
        mock_process_shard.side_effect = [
            iter([torch.tensor([1])]),
            iter([torch.tensor([2])])
        ]

        dataset = IterableFITSDataset("http://example.com")
        samples = list(dataset)

        assert len(samples) == 2
        mock_get_shards.assert_called_once_with(1, 4)


class TestTableChunkDataset:
    def test_init(self):
        file_paths = ["file1.fits", "file2.fits"]
        columns = ["col1", "col2"]
        dataset = TableChunkDataset(
            file_paths=file_paths,
            hdu=2,
            columns=columns,
            chunk_rows=5000,
            max_chunks=10,
            mmap=True,
            device="cuda",
            non_blocking_transfer=False,
            pin_memory_transfer=True,
            include_header=True,
        )

        assert dataset.file_paths == file_paths
        assert dataset.hdu == 2
        assert dataset.columns == columns
        assert dataset.chunk_rows == 5000
        assert dataset.max_chunks == 10
        assert dataset.mmap is True
        assert dataset.device == "cuda"
        assert dataset.non_blocking_transfer is False
        assert dataset.pin_memory_transfer is True
        assert dataset.include_header is True

    @patch("torchfits.table.scan_torch", create=True)
    @patch("torchfits.get_header", create=True)
    def test_iter_without_header(self, mock_get_header, mock_scan_torch):
        file_paths = ["file1.fits"]
        dataset = TableChunkDataset(file_paths, hdu=1, include_header=False, columns=["a"])

        chunk1 = {"a": torch.tensor([1])}
        chunk2 = {"a": torch.tensor([2])}
        mock_scan_torch.return_value = iter([chunk1, chunk2])

        chunks = list(dataset)

        assert len(chunks) == 2
        assert chunks[0] == chunk1
        assert chunks[1] == chunk2

        mock_get_header.assert_not_called()
        mock_scan_torch.assert_called_once_with(
            "file1.fits",
            hdu=1,
            columns=["a"],
            batch_size=10000,
            mmap=False,
            device="cpu",
            non_blocking=True,
            pin_memory=False,
        )

    @patch("torchfits.table.scan_torch", create=True)
    @patch("torchfits.get_header", create=True)
    def test_iter_with_header(self, mock_get_header, mock_scan_torch):
        file_paths = ["file1.fits"]
        dataset = TableChunkDataset(file_paths, hdu=1, include_header=True)

        mock_header = {"KEY": "VAL"}
        mock_get_header.return_value = mock_header

        chunk1 = {"a": torch.tensor([1])}
        mock_scan_torch.return_value = iter([chunk1])

        chunks = list(dataset)

        assert len(chunks) == 1
        data, header = chunks[0]
        assert data == chunk1
        assert header == mock_header
        mock_get_header.assert_called_once_with("file1.fits", 1)

    @patch("torchfits.table.scan_torch", create=True)
    def test_iter_with_transform(self, mock_scan_torch):
        dataset = TableChunkDataset(["file1.fits"], transform=lambda x: {"b": x["a"] + 1})

        mock_scan_torch.return_value = iter([{"a": torch.tensor([1])}])

        chunks = list(dataset)
        assert chunks[0]["b"].item() == 2

    @patch("torchfits.table.scan_torch", create=True)
    def test_iter_max_chunks(self, mock_scan_torch):
        dataset = TableChunkDataset(["file1.fits", "file2.fits"], max_chunks=3)

        # Each file yields 2 chunks, total 4
        # We need to use lists instead of iterators for side_effect to work correctly
        # when called multiple times in a loop
        def scan_generator(*args, **kwargs):
            if args[0] == "file1.fits":
                yield {"a": torch.tensor([1])}
                yield {"a": torch.tensor([2])}
            elif args[0] == "file2.fits":
                yield {"a": torch.tensor([3])}
                yield {"a": torch.tensor([4])}
            else:
                yield {"a": torch.tensor([5])}

        mock_scan_torch.side_effect = scan_generator

        chunks = list(dataset)

        # Should stop after 3 chunks
        assert len(chunks) == 3
        assert chunks[0]["a"].item() == 1
        assert chunks[1]["a"].item() == 2
        assert chunks[2]["a"].item() == 3
