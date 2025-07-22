from ssd.data import DataAugmenter


class TestDataAugmenter:
    def test_init(self):
        """
        Test that we can initialise the data augmenter.
        """
        augmenter = DataAugmenter(300, 300)
        assert isinstance(augmenter, DataAugmenter)
