import json
import os
import tempfile
import unittest

from prepare_dataset import copy_annotations


class AnnotationCopyTests(unittest.TestCase):
    def test_copies_annotations_into_prepared_split(self):
        with tempfile.TemporaryDirectory() as root:
            source = os.path.join(root, "source")
            output = os.path.join(root, "prepared")
            os.makedirs(os.path.join(source, "Train"))
            expected = {"Actions": {"0": {"Action class": "Tackling"}}}
            source_file = os.path.join(source, "Train", "annotations.json")
            with open(source_file, "w", encoding="utf-8") as handle:
                json.dump(expected, handle)

            copied = copy_annotations(source, output, ["Train"])

            destination = os.path.join(output, "Train", "annotations.json")
            self.assertEqual(copied, [destination])
            with open(destination, encoding="utf-8") as handle:
                self.assertEqual(json.load(handle), expected)


if __name__ == "__main__":
    unittest.main()
