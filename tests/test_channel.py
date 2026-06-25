import unittest

from glyph.channel import (English, Native, channel_bytes, glyph,
                           INVENTORY_SIZE, SYMBOL_BYTES)


class TestChannel(unittest.TestCase):
    def test_glyphs_distinct(self):
        n = Native()
        self.assertEqual(len(set(n.glyphs)), n.size)

    def test_glyph_out_of_range(self):
        with self.assertRaises(ValueError):
            glyph(INVENTORY_SIZE)

    def test_english_bytes_utf8(self):
        self.assertEqual(English().bytes("abc"), 3)

    def test_native_counts_symbols_only(self):
        n = Native()
        self.assertEqual(n.bytes(n.glyphs[0] + n.glyphs[1]), 2 * SYMBOL_BYTES)
        self.assertEqual(n.bytes("hello world"), 0)  # non-symbols are free-floor 0

    def test_channel_bytes_is_symbol_count(self):
        self.assertEqual(channel_bytes([0, 1, 2]), 3 * SYMBOL_BYTES)

    def test_mask_excludes_multitoken_glyphs(self):
        n = Native(size=4)
        g = n.glyphs

        def tokenize(s):
            return [1, 2] if s == g[2] else [1000 + g.index(s)]  # g[2] multi-token

        self.assertEqual(n.allowed_token_ids(tokenize), {1000, 1001, 1003})


if __name__ == "__main__":
    unittest.main()
