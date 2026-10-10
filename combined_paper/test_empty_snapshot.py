import unittest
from replay_combined import RawBook, Market, Wallet
from test_combined import META


class EmptySnapshotTests(unittest.TestCase):
    def test_empty_snapshot_clears_old_depth_and_cannot_fill(self):
        b = RawBook()
        b.snapshot({'yes_dollars_fp': [['.49','10']], 'no_dollars_fp': [['.50','10']]}, 0)
        self.assertIsNotNone(b.quote())
        b.snapshot({'market_ticker': 'A', 'market_id': 'test'}, 1)
        self.assertEqual(b.levels, {'yes': {}, 'no': {}})
        self.assertIsNone(b.quote())
        b.delta({'side': 'yes', 'price_dollars': '.49', 'delta_fp': '10'}, 2)
        self.assertIsNone(b.quote())
        b.delta({'side': 'no', 'price_dollars': '.50', 'delta_fp': '10'}, 3)
        self.assertIsNotNone(b.quote())

    def test_one_omitted_side_and_legacy_rejection(self):
        b = RawBook()
        b.snapshot({'market_ticker': 'A', 'no_dollars_fp': [['.50','10']]}, 0)
        self.assertEqual(b.levels['yes'], {})
        self.assertIsNone(b.quote())
        with self.assertRaises(ValueError):
            b.snapshot({'market_ticker': 'A', 'yes': [[49,10]], 'no': [[50,10]]}, 1)
        with self.assertRaises(ValueError):
            b.snapshot({'market_ticker': 'A', 'yes_dollars_fp': 'bad'}, 1)

    def test_feed_empty_resnapshot_invalidates_prior_quote(self):
        m = Market('A', META, Wallet())
        m.feed.message({'type':'orderbook_snapshot','sid':1,'seq':1,'msg':
                       {'market_ticker':'A','yes_dollars_fp':[['.49','10']],
                        'no_dollars_fp':[['.50','10']]}}, 0, 0)
        self.assertIsNotNone(m.e.book.quote())
        m.feed.message({'type':'orderbook_snapshot','sid':1,'seq':2,
                        'msg':{'market_ticker':'A','market_id':'test'}}, 1, 1)
        self.assertIsNone(m.e.book.quote())


if __name__ == '__main__':
    unittest.main()
