import unittest
from replay_combined import *

META=dict(market=dict(event_ticker='EVENT',price_ranges=[dict(start='0',end='1',step='.01')]),
          series=dict(fee_type='quadratic_with_maker_fees',fee_multiplier=.5))

class CombinedTests(unittest.TestCase):
    def engine(self, wallet, ticker='A'):
        e=SharedTaker(Settings(ticker=ticker,warmup=1),.5,wallet)
        e.book.snapshot(dict(yes_dollars_fp=[['.47','100']],no_dollars_fp=[['.52','100']]),0)
        key=e.current_key(1)
        e.counts[key]=np.array([0,0,20]);e.history_n=20
        return e

    def test_compress_five_cents_to_one_new_price_pair(self):
        self.assertEqual(one_tick_quotes(4500,5000,100),(4700,4800))

    def test_existing_tick_does_not_claim_price_improvement(self):
        self.assertEqual(one_tick_quotes(4700,4800,100),(4700,4800))

    def test_resolved_fee_multiplier(self):
        m=Market('A',META,Wallet())
        self.assertEqual(m.c.maker_multiplier,.5)
        self.assertEqual(m.e.multiplier,.5)

    def test_one_tick_queue_zero_inside_old_spread(self):
        b=Book();b.update([(0,4500,1000),(1,5000,1000)],0,True)
        q=DynamicQuote(0,0,4700,4800)
        q.step(5_000_000,0,b)
        self.assertEqual([o['ahead'] for o in q.orders],[0,0])
        self.assertEqual(q.filled,[0,0])

    def test_existing_tick_joins_existing_queue(self):
        b=Book();b.update([(0,4700,1000),(1,4800,2000)],0,True)
        q=DynamicQuote(0,0,4700,4800);q.step(5_000_000,0,b)
        self.assertEqual([o['ahead'] for o in q.orders],[1000,2000])

    def test_governor_blocks_taker_but_keeps_shadow_labels(self):
        w=Wallet();w.owner=('maker','OTHER');e=self.engine(w)
        # A direct anchor with this bucket is made ready independently of sign.
        from paper_engine import bucket
        qt=e.book.quote();e.counts[e.current_key(bucket(qt['imbalance']))]=np.array([0,0,20])
        e.anchor(0,qt)
        self.assertIsNone(e.order);self.assertEqual(len(e.labels),1)
        self.assertEqual(w.owner,('maker','OTHER'))

    def test_funded_taker_entry_debits_shared_wallet(self):
        from paper_engine import bucket
        w=Wallet();e=self.engine(w);qt=e.book.quote()
        e.counts[e.current_key(bucket(qt['imbalance']))]=np.array([0,0,20])
        e.anchor(0,qt);self.assertEqual(w.owner,e.tag)
        e.execution(.004,True);self.assertEqual(w.cash,100)
        e.execution(.005,True);self.assertLess(w.cash,100)
        self.assertAlmostEqual(w.cash,100-value(e.position['entry'])-e.position['entry_fee'])

    def test_maker_warmup_does_not_book_shadow_profit(self):
        w=Wallet();m=Market('A',META,w)
        m.b.update([(0,4500,100),(1,5000,100)],0,True)
        ctx=dict(coarse=(1,),imb=(1,),full=(1,))
        m.decide_maker(0,ctx,w,False)
        self.assertEqual(w.cash,100);self.assertIsNone(m.active)
        self.assertEqual(len(m.shadows),1)

    def test_gate_allocates_only_one_maker_pocket(self):
        w=Wallet();m=Market('A',META,w)
        m.b.update([(0,4500,100),(1,5000,100)],0,True)
        ctx=dict(coarse=(1,),imb=(1,),full=(1,))
        for _ in range(100):m.quote_bank.add(ctx,[1,.005,1,0,0])
        m.decide_maker(0,ctx,w,False)
        self.assertIsNotNone(m.active);self.assertEqual(w.cash,95)
        m.decide_maker(1,ctx,w,False);self.assertEqual(w.cash,95)

    def test_completed_maker_pair_reconciles_shared_cash(self):
        w=Wallet();m=Market('A',META,w)
        m.b.update([(0,4500,100),(1,5000,100)],0,True)
        ctx=dict(coarse=(1,),imb=(1,),full=(1,))
        for _ in range(100):m.quote_bank.add(ctx,[1,.005,1,0,0])
        m.decide_maker(0,ctx,w,False)
        m.process_quotes(5_000_000,None,False,w,False)
        sell=dict(t=6_000_000,side=1,price=4500,qty=1,exchange_t=6_000_000)
        buy=dict(t=7_000_000,side=0,price=5000,qty=1,exchange_t=7_000_000)
        m.process_quotes(sell['t'],sell,False,w,False)
        self.assertEqual(w.cash,95);self.assertIsNotNone(m.active)
        m.process_quotes(buy['t'],buy,False,w,False)
        self.assertIsNone(m.active);self.assertIsNone(w.owner)
        self.assertAlmostEqual(w.cash,100+m.ledger[0]['net'])
        self.assertAlmostEqual(m.ledger[0]['gross'],.01)

if __name__=='__main__':unittest.main(verbosity=2)
