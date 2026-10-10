import unittest
import numpy as np
from distributions import *
from data_and_scores import pair_cost,Book,Config,flow_test
from execution import DynamicQuote,PairTaker
from fees import charge

def toy_rows(n=120):
    return [dict(t=(i+1)*10**9,side=i%2,qty=float(i+1),seg=0,valid=True,mid=.5,imb=0.,midstate=0,
                 bid=.48,ask=.52,bq=10.,aq=10.) for i in range(n)]

class Tests(unittest.TestCase):
    def book(self):
        b=Book();b.update([(0,4800,2),(1,5200,2)],0,True);return b
    def test_wait_targets_and_side_size_joint(self):
        rows=toy_rows();r=target(rows,0);y=r['y']
        self.assertEqual(r['truth']['sell_trades'],1);self.assertEqual(r['truth']['buy_trades'],2)
        self.assertEqual(y[SLICES['side_size']].argmax(),6)
        self.assertEqual(r['mature'],rows[50]['t'])
    def test_gap_and_end_not_zero_labels(self):
        rows=toy_rows();rows[10]['seg']=1
        self.assertIsNone(target(rows,0));self.assertIsNone(target(rows,100))
    def test_bank_probability_normalization(self):
        rows=toy_rows();c=context(rows,0);b=Bank(WIDTH)
        for i in range(20):b.add(context(rows,i),target(rows,i)['y'])
        p=b.predict(c,prior=PRIOR)
        for k in ['side','side_size','buy_n','sell_n','opp_n']:self.assertAlmostEqual(sum(p[SLICES[k]]),1)
        self.assertTrue(np.all(np.diff(wait_stats(p,'buy')['cdf_trades'])>=0))
    def test_no_arrival_mass_not_dropped(self):
        rows=toy_rows()
        for r in rows:r['side']=0
        lab=target(rows,0);self.assertEqual(lab['y'][SLICES['opp_n']][-1],1)
        b=Bank(WIDTH);c=context(rows,0);b.add(c,lab['y']);p=b.predict(c,prior=PRIOR)
        self.assertAlmostEqual(wait_stats(p,'opp')['restricted_mean_trades'],50)
        self.assertEqual(wait_stats(p,'opp')['probability_before_stop'],0)
    def test_future_extension_leaves_earlier_forecasts_unchanged(self):
        rows=toy_rows(900)
        _,_,a=flow_test(rows[:800]);_,_,b=flow_test(rows)
        self.assertGreater(len(a),0)
        for i,r in a.items():self.assertEqual(r['forecast'],b[i]['forecast'])
    def test_fractional_fee_plus_cost_rounding(self):
        c=Config();q=.01;p=.439;fee=charge(q,p,False,c)
        self.assertAlmostEqual((q*p+fee)/c.fee_quantum,round((q*p+fee)/c.fee_quantum))
        self.assertGreaterEqual(fee,.07*q*p*(1-p)-1e-12)
    def test_consistent_book_cannot_arbitrage(self):
        self.assertGreater(pair_cost(self.book(),1,Config()),1)
    def test_crossed_not_certified(self):
        p=PairTaker();self.assertFalse(p.propose(0,.47,.48,certified=False));self.assertEqual(p.cash,5)
    def test_pair_latency_and_hold_until_settlement(self):
        p=PairTaker();self.assertTrue(p.propose(0,.47,.48,certified=True))
        p.step(4_000_000,[.47,.48],[10,10]);self.assertEqual(p.cash,5)
        p.step(5_000_000,[.47,.48],[10,10]);self.assertEqual(p.state,'second_pending')
        p.step(10_000_000,[.47,.48],[10,10]);self.assertEqual(p.state,'held_pair');self.assertLess(p.cash,5)
        self.assertEqual(p.realized,0);p.settle();self.assertGreater(p.realized,0)
    def test_pair_second_leg_miss_unwinds_loss(self):
        p=PairTaker();p.propose(0,.47,.48,certified=True);p.step(5_000_000,[.47,.48],[10,10])
        p.step(10_000_000,[.47,.60],[10,10]);self.assertEqual(p.state,'unwind_pending')
        self.assertFalse(p.unwind(14_000_000,.40,10));self.assertTrue(p.unwind(15_000_000,.40,10));self.assertLess(p.realized,0)
    def test_partial_reprice_preserves_cost_basis(self):
        b=self.book();e=DynamicQuote(0,0,4800,5200);e.step(5_000_000,0,b)
        e.fill(0,.5,6_000_000,1);e.replace(0,4700,7_000_000)
        e.step(12_000_000,2,b);self.assertFalse(e.orders[0]['live'])
        e.step(17_000_000,3,b);self.assertTrue(e.orders[0]['live'])
        e.fill(0,.5,18_000_000,4);e.fill(1,1.,19_000_000,5)
        self.assertAlmostEqual(e.gross,.045);self.assertAlmostEqual(e.cash-5,e.gross-e.fees)
    def test_replace_resets_queue_and_cancel_race(self):
        b=self.book();e=DynamicQuote(0,0,4800,5200);e.step(5_000_000,0,b)
        e.replace(0,4700,6_000_000)
        e.trade(dict(t=10_000_000,side=1,price=4800,qty=2.4,exchange_t=10_000_000),1)
        self.assertAlmostEqual(e.filled[0],.4)
        e.step(11_000_000,2,b);e.trade(dict(t=12_000_000,side=1,price=4700,qty=10,exchange_t=12_000_000),3)
        self.assertAlmostEqual(e.filled[0],.4)
    def test_exit_delay_and_insufficient_depth(self):
        b=self.book();e=DynamicQuote(0,0,4800,5200);e.step(5_000_000,0,b);e.fill(0,1,6_000_000,1)
        e.stop(10_000_000,'test');e.step(19_000_000,2,b,True);self.assertEqual(e.held[0],1)
        b.update([(0,4000,.1),(1,4200,.1)],20_000_000,True);e.step(20_000_000,3,b,True)
        self.assertFalse(e.done);self.assertIsNone(e.last_mark)
        b.update([(0,4000,10),(1,4200,10)],21_000_000,True);e.step(21_000_000,4,b,True)
        self.assertTrue(e.done);self.assertLess(e.result()['net'],0)

if __name__=='__main__':unittest.main(verbosity=2)
