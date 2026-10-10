"""Deterministic offline tests; no API credential or exchange connection required."""
import asyncio
from dataclasses import replace
import gzip
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from paper_engine import (Settings,Book,Engine,Feed,PaperSession,action,fee,value,
                          replay_recording,auth_headers,market_metadata)


def snap(bid=.49, ask=.51, bq=100, aq=10):
    return {'yes_dollars_fp':[[f'{bid:.4f}',str(bq)]],
            'no_dollars_fp':[[f'{1-ask:.4f}',str(aq)]]}


def setup(**kw):
    cfg=Settings(ticker='TEST',depth_fraction=1,adverse_slippage=0,**kw)
    e=Engine(cfg)
    e.book.snapshot(snap(),0)
    return e


def advance_book(e,t,bid=.49,ask=.51):
    e.book.snapshot(snap(bid,ask),t)
    e.tick(t,is_book=True)


class CoreTests(unittest.TestCase):
    def test_complement_bid_ask_and_vwap(self):
        e=setup()
        e.book.snapshot({'yes_dollars_fp':[['.49','2'],['.48','3']],
                         'no_dollars_fp':[['.49','2'],['.48','3']]},0)
        self.assertAlmostEqual(value(e.book.fill(1,True,4,e.cfg)),2*.51+2*.52)
        self.assertAlmostEqual(value(e.book.fill(-1,True,4,e.cfg)),2*.51+2*.52)
        self.assertAlmostEqual(value(e.book.fill(-1,False,4,e.cfg)),2*.49+2*.48)
        self.assertIsNone(e.book.fill(1,True,6,e.cfg))

    def test_depth_fraction_and_slippage(self):
        e=setup()
        cfg=replace(e.cfg,depth_fraction=.5,adverse_slippage=.002)
        self.assertIsNone(e.book.fill(1,True,6,cfg))
        self.assertAlmostEqual(value(e.book.fill(1,True,5,cfg)),5*.512)

    def test_snapshot_and_negative_delta_validation(self):
        e=setup()
        with self.assertRaises(ValueError): e.book.snapshot({'yes_dollars_fp':[['.49','nan']],'no_dollars_fp':[]},1)
        with self.assertRaises(ValueError): e.book.delta({'side':'yes','price_dollars':'.49','delta_fp':'-200'},1)

    def test_fees_rounding_and_multiplier(self):
        c=Settings()
        self.assertAlmostEqual(fee([(1,.5)],1,c),.0175)
        self.assertAlmostEqual(fee([(1,.5)],1,replace(c,fee_quantum='.01')),.02)
        self.assertAlmostEqual(fee([(1,.5)],2,c),.035)
        self.assertEqual(fee([(10,.5)],0,c),0)

    def test_action_45_strict_and_ties(self):
        self.assertEqual(action([.04,.50,.46]),(1,.45,'reduced_45'))
        self.assertEqual(action([.46,.50,.04]),(-1,.45,'reduced_45'))
        self.assertEqual(action([.05,.50,.45])[0],0)
        self.assertEqual(action([.01,.495,.495])[0],0)
        self.assertEqual(action([.05,.15,.8]),(1,1.,'normal'))

    def test_position_size_floor_and_cash(self):
        e=setup()
        q,n=e.size([.04,.5,.46],1,.45)
        self.assertEqual(q,math_floor(n*.45))
        self.assertLessEqual(value(e.book.fill(1,True,n,e.cfg))+fee(e.book.fill(1,True,n,e.cfg),1,e.cfg),5*.46+1e-8)
        e.cash=.1
        self.assertEqual(e.size([.04,.5,.46],1,.45),(0,0))

    def test_smoothing_and_no_unconditional_fallback(self):
        e=setup()
        e.state=1
        e.counts[(4,1)]=np.array([3,0,0])
        np.testing.assert_allclose(e.probabilities(4)[0],[3.5/4.5,.5/4.5,.5/4.5])
        np.testing.assert_allclose(e.probabilities(0)[0],np.ones(3)/3)

    def test_label_matures_after50_not_early_hit(self):
        e=setup()
        e.anchor(0,e.book.quote())
        advance_book(e,.01)
        advance_book(e,.02,.56,.58)
        self.assertEqual(e.forecasts[0]['y'],1)
        self.assertEqual(e.history_n,0)
        e.tick(.03)
        self.assertEqual(e.state,1)
        for i in range(50): e.tick(.04+i*.001,is_trade=True)
        self.assertEqual(e.history_n,0)
        e.tick(.2)
        self.assertGreaterEqual(e.history_n,1)

    def test_reference_no_crossing_timeout(self):
        e=setup(); e.anchor(0,e.book.quote()); advance_book(e,.01)
        for i in range(50): e.tick(.02+i*.001,is_trade=True)
        e.tick(.2)
        self.assertEqual(e.forecasts[0]['y'],0)
        self.assertEqual(e.forecasts[0]['label_status'],'matured')

    def test_entry_and_exit_both_delayed_no_future_price_filter(self):
        e=setup()
        e.forecasts=[{'taken':False}]
        e.order=dict(id=0,side=1,q=2,t=0,due=.005,start_count=0,tier='normal')
        e.tick(.003,is_book=True)
        self.assertIsNone(e.position)
        advance_book(e,.006)
        self.assertIsNotNone(e.position)
        advance_book(e,.02,.56,.58)
        self.assertEqual(len(e.trades),0)
        advance_book(e,.026,.47,.49)
        self.assertEqual(len(e.trades),1)
        self.assertLess(e.trades[0]['net'],0)
        self.assertEqual(e.trades[0]['exit_reason'],'profit_opportunity')
        self.assertAlmostEqual(e.cash,e.cfg.bankroll+e.realized_net)

    def test_insufficient_exit_not_fabricated(self):
        e=setup(); e.forecasts=[{'taken':False}]
        e.order=dict(id=0,side=1,q=2,t=0,due=.005,start_count=0,tier='normal')
        advance_book(e,.01)
        e.request_exit(.02,'50_trade_timeout')
        e.book.snapshot(snap(bq=1),.03); e.tick(.03,is_book=True)
        self.assertIsNotNone(e.position)
        self.assertIsNone(e.mark)
        advance_book(e,.04)
        self.assertIsNone(e.position)

    def test_gap_preserves_position_and_censors_label(self):
        e=setup(); e.anchor(0,e.book.quote())
        e.order=dict(id=0,side=1,q=1,t=0,due=.005,start_count=0,tier='normal')
        advance_book(e,.01)
        e.gap(.02,'test_gap')
        self.assertIsNotNone(e.position)
        self.assertTrue(e.forecasts[0]['label_status'].startswith('censored'))
        self.assertEqual(e.history_n,0)
        e.connected=True
        advance_book(e,.04,.45,.47)
        self.assertEqual(len(e.trades),1)
        self.assertTrue(e.trades[0]['gap_exposed'])

    def test_settlement_never_closes_position(self):
        e=setup(); e.forecasts=[{'taken':False}]
        e.order=dict(id=0,side=1,q=1,t=0,due=.005,start_count=0,tier='normal')
        advance_book(e,.01)
        e.tradable=False
        advance_book(e,.03,.98,.99)
        self.assertIsNotNone(e.position)
        self.assertEqual(len(e.trades),0)
        self.assertIsNone(e.mark)

    def test_trade_id_dedup_and_not_volume_clock(self):
        e=setup(); f=Feed(e)
        for seq in [1,2]:
            f.message({'type':'trade','sid':2,'seq':seq,'msg':{'trade_id':'abc','market_ticker':'TEST','count_fp':'500'}},.01*seq,0)
        self.assertEqual(e.count,1)

    def test_sequence_gap_fails_closed(self):
        e=setup(); f=Feed(e)
        f.message({'type':'orderbook_snapshot','sid':1,'seq':1,'msg':{'market_ticker':'TEST',**snap()}},.01,0)
        with self.assertRaises(RuntimeError):
            f.message({'type':'orderbook_delta','sid':1,'seq':3,'msg':{'market_ticker':'TEST'}},.02,0)

    def test_lifecycle_terminal_freezes(self):
        e=setup(); f=Feed(e)
        f.message({'type':'market_lifecycle_v2','msg':{'market_ticker':'TEST','event_type':'determined'}},.01,0)
        self.assertFalse(e.tradable)

    def test_imbalance_does_not_impose_direction(self):
        e=setup(warmup=1)
        e.history_n=30; e.state=1; e.counts[(4,1)]=np.array([20,1,0])
        e.book.snapshot(snap(bq=1000,aq=10),0)
        e.anchor(0,e.book.quote())
        self.assertEqual(e.forecasts[0]['side'],-1)

    def test_budget_never_negative_on_large_request(self):
        e=setup();e.cash=.01;e.forecasts=[{'taken':False}]
        e.order=dict(id=0,side=1,q=2,t=0,due=.005,start_count=0,tier='normal')
        advance_book(e,.01)
        self.assertIsNone(e.position)
        self.assertEqual(e.cash,.01)

    def test_event_fee_override(self):
        responses=[{'market':{'event_ticker':'E'}},{'event':{'series_ticker':'S','fee_multiplier_override':2}},
                   {'series':{'fee_type':'quadratic','fee_multiplier':1}}]
        with patch('paper_engine.public_get',side_effect=responses):
            self.assertEqual(market_metadata('TEST')[1]['fee_multiplier'],2)

    def test_rsa_signature_verifies(self):
        from cryptography.hazmat.primitives.asymmetric import rsa,padding
        from cryptography.hazmat.primitives import hashes
        import base64
        key=rsa.generate_private_key(public_exponent=65537,key_size=2048)
        h=auth_headers('dummy',key)
        key.public_key().verify(base64.b64decode(h['KALSHI-ACCESS-SIGNATURE']),
            (h['KALSHI-ACCESS-TIMESTAMP']+'GET/trade-api/ws/v2').encode(),
            padding.PSS(mgf=padding.MGF1(hashes.SHA256()),salt_length=padding.PSS.DIGEST_LENGTH),hashes.SHA256())

    def test_recording_replay_same_counts_and_pnl(self):
        e=setup();f=Feed(e)
        rows=[{'kind':'config','t':0,'config':as_dict(e.cfg)}]
        for i in range(100):
            data={'type':'orderbook_snapshot' if i==0 else 'orderbook_delta','sid':1,'seq':i+1,
                  'msg':{'market_ticker':'TEST',**(snap() if i==0 else {'price_dollars':'.49','delta_fp':'0','side':'yes'})}}
            t=i*.02
            rows.append(dict(kind='message',t=t,wall=0,data=data)); f.message(data,t,0)
            data={'type':'trade','sid':2,'seq':i+1,'msg':{'market_ticker':'TEST','trade_id':str(i)}}
            rows.append(dict(kind='message',t=t+.001,wall=0,data=data));f.message(data,t+.001,0)
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'feed.gz'
            with gzip.open(path,'wt') as out:
                for row in rows: out.write(json.dumps(row)+'\n')
            re=replay_recording(path)
            self.assertEqual(re.stats(),e.stats())

    def test_session_ack_before_snapshot_and_market_data_only(self):
        import sys, time
        from types import SimpleNamespace
        sent=[]
        calls=[]
        async def exercise(directory):
            session=PaperSession(Settings(ticker='TEST'), 'TEST-ID', object(), directory)
            messages=[{'type':'subscribed','msg':{'channel':'orderbook_delta','sid':1}},
                      {'type':'orderbook_snapshot','sid':1,'seq':1,
                       'msg':{'market_ticker':'TEST',**snap()}},
                      {'type':'trade','sid':2,'seq':1,
                       'msg':{'market_ticker':'TEST','trade_id':'one'}}]
            class FakeSocket:
                async def __aenter__(self): return self
                async def __aexit__(self,*args): return False
                async def send(self,data): sent.append(json.loads(data))
                async def recv(self):
                    if messages: return json.dumps(messages.pop(0))
                    session.engine.request_stop(time.monotonic())
                    return json.dumps({'type':'ok'})
            def connect(*args,**kwargs):
                calls.append(args)
                return FakeSocket()
            async def metadata():
                session.engine.metadata_t=time.monotonic()
            session.refresh_metadata=metadata
            with patch.dict(sys.modules,{'websockets':SimpleNamespace(connect=connect)}), patch('paper_engine.auth_headers',return_value={}):
                await asyncio.wait_for(session.run(),2)
            self.assertIsNone(session.error)
            self.assertEqual(session.engine.count,1)
            self.assertEqual(session.engine.mark,session.cfg.bankroll)
        with tempfile.TemporaryDirectory() as d: asyncio.run(exercise(d))
        self.assertEqual(len(calls),1)
        self.assertEqual([x['cmd'] for x in sent],['subscribe']*3)
        self.assertEqual([x['params']['channels'][0] for x in sent],
                         ['orderbook_delta','trade','market_lifecycle_v2'])
        self.assertNotIn('market_tickers',sent[-1]['params'])


def math_floor(x):
    import math
    return math.floor(x+1e-12)


def as_dict(x):
    from dataclasses import asdict
    return asdict(x)


if __name__=='__main__': unittest.main()
