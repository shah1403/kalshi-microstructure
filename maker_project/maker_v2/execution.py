"""Maker cancel/replace and inventory accounting, plus separate paired taker ledger."""
from pathlib import Path
import sys,math
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from maker_replay import Config,EPS
from fees import Book,charge
NS=10**9

class DynamicQuote:
    def __init__(self,t,n,bid,ask,c=None):
        self.c=c or Config();self.t=t;self.n=n;self.lat=round(self.c.latency_ms*1e6)
        self.cash=self.c.bankroll;self.held=[0.,0.];self.basis=[0.,0.];self.filled=[0.,0.]
        self.gross=0.;self.fees=0.;self.matched=0.;self.fills=[];self.marks=[];self.actions=[]
        self.peak=self.cash;self.dd=0.;self.unknown_marks=0;self.last_mark=self.cash
        self.done=False;self.closed_t=None;self.stopping=False;self.exit_due=None;self.censored=False;self.reason=None
        self.orders=[dict(p=bid,arrival=t+self.lat,live=False,ahead=0.,cancel=None,replacement=None),
                     dict(p=ask,arrival=t+self.lat,live=False,ahead=0.,cancel=None,replacement=None)]
    def fill(self,side,qty,t,n):
        qty=min(qty,self.c.quantity-self.filled[side])
        if qty<=EPS:return
        p=self.orders[side]['p']/10000 if side==0 else 1-self.orders[side]['p']/10000
        fee=charge(qty,p,True,self.c);self.cash-=qty*p+fee;self.fees+=fee
        self.held[side]+=qty;self.basis[side]+=qty*p;self.filled[side]+=qty
        self.fills.append(dict(t=t,n=n,side=side,qty=qty,price=p,fee=fee,maker=True))
        if self.filled[side]>=self.c.quantity-EPS:self.orders[side]['live']=False
        pair=min(self.held)
        if pair>EPS:
            costs=[self.basis[s]/self.held[s] for s in (0,1)]
            self.cash+=pair;self.gross+=pair*(1-sum(costs));self.matched+=pair
            for s in (0,1):self.held[s]-=pair;self.basis[s]-=pair*costs[s]
        if min(self.filled)>=self.c.quantity-EPS and max(self.held)<EPS:self.finish(t)
    def finish(self,t):
        self.done=True;self.closed_t=t
        for o in self.orders:o['live']=False
        assert abs(self.cash-self.c.bankroll-self.gross+self.fees)<1e-7
    def trade(self,e,n):
        if self.done:return
        side=1-e['side'];o=self.orders[side];t=e['t']
        if not o['live'] or (o['cancel'] is not None and t>=o['cancel']):return
        if e.get('exchange_t') and e['exchange_t']<o['arrival']:return
        p=e['price'];through=p>o['p'] if side==1 else p<o['p']
        if p!=o['p'] and not through:return
        if through:o['ahead']=0.;v=e['qty']
        else:
            used=min(o['ahead'],e['qty']);o['ahead']-=used;v=e['qty']-used
        self.fill(side,v,t,n)
    def replace(self,side,price,t):
        o=self.orders[side]
        if self.stopping or self.filled[side]>=self.c.quantity-EPS or o['p']==price or o['cancel'] is not None:return
        # Cancel acknowledgement first; replacement incurs an additional entry delay.
        o['cancel']=t+self.lat;o['replacement']=price
        self.actions.append(dict(t=t,action='replace_request',side=side,price=price/10000))
    def stop(self,t,reason):
        if self.stopping:return
        self.stopping=True;self.reason=reason
        for o in self.orders:
            o['cancel']=min(o['cancel'],t+self.lat) if o['cancel'] is not None else t+self.lat
            o['replacement']=None
        self.exit_due=max(o['cancel'] for o in self.orders)+self.lat
        self.actions.append(dict(t=t,action='cancel_and_exit',reason=reason))
    def mark(self,t,b):
        if max(self.held)<EPS:equity=self.cash
        elif not b.valid:self.last_mark=None;self.unknown_marks+=1;return
        else:
            equity=self.cash
            for s,q in enumerate(self.held):
                if q<=EPS:continue
                f=b.liquidate(s,q,self.c)
                if f is None:self.last_mark=None;self.unknown_marks+=1;return
                equity+=f[0]-f[1]
        self.last_mark=equity;self.peak=max(self.peak,equity);self.dd=max(self.dd,self.peak-equity)
        if not self.marks or abs(self.marks[-1][1]-equity)>1e-9:self.marks.append([t,equity])
    def step(self,t,n,b,is_book=False):
        if self.done:self.mark(t,b);return
        if n-self.n>=50 or t-self.t>=300*NS:self.stop(t,'trade_or_wall_timeout')
        qt=b.quote() if b.valid else None
        for s,o in enumerate(self.orders):
            if o['cancel'] is not None and t>=o['cancel']:
                o['live']=False
                if o['replacement'] is not None and not self.stopping:
                    o['p']=o['replacement'];o['arrival']=o['cancel']+self.lat;o['replacement']=None;o['cancel']=None
                    o['pending']=True
            if self.stopping or self.filled[s]>=self.c.quantity-EPS:continue
            if not o['live'] and o['cancel'] is None and t>=o['arrival']:
                if not qt:continue
                crosses=o['p']>=qt[1] if s==0 else o['p']<=qt[0]
                if crosses:
                    self.stop(t,'post_only_reject');break
                o['ahead']=b.quantity(s,o['p'])*self.c.queue_fraction;o['live']=True
                self.actions.append(dict(t=t,action='order_active',side=s,queue=o['ahead'],price=o['p']/10000))
        if self.stopping and t>=max(o['cancel'] or t for o in self.orders) and max(self.held)<EPS:
            self.finish(t)
        elif self.exit_due is not None and t>=self.exit_due and is_book and b.valid:
            for s,q in enumerate(self.held):
                if q<=EPS:continue
                f=b.liquidate(s,q,self.c)
                if f is None:continue
                gross,fee=f;self.cash+=gross-fee;self.gross+=gross-self.basis[s];self.fees+=fee
                self.fills.append(dict(t=t,n=n,side=s,qty=q,price=gross/q,fee=fee,maker=False))
                self.held[s]=self.basis[s]=0.
            if max(self.held)<EPS:self.finish(t)
        self.mark(t,b)
    def result(self):
        return dict(t=self.t,closed_t=self.closed_t,gross=self.gross,fees=self.fees,
                    net=self.cash-self.c.bankroll if self.done else None,held=self.held,filled=self.filled,
                    matched=self.matched,maxdd=self.dd,censored=self.censored,reason=self.reason,
                    last_mark=self.last_mark,unknown_marks=self.unknown_marks,actions=self.actions,fills=self.fills,marks=self.marks)

class PairTaker:
    """Two sequential fill-or-kill legs. Paired funds are locked until settlement.

    Real-data caller must certify quote consistency; crossed reconstructions fail
    this gate. This ledger is independently tested with synthetic quote streams.
    """
    def __init__(self,c=None):
        self.c=c or Config();self.cash=self.c.bankroll;self.state='idle';self.filled=[0.,0.]
        self.costs=0.;self.fees=0.;self.due=0;self.limits=None;self.q=0.;self.locked=0.;self.realized=0.
    def propose(self,t,yes,no,q=1,certified=False):
        cost=q*(yes+no)+charge(q,yes,False,self.c)+charge(q,no,False,self.c)
        if not certified or self.state!='idle' or cost>=q or cost>self.cash:return False
        self.limits=[yes,no];self.q=q;self.state='first_pending';self.due=t+int(self.c.latency_ms*1e6);return True
    def step(self,t,asks,sizes,certified=True):
        if t<self.due or not certified:return
        if self.state in ('first_pending','second_pending'):
            s=0 if self.state=='first_pending' else 1
            if asks[s]>self.limits[s]+EPS or sizes[s]*self.c.depth_fraction<self.q:
                self.state='idle' if s==0 else 'unwind_pending';self.due=t+int(self.c.latency_ms*1e6);return
            fee=charge(self.q,asks[s],False,self.c);paid=self.q*asks[s]+fee
            if paid>self.cash:raise AssertionError('cash exceeded')
            self.cash-=paid;self.costs+=self.q*asks[s];self.fees+=fee;self.filled[s]=self.q
            self.state='second_pending' if s==0 else 'held_pair';self.due=t+int(self.c.latency_ms*1e6)
            if s==1:self.locked=self.q
    def unwind(self,t,bid,depth):
        if self.state!='unwind_pending' or t<self.due or depth*self.c.depth_fraction<self.q:return False
        fee=charge(self.q,bid,False,self.c);self.cash+=self.q*bid-fee;self.fees+=fee
        self.realized=self.cash-self.c.bankroll;self.state='finished';return True
    def settle(self):
        if self.state!='held_pair':raise ValueError('Both legs must be filled')
        self.cash+=self.locked;self.locked=0.;self.state='settled';self.realized=self.cash-self.c.bankroll
