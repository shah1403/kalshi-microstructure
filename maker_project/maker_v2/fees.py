"""Fee + position cost rounding, including fractional fills."""
import math
from maker_replay import Book as BaseBook,EPS

def charge(q,p,maker,c):
    rate=.0175*c.maker_multiplier if maker else .07*c.taker_multiplier
    cost=q*p;raw=rate*q*p*(1-p)
    return max(0.,math.ceil((cost+raw)/c.fee_quantum-1e-8)*c.fee_quantum-cost)

class Book(BaseBook):
    def liquidate(self,side,q,c):
        remaining=q;gross=fees=0.
        for p,v in sorted(self.levels[side].items(),reverse=side==0):
            take=min(remaining,v*c.depth_fraction)
            px=(p/10000 if side==0 else 1-p/10000)-c.taker_slippage
            if px<0:continue
            gross+=take*px;fees+=charge(take,px,False,c);remaining-=take
            if remaining<EPS:return gross,fees
        return None
