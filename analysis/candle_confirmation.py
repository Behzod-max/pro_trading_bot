from dataclasses import dataclass, field
from typing import Any, Dict, List

@dataclass
class CandleConfirmation:
    direction:str="NEUTRAL"
    pattern:str="NO_PATTERN"
    score:float=0.0
    confirmed:bool=False
    strength:str="NONE"
    reasons:List[str]=field(default_factory=list)

class CandleConfirmationEngine:
    CONFIRM_THRESHOLD=45.0
    @staticmethod
    def num(v,d=0.0):
        try:return float(v) if float(v)==float(v) else d
        except:return d
    def analyze(self,df:Any)->CandleConfirmation:
        try:
            if df is None or len(df)<5 or not {"open","high","low","close"}.issubset(set(df.columns)):raise ValueError("OHLC ma'lumotlari yetarli emas")
            c=df.iloc[-2];p=df.iloc[-3]
        except Exception as e:return CandleConfirmation(reasons=[f"Candle ma'lumotlari xatosi: {e}"])
        def stats(x):
            o,h,l,cl=[self.num(x[k]) for k in ("open","high","low","close")];r=max(h-l,0);b=abs(cl-o);return dict(o=o,h=h,l=l,c=cl,r=r,b=b,up=h-max(o,cl),lo=min(o,cl)-l, bull=cl>o,bear=cl<o)
        c=stats(c);p=stats(p);buy=sell=0;bp=[];sp=[];rs=[]
        if p["bear"] and c["bull"] and c["o"]<=p["c"] and c["c"]>=p["o"] and c["b"]>=p["b"]*.9:buy+=45;bp.append("BULLISH_ENGULFING");rs.append("Bullish Engulfing aniqlandi.")
        if p["bull"] and c["bear"] and c["o"]>=p["c"] and c["c"]<=p["o"] and c["b"]>=p["b"]*.9:sell+=45;sp.append("BEARISH_ENGULFING");rs.append("Bearish Engulfing aniqlandi.")
        if c["lo"]>=c["b"]*2 and c["up"]<=max(c["b"]*.75,c["r"]*.15):buy+=35;bp.append("BULLISH_PIN_BAR");rs.append("Bullish Pin Bar aniqlandi.")
        if c["up"]>=c["b"]*2 and c["lo"]<=max(c["b"]*.75,c["r"]*.15):sell+=35;sp.append("BEARISH_PIN_BAR");rs.append("Bearish Pin Bar aniqlandi.")
        if c["bull"] and c["b"]>=c["r"]*.65 and c["up"]<=c["r"]*.2:buy+=20;bp.append("STRONG_BULLISH_CANDLE");rs.append("Kuchli bullish candle.")
        if c["bear"] and c["b"]>=c["r"]*.65 and c["lo"]<=c["r"]*.2:sell+=20;sp.append("STRONG_BEARISH_CANDLE");rs.append("Kuchli bearish candle.")
        if buy>sell:d="BUY";score=min(100,buy);pat=" + ".join(bp)
        elif sell>buy:d="SELL";score=min(100,sell);pat=" + ".join(sp)
        else:d="NEUTRAL";score=0;pat="NO_PATTERN"
        strength="STRONG" if score>=70 else "MEDIUM" if score>=45 else "WEAK" if score>=20 else "NONE"
        return CandleConfirmation(d,pat,round(score,2),score>=45,strength,rs or ["Tasdiqlovchi pattern topilmadi."])
    def analyze_dict(self,df):
        r=self.analyze(df);return {"direction":r.direction,"pattern":r.pattern,"score":r.score,"confirmed":r.confirmed,"strength":r.strength,"reasons":r.reasons}
