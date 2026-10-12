from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

@dataclass
class SignalResult:
    direction: str = "NEUTRAL"
    score: float = 0.0
    status: str = "NO_SIGNAL"
    entry: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit_1: Optional[float] = None
    take_profit_2: Optional[float] = None
    risk_reward_1: Optional[float] = None
    risk_reward_2: Optional[float] = None
    reasons: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    components: Dict[str, float] = field(default_factory=dict)

class SignalEngine:
    WEIGHTS={"MTF":25.0,"MARKET_STRUCTURE":20.0,"LIQUIDITY":15.0,"FIBONACCI":15.0,"TRENDLINE":10.0,"CANDLE":15.0}
    def __init__(self,early_threshold=60,confirmed_threshold=75,high_confidence_threshold=85):
        self.early_threshold=early_threshold; self.confirmed_threshold=confirmed_threshold; self.high_confidence_threshold=high_confidence_threshold
    @staticmethod
    def get(o,*names,default=None):
        for n in names:
            if isinstance(o,dict) and o.get(n) is not None:return o[n]
            try:
                v=getattr(o,n)
                if v is not None:return v
            except Exception:pass
        return default
    @staticmethod
    def num(v,d=0.0):
        try:return float(v) if float(v)==float(v) else d
        except:return d
    @staticmethod
    def direction(v):
        s=str(v or "NEUTRAL").upper().strip()
        return {"LONG":"BUY","BULLISH":"BUY","BULL":"BUY","UP":"BUY","SHORT":"SELL","BEARISH":"SELL","BEAR":"SELL","DOWN":"SELL"}.get(s,s if s in ("BUY","SELL") else "NEUTRAL")
    @staticmethod
    def one(v):
        if v is None:return []
        return list(v) if isinstance(v,(list,tuple)) else [v]

    def score_mtf(self,x):
        w={"D1":5,"H4":5,"H1":4,"M30":4,"M15":3,"M5":3,"M1":1}; b=s=0; r=[]
        for tf,wt in w.items():
            v=x.get(tf) if isinstance(x,dict) else None
            d=self.direction(self.get(v,"direction","trend","bias",default=v))
            if d=="BUY":b+=wt;r.append(f"{tf}: BUY")
            elif d=="SELL":s+=wt;r.append(f"{tf}: SELL")
        return (("BUY",b,r) if b>s else ("SELL",s,r) if s>b else ("NEUTRAL",0,r))

    def score_structure(self,x):
        x=x.get("M5") if isinstance(x,dict) else x; trend=self.direction(self.get(x,"trend","direction","bias",default="NEUTRAL")); ev=self.one(self.get(x,"events",default=[])); e=ev[-1] if ev else None; typ=str(self.get(e,"event","type","name",default="")).upper(); score=12 if trend in ("BUY","SELL") else 0; r=[]
        if trend!="NEUTRAL":r.append(f"Structure: {trend}")
        ed=self.direction(self.get(e,"direction","trend",default=trend))
        if typ=="BOS":score+=8 if ed==trend else 4;r.append(f"BOS: {ed}")
        elif typ=="CHOCH":score+=6 if ed==trend else 3;r.append(f"CHoCH: {ed}")
        return trend,min(20,score),r

    def score_liquidity(self,x):
        x=x.get("M5") if isinstance(x,dict) else x; a=self.one(self.get(x,"sweeps","liquidity_sweeps",default=[])); e=a[-1] if a else None
        if e is None:return "NEUTRAL",0,[]
        d=self.direction(self.get(e,"direction","bias",default="NEUTRAL")); score=7;r=["Liquidity sweep aniqlandi"]
        if self.get(e,"confirmed","is_confirmed",default=False):score+=5;r.append("Sweep tasdiqlangan")
        strength=self.num(self.get(e,"strength",default=0))
        if strength>=.7:score+=3;r.append("Sweep kuchli")
        return d,min(15,score),r

    def score_fibonacci(self,x):
        x=x.get("M5") if isinstance(x,dict) else x
        if x is None:return "NEUTRAL",0,[]
        d=self.direction(self.get(x,"direction","trend","bias",default="NEUTRAL"));score=6 if d!="NEUTRAL" else 0;r=[]
        if d!="NEUTRAL":r.append(f"Fibonacci: {d}")
        g=self.get(x,"in_golden_zone","golden_zone","is_golden_zone",default=False)
        if isinstance(g,str):g=g.upper() in ("TRUE","HA","YES","1")
        if g:score+=5;r.append("Golden Zone 0.618-0.786")
        n=self.get(x,"nearest_level","nearest_fibonacci_level",default=None)
        if n is not None:score+=2;r.append(f"Yaqin Fibonacci: {n}")
        p=self.get(x,"premium","is_premium",default=False);di=self.get(x,"discount","is_discount",default=False)
        if (d=="SELL" and p) or (d=="BUY" and di):score+=2;r.append("Premium/Discount mos")
        return d,min(15,score),r

    def score_trendline(self,x):
        x=x.get("M5") if isinstance(x,dict) else x; a=self.one(self.get(x,"breakouts","events","trendline_events",default=[]));e=a[-1] if a else None
        if e is None:return "NEUTRAL",0,[]
        d=self.direction(self.get(e,"direction","bias",default="NEUTRAL"));
        if d=="NEUTRAL":return d,0,[]
        score=6;strength=self.num(self.get(e,"strength",default=0));score+=4 if strength>=.7 else 2 if strength>=.4 else 0
        return d,min(10,score),[f"Trendline: {d}"]

    def score_candle(self,x):
        x=x.get("M5") if isinstance(x,dict) else x
        if x is None:return "NEUTRAL",0,[]
        d=self.direction(self.get(x,"direction","bias",default="NEUTRAL"));raw=max(0,min(100,self.num(self.get(x,"score",default=0))));score=raw*.15
        return d,min(15,score),[f"Candle: {self.get(x,'pattern',default='NO_PATTERN')}",f"Candle score: {raw:.1f}/100"]

    def calculate_levels(self,d,df,structure=None):
        if d not in ("BUY","SELL") or df is None or len(df)<2:return (None,)*6
        row=df.iloc[-2];entry=self.num(row.get("close"));atr=self.num(row.get("atr_14",row.get("atr",0)));atr=max(atr,entry*.001,.01)
        hi=self.get(self.get(structure,"last_swing_high",default=None),"price",default=self.get(structure,"last_swing_high",default=0));lo=self.get(self.get(structure,"last_swing_low",default=None),"price",default=self.get(structure,"last_swing_low",default=0));hi=self.num(hi);lo=self.num(lo)
        if d=="BUY":sl=min(entry-.8*atr,lo if 0<lo<entry else entry-1.5*atr);risk=entry-sl;tp1=entry+1.5*risk;tp2=entry+2.5*risk
        else:sl=max(entry+.8*atr,hi if hi>entry else entry+1.5*atr);risk=sl-entry;tp1=entry-1.5*risk;tp2=entry-2.5*risk
        return entry,sl,tp1,tp2,abs(tp1-entry)/risk,abs(tp2-entry)/risk

    def generate(self,row,mtf_trends,structure_result=None,structure_events=None,liquidity_result=None,trendline_result=None,fibonacci_result=None,candle_result=None,df=None):
        funcs=[("MTF",self.score_mtf(mtf_trends)),("MARKET_STRUCTURE",self.score_structure(structure_result)),("LIQUIDITY",self.score_liquidity(liquidity_result)),("FIBONACCI",self.score_fibonacci(fibonacci_result)),("TRENDLINE",self.score_trendline(trendline_result)),("CANDLE",self.score_candle(candle_result))]
        comp={k:v[1] for k,v in funcs};dirs={k:v[0] for k,v in funcs};reasons=[z for _,v in funcs for z in v[2]];buy=sum(comp[k] for k in comp if dirs[k]=="BUY");sell=sum(comp[k] for k in comp if dirs[k]=="SELL");d="BUY" if buy>sell else "SELL" if sell>buy else "NEUTRAL";score=max(buy,sell);warnings=[];agree=sum(1 for k in comp if dirs[k]==d and comp[k]>0)
        if d=="NEUTRAL":status="NO_SIGNAL";score=0
        else:
            if agree<2:score=min(score,49);warnings.append("Kamida 2 komponent bir yo'nalishda emas.")
            if min(buy,sell)>0 and min(buy,sell)/score>=.5:score=min(score,59);warnings.append("Komponentlar konflikti kuchli.")
            status="HIGH_CONFIDENCE" if score>=85 else "CONFIRMED" if score>=75 else "EARLY" if score>=60 else "WATCH"
        levels=self.calculate_levels(d,df,structure_result) if d!="NEUTRAL" else (None,)*6
        return SignalResult(d,round(score,2),status,*levels,reasons,warnings,{k:round(v,2) for k,v in comp.items()})

    def analyze(self,mtf_data=None,mtf_trends=None,structure_results=None,trendline_results=None,liquidity_results=None,fibonacci_results=None,candle_results=None,df=None,**kwargs):
        m5=df
        if m5 is None and isinstance(mtf_data,dict):m5=mtf_data.get("M5") or mtf_data.get("m5")
        sr=structure_results.get("M5") if isinstance(structure_results,dict) else structure_results
        return self.generate(m5.iloc[-2] if m5 is not None and len(m5)>=2 else None,mtf_trends or {},sr,liquidity_result=liquidity_results,trendline_result=trendline_results,fibonacci_result=fibonacci_results,candle_result=candle_results,df=m5)
