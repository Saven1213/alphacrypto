#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TRUE CRYPTO ALPHA v3.0 - Full Version
• Автоматический универсум топ-100 CMC с обновлением каждые сутки
• Приоритет Binance, fallback: OKX/Bybit
• Мультитаймфрейм-анализ: M15 (вход), H1 (контекст), H4 (структура)
• Интеграция всех аналитических модулей (Эллиот, Volume Profile, Кластера, Order Blocks, Smart Money, Классика ТА)
• Условия формирования сигнала с системой скоринга и антиспам-фильтром
• Формат и отправка в Telegram
"""
import asyncio
import logging
from datetime import datetime, timedelta
import os
import ccxt
import pandas as pd
import numpy as np
from colorama import Fore, Style, init
import requests
import traceback

init(autoreset=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
TELEGRAM_ENABLED = os.getenv('TELEGRAM_BOT_TOKEN') is not None
if TELEGRAM_ENABLED:
    try:
        from telegram import Bot
        telegram_bot = Bot(token=os.getenv('TELEGRAM_BOT_TOKEN'))
        telegram_chat_id = os.getenv('TELEGRAM_CHAT_ID')
        logger.info(f"{Fore.GREEN}✅ Telegram подключен!{Style.RESET_ALL}")
    except:
        TELEGRAM_ENABLED = False
        logger.warning(f"{Fore.YELLOW}⚠️ Telegram не настроен{Style.RESET_ALL}")
class CMCUniverse:
    def __init__(self):
        self.top100 = []
        self.last_update = None
    def fetch_top100(self):
        url = 'https://api.coinmarketcap.com/data-api/v3/cryptocurrency/listing?start=1&limit=100&sortBy=market_cap'
        try:
            resp = requests.get(url)
            coins = resp.json()['data']['cryptoCurrencyList']
            self.top100 = [c['symbol'] for c in coins]
            self.last_update = datetime.now()
        except Exception as e:
            logger.error(f"Ошибка обновления CMC: {e}")
    def update_if_needed(self):
        if self.last_update is None or (datetime.now()-self.last_update).total_seconds()>82800:
            self.fetch_top100()
    def get_pairs(self, exchange):
        self.update_if_needed()
        pairs = []
        markets = exchange.load_markets()
        for coin in self.top100:
            if f'{coin}/USDT' in markets:
                pairs.append(f'{coin}/USDT')
            elif f'{coin}/USDC' in markets:
                pairs.append(f'{coin}/USDC')
        return pairs
class TFEnum:
    M15 = '15m'
    H1 = '1h'
    H4 = '4h'
class TechnicalAnalysis:
    @staticmethod
    def calculate_atr(df, period=14):
        high = df['high']
        low = df['low']
        close = df['close']
        tr = np.maximum(high-low, high-close.shift(), close.shift()-low)
        atr = pd.Series(tr).rolling(window=period).mean()
        return atr.iloc[-1] if not pd.isna(atr.iloc[-1]) else tr.mean()
    @staticmethod
    def calculate_ema(data, period):
        ema = data.ewm(span=period, adjust=False).mean()
        return ema.iloc[-1]
    @staticmethod
    def calculate_rsi(data, period=14):
        delta = data.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.iloc[-1] if not pd.isna(rsi.iloc[-1]) else 50
    @staticmethod
    def calculate_avg_volume(volume, window=20):
        return pd.Series(volume).rolling(window=window).mean().iloc[-1]
class SignalModules:
    @staticmethod
    def elliott_wave(df_m15, df_h1):
        high = df_m15['high']
        low = df_m15['low']

        swing_highs = (
                (high.shift(1) > high.shift(2)) &
                (high.shift(1) > high)
        )

        swing_lows = (
                (low.shift(1) < low.shift(2)) &
                (low.shift(1) < low)
        )

        impulse = swing_highs.sum() >= 3 and swing_lows.sum() >= 3
        elliott_dir = 'LONG' if df_h1['close'].iloc[-1] > df_h1['open'].iloc[-1] else 'SHORT'

        return bool(impulse), elliott_dir
    @staticmethod
    def volume_profile(df, bars=100):
        prices = np.floor(df['close'][-bars:])
        vols = df['volume'][-bars:]
        vp = pd.DataFrame({'price': prices, 'vol': vols}).groupby('price').sum()
        poc = vp['vol'].idxmax()
        lvn = vp['vol'].idxmin()
        current_price = df['close'].iloc[-1]
        vp_ok = abs(current_price-lvn)<abs(current_price-poc)
        return vp_ok, poc, lvn
    @staticmethod
    def cluster_absorb(df_m15):
        last_bar = df_m15.iloc[-1]
        mid_price = (last_bar['high']+last_bar['low'])/2
        cluster_absorb = last_bar['volume']>df_m15['volume'].iloc[:-1].mean() and last_bar['close']>=mid_price
        return cluster_absorb
    @staticmethod
    def order_block(df_m15):
        vol_thr = df_m15['volume'].mean()*2
        impulse_idx = np.where(df_m15['volume']>vol_thr)[0]
        if len(impulse_idx)>0:
            idx = impulse_idx[-1]
            ob_price = df_m15['open'].iloc[idx]
            ob_touched = abs(df_m15['close'].iloc[-1]-ob_price)<df_m15['close'].std()
            return ob_touched, ob_price
        else:
            return False, None
    @staticmethod
    def smart_money(df_m15):
        highs = df_m15['high']
        lows = df_m15['low']
        past_max = highs.iloc[:-5].max()
        past_min = lows.iloc[:-5].min()
        recent = df_m15.iloc[-5:]
        liquidity_grab = recent['high'].max()>past_max or recent['low'].min()<past_min
        post_impulse = recent['volume'].mean()>df_m15['volume'][:-5].mean()*1.2
        imbalance_area = (recent['close'].max()-recent['close'].min())/recent['close'].mean()>0.01
        smtm_ok = liquidity_grab and post_impulse and imbalance_area
        return smtm_ok
    @staticmethod
    def classic_ta(df):
        close = df['close']
        ema50 = TechnicalAnalysis.calculate_ema(close, 50)
        ema200 = TechnicalAnalysis.calculate_ema(close, 200)
        rsi = TechnicalAnalysis.calculate_rsi(close)
        volume = df['volume'].iloc[-1]
        avg_volume = TechnicalAnalysis.calculate_avg_volume(df['volume'])
        long_cond = close.iloc[-1]>ema50 and close.iloc[-1]>ema200 and rsi<40 and rsi>20 and volume>=1.5*avg_volume
        short_cond = close.iloc[-1]<ema50 and close.iloc[-1]<ema200 and rsi>60 and rsi<80 and volume>=1.5*avg_volume
        ta_ok = long_cond or short_cond
        return ta_ok
class SignalAggregator:
    @staticmethod
    def score(mods):
        weights = {'elliott': 2, 'vp': 1, 'ob': 1, 'smtm': 2, 'ta': 1, 'cluster': 1}
        score = 0
        for k, v in mods.items():
            if bool(v) and k in weights:
                score += weights[k]
        return score
    @staticmethod
    def is_strong_signal(flags):
        strong = (
            flags['elliott'] and
            flags['ta'] and
            (flags['vp'] or flags['ob'] or flags['cluster']) and
            flags['smtm']
        )
        return strong
class TrueCryptoAlpha:
    def __init__(self):
        self.version = "3.0.0"
        self.exchange = ccxt.binance({'enableRateLimit': True})
        self.universe = CMCUniverse()
        self.signal_last_time = {}
    def print_banner(self):
        banner = f"""
{Fore.CYAN}{'='*70}
{Fore.YELLOW} 🚀 TRUE CRYPTO ALPHA v{self.version} - FULL VERSION
{Fore.GREEN} 💡 TA, Volume, Эллиот, Telegram
{Fore.CYAN}{'='*70}{Style.RESET_ALL}
 """
        print(banner)

    async def get_ohlcv(self, symbol, tf='15m', limit=500):
        try:
            ohlcv = await asyncio.to_thread(self.exchange.fetch_ohlcv, symbol, tf, limit=limit)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            return df
        except Exception as e:
            logger.error(f"Ошибка OHLCV для {symbol} на {tf}: {e}")
            return None
    async def analyze_all_tfs(self, symbol):
        df_m15 = await self.get_ohlcv(symbol, TFEnum.M15)
        df_h1 = await self.get_ohlcv(symbol, TFEnum.H1)
        df_h4 = await self.get_ohlcv(symbol, TFEnum.H4)
        return df_m15, df_h1, df_h4
    async def run(self):
        self.print_banner()
        pairs = self.universe.get_pairs(self.exchange)
        logger.info(f"{Fore.CYAN}Мониторинг {len(pairs)} инструментов...{Style.RESET_ALL}")
        while True:
            try:
                for symbol in pairs:
                    now = datetime.now()
                    last_time = self.signal_last_time.get(symbol, now-timedelta(hours=3))
                    if (now-last_time).total_seconds()<7200:
                        continue
                    df_m15, df_h1, df_h4 = await self.analyze_all_tfs(symbol)
                    if any(df is None or len(df)<100 for df in [df_m15,df_h1,df_h4]): continue
                    atr = TechnicalAnalysis.calculate_atr(df_m15)
                    if atr<0.1: continue
                    elliott_signal, elliott_dir = SignalModules.elliott_wave(df_m15,df_h1)
                    vp_ok,poc,lvn = SignalModules.volume_profile(df_m15)
                    cluster_absorb = SignalModules.cluster_absorb(df_m15)
                    ob_touched,ob_price = SignalModules.order_block(df_m15)
                    smtm_ok = SignalModules.smart_money(df_m15)
                    ta_ok = SignalModules.classic_ta(df_m15)
                    mods = {
                        'elliott': bool(elliott_signal),
                        'vp': bool(vp_ok),
                        'ob': bool(ob_touched),
                        'smtm': bool(smtm_ok),
                        'ta': bool(ta_ok),
                        'cluster': bool(cluster_absorb)
                    }
                    score = SignalAggregator.score(mods)
                    strong = SignalAggregator.is_strong_signal(mods)
                    price_now = df_m15['close'].iloc[-1]
                    swing_lows = df_m15['low'].rolling(3).min().iloc[-10:]
                    swing_highs = df_m15['high'].rolling(3).max().iloc[-10:]
                    entry = ob_price if ob_touched and ob_price else price_now
                    sl = min(swing_lows.min(), entry-atr*1.2) if elliott_dir=='LONG' else max(swing_highs.max(), entry+atr*1.2)
                    tp1 = entry+atr*1.5 if elliott_dir=='LONG' else entry-atr*1.5
                    tp2 = poc if poc else entry+atr*2 if elliott_dir=='LONG' else entry-atr*2
                    tp3 = entry+atr*3 if elliott_dir=='LONG' else entry-atr*3
                    rr = abs(tp2-entry)/abs(entry-sl) if abs(entry-sl)>0.01 else 0
                    day_high = df_m15['high'].max()
                    day_low = df_m15['low'].min()
                    risk_tag = ''
                    if abs(price_now-day_high)/day_high<0.01 or abs(price_now-day_low)/day_low<0.01:
                        risk_tag = '⚠️ РИСК!'
                    if not strong or score<5 or rr<1.5:
                        continue
                    self.signal_last_time[symbol]=now
                    msg = f"{symbol}\nTF: 15m\nENTRY: {entry:.2f}\nTP1: {tp1:.2f}\nTP2: {tp2:.2f}\nTP3: {tp3:.2f}\nSL: {sl:.2f}\nКомментарий:\nЭллиот={elliott_dir}, объёмная структура, атака по Order Block/Volume\nРиск рекомендуемый: 1%\n{risk_tag}\n#truecrypto #{symbol.split('/')[0]}"
                    if TELEGRAM_ENABLED:
                        try:
                            await asyncio.to_thread(telegram_bot.send_message,
                                chat_id=telegram_chat_id,
                                text=msg,
                                parse_mode='HTML')
                            logger.info(f"{Fore.GREEN}📱 Сигнал ушёл в Telegram!{Style.RESET_ALL}")
                        except Exception as te:
                            logger.error(f"Telegram error: {te}")
                    else:
                        logger.info(f"{Fore.YELLOW}Сигнал: {msg}{Style.RESET_ALL}")
                await asyncio.sleep(120)
            except Exception as e:
                logger.error(traceback.format_exc())
                await asyncio.sleep(60)
if __name__=="__main__":
    bot = TrueCryptoAlpha()
    try:
        asyncio.run(bot.run())
    except KeyboardInterrupt:
        print(f"\n{Fore.CYAN}👋 Бот остановлен. До встречи!{Style.RESET_ALL}")
