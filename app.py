import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.trend import MACD
import twstock
import warnings
import requests

# 🛡️ 終極偽裝術 (給 Yahoo 用的)
session = requests.Session()
session.headers.update({
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
})

warnings.filterwarnings('ignore') 
st.set_page_config(page_title="帥順股市分析與資產管理神器", layout="wide")

# ==========================================
# 🔑 系統機密與 API 設定 (富果 Fugle API)
# ==========================================
# 系統會優先讀取 Streamlit Secrets，如果沒有，就用你提供的這組預設值
FUGLE_API_KEY = st.secrets.get("FUGLE_API_KEY", "YWYyMmIyOTQtNzViZi00YzBjLTk3YjUtYTE0YjQ2MTNiNGUwIGNkYzQ5MWI0LTdkNGYtNGMwOC04OTJhLTBmOTJhMmUxZTFhYw==")

# 專屬富果即時報價抓取函式
def fetch_fugle_price(symbol_code, api_key):
    if not api_key: return None, None
    pure_code = symbol_code.split('.')[0]
    url = f"https://api.fugle.tw/marketdata/v1.0/stock/intraday/quote/{pure_code}"
    try:
        # X-API-KEY 是富果官方要求的通行證格式
        res = requests.get(url, headers={"X-API-KEY": api_key}, timeout=5)
        if res.status_code == 200:
            data = res.json()
            curr = data.get("lastPrice") or data.get("closePrice")
            if not curr and data.get("lastTrade"):
                curr = data.get("lastTrade").get("price")
            prev = data.get("referencePrice") # 昨日收盤價
            
            if curr and prev:
                return float(curr), float(prev)
    except Exception:
        pass
    return None, None

# ==========================================
# 🎨 專屬介面優化：自適應表格寬度
# ==========================================
st.markdown("""
<style>
[data-testid="stTable"] table { width: max-content !important; }
[data-testid="stTable"] { display: flex; justify-content: flex-start; }
</style>
""", unsafe_allow_html=True)

def color_tw_col(s):
    return ['color: #FF4B4B' if isinstance(v, (int, float)) and v > 0 
            else 'color: #00D26A' if isinstance(v, (int, float)) and v < 0 
            else '' for v in s]

def fmt_pct(val):
    if val is None or pd.isna(val): return "無資料"
    return f"{val * 100:.2f}%"

def fmt_val(val):
    if val is None or pd.isna(val): return "無資料"
    return f"{val:.2f}"

st.title("🚀 帥順股市分析與資產管理神器")

SHEET_URL = "https://docs.google.com/spreadsheets/d/e/2PACX-1vSQ4j2F1BSeWfRyA748KJh4hkU3KB26odS4uTfP7AZQgNcR0zvQVvjjYOfIvku-5vi8FcyW2BxNBDtq/pub?output=csv"

@st.cache_data(ttl=60)
def load_portfolio(url):
    try:
        df = pd.read_csv(url)
        portfolio = [] 
        for index, row in df.iterrows():
            if pd.notna(row['代號']):
                symbol = str(row['代號']).strip()
                pure_code = symbol.split('.')[0]
                
                if pure_code in twstock.codes:
                    stock_name = twstock.codes[pure_code].name
                else:
                    stock_name = str(row['股票名稱']).strip() if '股票名稱' in df.columns and pd.notna(row['股票名稱']) else "未知"
                
                category = str(row['分類']).strip() if '分類' in df.columns and pd.notna(row['分類']) else "本人"
                    
                portfolio.append({
                    'symbol': symbol,
                    'cost': float(row['成本']), 
                    'shares': int(row['股數']),
                    'name': stock_name,
                    'category': category
                })
        return portfolio
    except Exception as e:
        st.error("讀取試算表失敗，請確認網址是否正確且已設定為 CSV 發布。")
        return []

MY_PORTFOLIO = load_portfolio(SHEET_URL)

tab1, tab2, tab3 = st.tabs(["📈 個股技術分析", "💰 我的投資組合", "🌍 台股主力 800 飆股雷達與觀測站"])

# ----------------------------------------
# 分頁 1：個股技術分析與基本面 (保留 Yahoo 提供歷史 10 年 K 線)
# ----------------------------------------
with tab1:
    unique_symbols = list(set([p['symbol'] for p in MY_PORTFOLIO]))
    symbol_name_map = {p['symbol']: p['name'] for p in MY_PORTFOLIO}

    def display_stock(symbol):
        if symbol in symbol_name_map and symbol_name_map[symbol]:
            return f"{symbol} ({symbol_name_map[symbol]})"
        return symbol

    col_search, col_space = st.columns([1, 2])
    with col_search:
        stock_options = unique_symbols + ["手動輸入其他代號..."]
        selected_option = st.selectbox("請選擇要分析的自選股 (或選擇手動輸入)", stock_options, format_func=display_stock)

        if selected_option == "手動輸入其他代號...":
            ticker_symbol = st.text_input("請輸入股票代號 (台股請加 .TW 或 .TWO)", "2330.TW")
            pure_code = ticker_symbol.split('.')[0]
            if pure_code in twstock.codes:
                display_name = f"{ticker_symbol} ({twstock.codes[pure_code].name})"
            else:
                display_name = ticker_symbol
        else:
            ticker_symbol = selected_option
            display_name = display_stock(ticker_symbol)

    st.markdown("---")
    
    if ticker_symbol:
        info = {}
        df_raw = pd.DataFrame()
        
        try:
            ticker_data = yf.Ticker(ticker_symbol, session=session) 
            try:
                info = ticker_data.info
            except Exception:
                st.warning("⚠️ Yahoo 財經目前限制基本面讀取，請專注於下方技術線圖。")
            try:
                df_raw = ticker_data.history(period="10y")
            except Exception:
                st.error("⚠️ 無法取得 K 線報價資料，請稍後再試。")
        except Exception:
            st.error("⚠️ Yahoo 財經伺服器阻擋了此股票的連線要求，請稍後再重整頁面。")
        
        st.subheader(f"🏢 **{display_name}** - 基本面與財務指標 (最新季報)")
            
        col_f1, col_f2, col_f3 = st.columns(3)
        with col_f1:
            st.markdown("##### 💰 獲利能力 (Profitability)")
            st.metric("毛利率 (Gross Margin)", fmt_pct(info.get('grossMargins')))
            st.metric("營業利益率 (Operating Margin)", fmt_pct(info.get('operatingMargins')))
            st.metric("稅後純益率 (Net Margin)", fmt_pct(info.get('profitMargins')))
            st.metric("股東權益報酬率 (ROE)", fmt_pct(info.get('returnOnEquity')))
            st.metric("資產報酬率 (ROA)", fmt_pct(info.get('returnOnAssets')))
            st.metric("每股稅後盈餘 (EPS)", fmt_val(info.get('trailingEps')))
            
        with col_f2:
            st.markdown("##### 🚀 成長性 (Growth - YoY)")
            st.metric("營收成長率 (季對季YoY)", fmt_pct(info.get('revenueGrowth')))
            st.metric("稅後淨利成長率 (季對季YoY)", fmt_pct(info.get('earningsGrowth')))
            
        with col_f3:
            st.markdown("##### ⚖️ 估值與其他")
            st.metric("本益比 (P/E Ratio)", fmt_val(info.get('trailingPE')))
            st.metric("股價淨值比 (P/B Ratio)", fmt_val(info.get('priceToBook')))
            st.metric("現金殖利率 (Dividend Yield)", fmt_pct(info.get('dividendYield')))
            st.metric("市值 (Market Cap)", f"{info.get('marketCap', 0) / 100000000:.2f} 億" if info.get('marketCap') else "無資料")

        st.divider()
        
        st.subheader(f"📊 **{display_name}** - 專業技術線圖")
        
        col_ctrl1, col_ctrl2, col_ctrl3 = st.columns(3)
        with col_ctrl1:
            tf_option = st.radio("⏳ K線週期", ["日線", "週線", "月線", "年線"], horizontal=True)
            show_pe_river = st.checkbox("🌊 疊加本益比河流圖", value=False)
        with col_ctrl2:
            ma_options = ["5", "10", "20", "30", "60", "120", "240"]
            selected_mas = st.multiselect("📈 顯示均線 (可複選)", ma_options, default=["5", "20", "60"])
            show_cross = st.checkbox("✨ 自動偵測 5日/20日 交叉訊號", value=True) 
        with col_ctrl3:
            ind_options = ["成交量", "KD", "MACD", "RSI"]
            selected_inds = st.multiselect("📉 附圖指標 (可複選)", ind_options, default=["成交量", "KD", "MACD"])
        
        if not df_raw.empty:
            df_raw.index = df_raw.index.tz_localize(None)
            
            if tf_option == "日線": df = df_raw.copy()
            elif tf_option == "週線": df = df_raw.resample('W-FRI').agg({'Open':'first', 'High':'max', 'Low':'min', 'Close':'last', 'Volume':'sum'}).dropna()
            elif tf_option == "月線": df = df_raw.resample('ME').agg({'Open':'first', 'High':'max', 'Low':'min', 'Close':'last', 'Volume':'sum'}).dropna()
            elif tf_option == "年線": df = df_raw.resample('YE').agg({'Open':'first', 'High':'max', 'Low':'min', 'Close':'last', 'Volume':'sum'}).dropna()

            ma_colors = ['#FFA500', '#FF1493', '#00BFFF', '#9932CC', '#32CD32', '#FF0000', '#0000FF']
            ma_lines = {}
            for i, ma_str in enumerate(selected_mas):
                ma_val = int(ma_str)
                df[f'MA{ma_val}'] = df['Close'].rolling(window=ma_val).mean()
                ma_lines[f'MA{ma_val}'] = ma_colors[i % len(ma_colors)]

            if show_cross:
                if 'MA5' not in df.columns: df['MA5'] = df['Close'].rolling(window=5).mean()
                if 'MA20' not in df.columns: df['MA20'] = df['Close'].rolling(window=20).mean()
                df['Golden_Cross'] = (df['MA5'] > df['MA20']) & (df['MA5'].shift(1) <= df['MA20'].shift(1))
                df['Death_Cross'] = (df['MA5'] < df['MA20']) & (df['MA5'].shift(1) >= df['MA20'].shift(1))

            if "KD" in selected_inds:
                kd = StochasticOscillator(high=df['High'], low=df['Low'], close=df['Close'], window=9, smooth_window=3)
                df['K'] = kd.stoch()
                df['D'] = kd.stoch_signal()
            if "MACD" in selected_inds:
                macd = MACD(close=df['Close'], window_slow=26, window_fast=12, window_sign=9)
                df['MACD'] = macd.macd()
                df['MACD_signal'] = macd.macd_signal()
                df['MACD_hist'] = macd.macd_diff()
            if "RSI" in selected_inds:
                rsi = RSIIndicator(close=df['Close'], window=14)
                df['RSI'] = rsi.rsi()

            display_bars = 150 if tf_option != "年線" else len(df)
            df_plot = df.tail(display_bars)
            
            rows = 1 + len(selected_inds)
            if rows == 1: row_heights = [1.0]
            else: row_heights = [0.5] + [0.5 / len(selected_inds)] * len(selected_inds)
                
            fig = make_subplots(rows=rows, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=row_heights)
            
            fig.add_trace(go.Candlestick(x=df_plot.index, open=df_plot['Open'], high=df_plot['High'], low=df_plot['Low'], close=df_plot['Close'], increasing_line_color='#FF4B4B', decreasing_line_color='#00D26A', name='K線'), row=1, col=1)
            fig.update_yaxes(rangemode='nonnegative', row=1, col=1)
            
            for ma_col, color in ma_lines.items():
                fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot[ma_col], line=dict(color=color, width=1.5), name=ma_col), row=1, col=1)

            if show_cross:
                golden_mask = df_plot['Golden_Cross'] == True
                if golden_mask.any(): fig.add_trace(go.Scatter(x=df_plot[golden_mask].index, y=df_plot[golden_mask]['Low'] * 0.98, mode='markers', marker=dict(symbol='triangle-up', size=14, color='#FF4B4B', line=dict(width=1, color='white')), name='黃金交叉 (5上穿20)'), row=1, col=1)
                death_mask = df_plot['Death_Cross'] == True
                if death_mask.any(): fig.add_trace(go.Scatter(x=df_plot[death_mask].index, y=df_plot[death_mask]['High'] * 1.02, mode='markers', marker=dict(symbol='triangle-down', size=14, color='#00D26A', line=dict(width=1, color='white')), name='死亡交叉 (5下穿20)'), row=1, col=1)

            if show_pe_river:
                try:
                    eps = info.get('trailingEps', 0)
                    if eps and eps > 0:
                        pe_ratios = [10, 12, 15, 18, 20, 25]
                        river_colors = ['#c6dbef', '#9ecae1', '#6baed6', '#4292c6', '#2171b5', '#084594']
                        for pe, color in zip(pe_ratios, river_colors):
                            fig.add_trace(go.Scatter(x=df_plot.index, y=[eps * pe]*len(df_plot), name=f"{pe}X 本益比", line=dict(color=color, dash='dot', width=1.5)), row=1, col=1)
                except: pass

            current_row = 2
            for ind in selected_inds:
                if ind == "成交量":
                    vol_colors = ['#FF4B4B' if row['Close'] >= row['Open'] else '#00D26A' for i, row in df_plot.iterrows()]
                    fig.add_trace(go.Bar(x=df_plot.index, y=df_plot['Volume'], marker_color=vol_colors, name='成交量'), row=current_row, col=1)
                    fig.update_yaxes(rangemode='nonnegative', row=current_row, col=1)
                elif ind == "KD":
                    fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['K'], name='K值', line=dict(color='#00BFFF')), row=current_row, col=1)
                    fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['D'], name='D值', line=dict(color='#FFA500')), row=current_row, col=1)
                    fig.update_yaxes(range=[0, 100], row=current_row, col=1)
                elif ind == "MACD":
                    macd_colors = ['#FF4B4B' if v > 0 else '#00D26A' for v in df_plot['MACD_hist']]
                    fig.add_trace(go.Bar(x=df_plot.index, y=df_plot['MACD_hist'], marker_color=macd_colors, name='OSC'), row=current_row, col=1)
                    fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['MACD'], name='DIF', line=dict(color='#00BFFF')), row=current_row, col=1)
                    fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['MACD_signal'], name='MACD', line=dict(color='#FFA500')), row=current_row, col=1)
                elif ind == "RSI":
                    fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['RSI'], name='RSI', line=dict(color='#9932CC')), row=current_row, col=1)
                    fig.add_trace(go.Scatter(x=df_plot.index, y=[70]*len(df_plot), line=dict(color='#FF4B4B', dash='dash'), showlegend=False), row=current_row, col=1)
                    fig.add_trace(go.Scatter(x=df_plot.index, y=[30]*len(df_plot), line=dict(color='#00D26A', dash='dash'), showlegend=False), row=current_row, col=1)
                    fig.update_yaxes(range=[0, 100], row=current_row, col=1)
                current_row += 1
                
            fig.update_layout(xaxis_rangeslider_visible=False, height=400 + 150 * len(selected_inds), margin=dict(l=10, r=10, t=80, b=10), legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0.01), dragmode='pan')
            st.plotly_chart(fig, use_container_width=True, config={'scrollZoom': True, 'displayModeBar': False})

# ----------------------------------------
# 🌟 分頁 2：我的投資組合 (已升級為富果 API 零時差引擎)
# ----------------------------------------
with tab2:
    st.info("⚡ 本頁面已全面切換至『富果 Fugle 零時差即時 API』，享受盤中無延遲看盤體驗！")
    
    if MY_PORTFOLIO:
        portfolio_data = []
        my_bar = st.progress(0, text="⚡ 正在透過富果光纖 API 抓取您的持股即時報價...")
        for i, info in enumerate(MY_PORTFOLIO):
            symbol = info['symbol']
            cost = info['cost']
            shares = info['shares']
            stock_name = info['name']
            category = info['category']
            
            current_price = None
            prev_price = None
            
            # 🚀 第一優先：使用富果 API (零時差)
            f_curr, f_prev = fetch_fugle_price(symbol, FUGLE_API_KEY)
            if f_curr and f_prev:
                current_price = f_curr
                prev_price = f_prev
            else:
                # 🛡️ 第二道防線：如果富果呼叫失敗 (或超過 60次/分 限制)，自動切換回 Yahoo 備用引擎
                try:
                    tick = yf.Ticker(symbol, session=session)
                    hist = tick.history(period="5d")
                    if not hist.empty:
                        current_price = float(hist['Close'].iloc[-1])
                        if len(hist) >= 2: prev_price = float(hist['Close'].iloc[-2])
                        else: prev_price = current_price
                except Exception:
                    pass
            
            if current_price is not None and prev_price is not None:
                daily_price_diff = current_price - prev_price
                daily_pct_diff = (daily_price_diff / prev_price) * 100 if prev_price > 0 else 0
                daily_profit_diff = daily_price_diff * shares
                
                stock_cost_raw = cost * shares
                stock_value_raw = current_price * shares
                discount = 0.6
                buy_fee = max(20, stock_cost_raw * 0.001425 * discount)
                sell_fee = max(20, stock_value_raw * 0.001425 * discount)
                
                if symbol.startswith("00"): tax = stock_value_raw * 0.001; type_label = "ETF"
                else: tax = stock_value_raw * 0.003; type_label = "個股"
                
                true_stock_cost = stock_cost_raw + buy_fee
                true_profit = stock_value_raw - stock_cost_raw - buy_fee - sell_fee - tax
                roi = (true_profit / true_stock_cost) * 100 if true_stock_cost > 0 else 0
                
                portfolio_data.append({
                    "category": category, "股票名稱": stock_name, "股票代號": f"{symbol} ({type_label})",
                    "持股數": shares, "平均成本": cost, "最新股價": round(current_price, 2),
                    "今日漲跌 (%)": round(daily_pct_diff, 2), "今日獲利增減": round(daily_profit_diff, 0),
                    "總成本": true_stock_cost, "目前市值": round(stock_value_raw, 2),
                    "淨損益": round(true_profit, 0), "報酬率 (%)": round(roi, 1) 
                })
            my_bar.progress((i + 1) / len(MY_PORTFOLIO), text="⚡ 正在透過富果光纖 API 抓取您的持股即時報價...")
        my_bar.empty()
        
        grouped_data = {}
        for p in portfolio_data:
            cat = p["category"]
            if cat not in grouped_data: grouped_data[cat] = []
            grouped_data[cat].append(p)
            
        def sort_key(cat): return 0 if cat in ["本人", "帥順"] else 1
        sorted_categories = sorted(grouped_data.keys(), key=sort_key)
        
        for cat in sorted_categories:
            cat_records = grouped_data[cat]
            cat_total_cost = sum([p["總成本"] for p in cat_records])
            cat_total_value = sum([p["目前市值"] for p in cat_records])
            cat_total_profit = sum([p["淨損益"] for p in cat_records])
            cat_total_roi = (cat_total_profit / cat_total_cost) * 100 if cat_total_cost > 0 else 0
            cat_daily_profit_total = sum([p["今日獲利增減"] for p in cat_records])
            
            st.markdown(f"### 👤 【{cat}】的專屬資產")
            col1, col2, col3 = st.columns(3)
            col1.metric("總成本 (含手續費)", f"${cat_total_cost:,.0f}")
            col2.metric("目前總市值", f"${cat_total_value:,.0f}", f"{cat_daily_profit_total:+,.0f}", delta_color="inverse")
            col3.metric("總未實現淨利", f"${cat_total_profit:,.0f}", f"{cat_total_roi:.1f}%", delta_color="inverse")
            
            display_list = []
            for p in cat_records:
                display_item = p.copy()
                del display_item["category"]
                display_list.append(display_item)
                
            df_portfolio = pd.DataFrame(display_list)
            if not df_portfolio.empty:
                df_portfolio.index = df_portfolio.index + 1
                styled_table = df_portfolio.style.apply(color_tw_col, subset=["淨損益", "報酬率 (%)", "今日漲跌 (%)", "今日獲利增減"]).format({
                    "持股數": "{:,.0f}", "平均成本": "{:.2f}", "最新股價": "{:.2f}", "今日漲跌 (%)": "{:.2f}",
                    "今日獲利增減": "${:,.0f}", "總成本": "${:,.0f}", "目前市值": "${:,.0f}",
                    "淨損益": "${:,.0f}", "報酬率 (%)": "{:.1f}"  
                })
                st.table(styled_table)
            
            st.divider() 
    else:
        st.info("尚未從試算表讀取到持股資料。請確認您的試算表 A、B、C 欄有正確輸入內容。")

# ----------------------------------------
# 🌟 分頁 3：台股主力 800 飆股雷達與觀測站 (繼續使用 Yahoo 批次引擎)
# ----------------------------------------
with tab3:
    st.subheader("🌍 台股主力 800 飆股雷達與產業觀測站")
    st.warning("⏱️ 溫馨提示：為突破 API 限制，此頁面由 Yahoo 批次引擎提供火力支援（可能有 15 分鐘延遲），每 30 分鐘自動重整，掃描台股最容易出飆股的 12 大熱門板塊！")
    
    user_etf_dict = {}
    for p in MY_PORTFOLIO:
        if str(p['symbol']).startswith("00"):
            user_etf_dict[p['symbol']] = p['name']
    user_etfs = tuple(user_etf_dict.items())
    
    @st.cache_data(ttl=1800) 
    def get_smart_market_data(etf_tuple):
        try:
            market_tickers = {
                "^TWII": ("加權指數", "大盤"), "^TWOII": ("櫃買指數", "大盤")
            }
            
            classic_etfs = ["0050.TW", "0056.TW", "00878.TW", "00881.TW", "0055.TW", "00929.TW", "00919.TW", "00713.TW"]
            for c_etf in classic_etfs:
                name = twstock.codes[c_etf.replace(".TW", "")].name if c_etf.replace(".TW", "") in twstock.codes else c_etf
                market_tickers[c_etf] = (name, "ETF")

            hot_keywords = ['半導體', '電腦', '電子零組件', '其他電子', '光電', '通信', '電機', '生技', '航運', '綠能', '雲端', '化學', '鋼鐵']
            
            for code, info in twstock.codes.items():
                if len(code) == 4 and info.type == '股票':
                    if any(k in info.group for k in hot_keywords):
                        sym = f"{code}.TW" if info.market == '上市' else f"{code}.TWO"
                        market_tickers[sym] = (info.name, info.group)
                        
            heavyweights = {
                "2881.TW": ("富邦金", "金融保險"), "2882.TW": ("國泰金", "金融保險"), "2891.TW": ("中信金", "金融保險"),
                "1301.TW": ("台塑", "塑膠工業"), "1303.TW": ("南亞", "塑膠工業"), "1216.TW": ("統一", "食品工業"),
                "2886.TW": ("兆豐金", "金融保險"), "2884.TW": ("玉山金", "金融保險"), "2892.TW": ("第一金", "金融保險"),
                "2207.TW": ("和泰車", "汽車工業"), "1101.TW": ("台泥", "水泥工業"), "9904.TW": ("寶成", "其他業")
            }
            market_tickers.update(heavyweights)

            for sym, name in etf_tuple:
                if sym not in market_tickers:
                    market_tickers[sym] = (f"{name} (我的)", "ETF")
                elif "(我的)" not in market_tickers[sym][0]:
                    market_tickers[sym] = (f"{market_tickers[sym][0]} (我的)", "ETF")

            target_symbols = list(market_tickers.keys())
            
            try:
                df_dl = yf.download(target_symbols, period="5d", group_by="ticker", threads=True, progress=False, session=session)
            except Exception:
                return pd.DataFrame()
                
            data_list = []
            
            if isinstance(df_dl.columns, pd.MultiIndex):
                available_symbols = df_dl.columns.levels[0]
            else:
                available_symbols = [sym for sym in target_symbols if sym in df_dl.columns]
                
            for sym in target_symbols:
                try:
                    if sym in available_symbols:
                        if isinstance(df_dl.columns, pd.MultiIndex):
                            hist = df_dl[sym].dropna()
                        else:
                            hist = df_dl.dropna()
                            
                        if len(hist) >= 2:
                            curr = float(hist['Close'].iloc[-1])
                            prev = float(hist['Close'].iloc[-2])
                            vol = float(hist['Volume'].iloc[-1])
                            
                            diff = curr - prev
                            pct = (diff / prev) * 100 if prev > 0 else 0
                            
                            data_list.append({
                                "代號": sym.replace(".TW", "").replace(".TWO", ""),
                                "名稱": market_tickers[sym][0],
                                "產業別": market_tickers[sym][1], 
                                "最新報價": round(curr, 2),
                                "漲跌點數": round(diff, 2),
                                "漲跌幅 (%)": round(pct, 2),
                                "成交量 (張)": round(vol / 1000, 0) if not sym.startswith("^") else "大盤總量" 
                            })
                except Exception:
                    pass
                    
            return pd.DataFrame(data_list)
        except Exception:
            return pd.DataFrame()
            
    with st.spinner("📡 系統啟動智能產業濾網，正對台股 800 大活躍飆股進行無死角掃描... (約需 10 秒)"):
        df_market = get_smart_market_data(user_etfs)
    
    if not df_market.empty:
        st.markdown("### 📊 大盤與櫃買指數表現")
        idx_cols = st.columns(2)
        twii_data = df_market[df_market["代號"] == "^TWII"]
        twoii_data = df_market[df_market["代號"] == "^TWOII"]
        
        if not twii_data.empty:
            twii = twii_data.iloc[0]
            idx_cols[0].metric(label="📈 加權指數 (集中市場)", value=f"{twii['最新報價']:,.2f}", delta=f"{twii['漲跌點數']:.2f} ({twii['漲跌幅 (%)']}%)", delta_color="inverse")
        if not twoii_data.empty:
            twoii = twoii_data.iloc[0]
            idx_cols[1].metric(label="📈 櫃買指數 (中小型股)", value=f"{twoii['最新報價']:,.2f}", delta=f"{twoii['漲跌點數']:.2f} ({twii['漲跌幅 (%)']}%)", delta_color="inverse")
            
        st.divider()
        
        df_stocks = df_market[~df_market["代號"].isin(["^TWII", "^TWOII"])].copy()
        
        st.markdown("### 🚀 盤中飆股雷達 (漲幅 > 4% 且具流動性)")
        df_corp = df_stocks[df_stocks["產業別"] != "ETF"].copy()
        
        df_soaring = df_corp[(df_corp["漲跌幅 (%)"] >= 4.0) & (df_corp["成交量 (張)"] != "大盤總量") & (pd.to_numeric(df_corp["成交量 (張)"], errors='coerce') >= 1000)]
        df_soaring = df_soaring.sort_values(by="漲跌幅 (%)", ascending=False).head(30)
        
        if not df_soaring.empty:
            df_soaring.index = range(1, len(df_soaring) + 1)
            st.table(df_soaring[["產業別", "名稱", "最新報價", "漲跌幅 (%)", "成交量 (張)"]].style.apply(color_tw_col, subset=["漲跌幅 (%)"]).format({
                "最新報價": "{:.2f}", "漲跌幅 (%)": "{:.2f}", "成交量 (張)": "{:,.0f}"
            }))
        else:
            st.info("💡 目前雷達掃描範圍內，暫無符合條件之強勢飆股。")

        st.divider()
        
        st.markdown("### 🏢 產業板塊與主題表現 (含專屬持股)")
        df_etf = df_stocks[df_stocks["產業別"] == "ETF"].copy()
        df_etf = df_etf.sort_values(by="漲跌幅 (%)", ascending=False).head(30) 
        df_etf.index = range(1, len(df_etf) + 1)
        st.table(df_etf[["名稱", "最新報價", "漲跌點數", "漲跌幅 (%)", "成交量 (張)"]].style.apply(color_tw_col, subset=["漲跌點數", "漲跌幅 (%)"]).format({
            "最新報價": "{:.2f}", "漲跌點數": "{:.2f}", "漲跌幅 (%)": "{:.2f}", "成交量 (張)": "{:,.0f}"
        }))
        
        st.divider()
        
        st.markdown("### 🔥 活躍資金焦點戰況 (Top 30)")
        
        df_corp["成交量 (張)"] = pd.to_numeric(df_corp["成交量 (張)"], errors='coerce').fillna(0)
        top_vol = df_corp.sort_values(by="成交量 (張)", ascending=False).head(30)
        
        col_c1, col_c2 = st.columns(2)
        with col_c1:
            st.markdown("#### 🎯 焦點資金產業佔比 (依成交量 Top 30)")
            sector_counts = top_vol['產業別'].value_counts().reset_index()
            sector_counts.columns = ['產業別', '檔數']
            
            fig_pie = go.Figure(data=[go.Pie(labels=sector_counts['產業別'], values=sector_counts['檔數'], hole=.4, textinfo='label+percent')])
            fig_pie.update_layout(margin=dict(t=10, b=10, l=10, r=10), height=300, showlegend=True)
            st.plotly_chart(fig_pie, use_container_width=True)
            
        with col_c2:
            st.markdown("#### 📊 各大產業板塊平均漲跌幅 (活躍池統計)")
            sector_perf = df_corp.groupby("產業別")["漲跌幅 (%)"].mean().reset_index()
            sector_perf = sector_perf.sort_values(by="漲跌幅 (%)", ascending=False).head(15)
            
            fig_bar = go.Figure(data=[go.Bar(
                x=sector_perf['產業別'], 
                y=sector_perf['漲跌幅 (%)'],
                marker_color=['#FF4B4B' if val > 0 else '#00D26A' for val in sector_perf['漲跌幅 (%)']],
                text=[f"{val:.2f}%" for val in sector_perf['漲跌幅 (%)']], textposition='outside'
            )])
            fig_bar.update_layout(margin=dict(t=10, b=10, l=10, r=10), height=300, yaxis=dict(title="平均漲跌幅 (%)"))
            st.plotly_chart(fig_bar, use_container_width=True)
            
        st.markdown("---")
        
        col_r1, col_r2 = st.columns(2)
        with col_r1:
            st.markdown("#### 🏆 強勢領漲排行 (排除低量股)")
            valid_gainers = df_corp[df_corp["成交量 (張)"] >= 1000]
            top_gainers = valid_gainers.sort_values(by="漲跌幅 (%)", ascending=False).head(30)
            top_gainers.index = range(1, len(top_gainers) + 1)
            st.table(top_gainers[["產業別", "名稱", "最新報價", "漲跌幅 (%)"]].style.apply(color_tw_col, subset=["漲跌幅 (%)"]).format({"最新報價": "{:.2f}", "漲跌幅 (%)": "{:.2f}"}))
            
        with col_r2:
            st.markdown("#### 📉 弱勢回檔排行 (排除低量股)")
            valid_losers = df_corp[df_corp["成交量 (張)"] >= 1000]
            top_losers = valid_losers.sort_values(by="漲跌幅 (%)", ascending=True).head(30)
            top_losers.index = range(1, len(top_losers) + 1)
            st.table(top_losers[["產業別", "名稱", "最新報價", "漲跌幅 (%)"]].style.apply(color_tw_col, subset=["漲跌幅 (%)"]).format({"最新報價": "{:.2f}", "漲跌幅 (%)": "{:.2f}"}))
            
        st.markdown("#### 💥 吸金人氣王 (成交量 Top 30)")
        top_vol.index = range(1, len(top_vol) + 1)
        st.table(top_vol[["產業別", "名稱", "最新報價", "漲跌幅 (%)", "成交量 (張)"]].style.apply(color_tw_col, subset=["漲跌幅 (%)"]).format({"最新報價": "{:.2f}", "漲跌幅 (%)": "{:.2f}", "成交量 (張)": "{:,.0f}"}))
        
    else:
        st.error("⚠️ 暫時無法取得大盤資料 (即使偽裝仍受限)，請稍後再重整頁面。")
