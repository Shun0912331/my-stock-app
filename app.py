import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.trend import MACD
import twstock

st.set_page_config(page_title="帥順股市分析與資產管理神器", layout="wide")

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

# 🌟 新增了第三個分頁：全市場大盤分析
tab1, tab2, tab3 = st.tabs(["📈 個股技術分析", "💰 我的投資組合", "🌍 全市場大盤分析"])

# ----------------------------------------
# 分頁 1：個股技術分析與基本面
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
        ticker_data = yf.Ticker(ticker_symbol)
        st.subheader(f"🏢 **{display_name}** - 基本面與財務指標 (最新季報)")
        info = ticker_data.info
        
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
            st.markdown("*(註：國際資料庫無提供台股獨有之「月營收 MoM」數據，此處為季度比較。)*")
            
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
            
        df_raw = ticker_data.history(period="10y")
        
        if not df_raw.empty:
            df_raw.index = df_raw.index.tz_localize(None)
            
            if tf_option == "日線":
                df = df_raw.copy()
            elif tf_option == "週線":
                df = df_raw.resample('W-FRI').agg({'Open':'first', 'High':'max', 'Low':'min', 'Close':'last', 'Volume':'sum'}).dropna()
            elif tf_option == "月線":
                df = df_raw.resample('ME').agg({'Open':'first', 'High':'max', 'Low':'min', 'Close':'last', 'Volume':'sum'}).dropna()
            elif tf_option == "年線":
                df = df_raw.resample('YE').agg({'Open':'first', 'High':'max', 'Low':'min', 'Close':'last', 'Volume':'sum'}).dropna()

            ma_colors = ['#FFA500', '#FF1493', '#00BFFF', '#9932CC', '#32CD32', '#FF0000', '#0000FF']
            ma_lines = {}
            for i, ma_str in enumerate(selected_mas):
                ma_val = int(ma_str)
                df[f'MA{ma_val}'] = df['Close'].rolling(window=ma_val).mean()
                ma_lines[f'MA{ma_val}'] = ma_colors[i % len(ma_colors)]

            if show_cross:
                if 'MA5' not in df.columns:
                    df['MA5'] = df['Close'].rolling(window=5).mean()
                if 'MA20' not in df.columns:
                    df['MA20'] = df['Close'].rolling(window=20).mean()
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
            if rows == 1:
                row_heights = [1.0]
            else:
                row_heights = [0.5] + [0.5 / len(selected_inds)] * len(selected_inds)
                
            fig = make_subplots(rows=rows, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=row_heights)
            
            fig.add_trace(go.Candlestick(x=df_plot.index, open=df_plot['Open'], high=df_plot['High'], low=df_plot['Low'], close=df_plot['Close'], increasing_line_color='#FF4B4B', decreasing_line_color='#00D26A', name='K線'), row=1, col=1)
            fig.update_yaxes(rangemode='nonnegative', fixedrange=True, row=1, col=1)
            
            for ma_col, color in ma_lines.items():
                fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot[ma_col], line=dict(color=color, width=1.5), name=ma_col), row=1, col=1)

            if show_cross:
                golden_mask = df_plot['Golden_Cross'] == True
                if golden_mask.any():
                    fig.add_trace(go.Scatter(x=df_plot[golden_mask].index, y=df_plot[golden_mask]['Low'] * 0.98, mode='markers', marker=dict(symbol='triangle-up', size=14, color='#FF4B4B', line=dict(width=1, color='white')), name='黃金交叉 (5上穿20)'), row=1, col=1)
                death_mask = df_plot['Death_Cross'] == True
                if death_mask.any():
                    fig.add_trace(go.Scatter(x=df_plot[death_mask].index, y=df_plot[death_mask]['High'] * 1.02, mode='markers', marker=dict(symbol='triangle-down', size=14, color='#00D26A', line=dict(width=1, color='white')), name='死亡交叉 (5下穿20)'), row=1, col=1)

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
                    fig.update_yaxes(rangemode='nonnegative', fixedrange=True, row=current_row, col=1)
                elif ind == "KD":
                    fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['K'], name='K值', line=dict(color='#00BFFF')), row=current_row, col=1)
                    fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['D'], name='D值', line=dict(color='#FFA500')), row=current_row, col=1)
                    fig.update_yaxes(range=[0, 100], fixedrange=True, row=current_row, col=1)
                elif ind == "MACD":
                    macd_colors = ['#FF4B4B' if v > 0 else '#00D26A' for v in df_plot['MACD_hist']]
                    fig.add_trace(go.Bar(x=df_plot.index, y=df_plot['MACD_hist'], marker_color=macd_colors, name='OSC'), row=current_row, col=1)
                    fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['MACD'], name='DIF', line=dict(color='#00BFFF')), row=current_row, col=1)
                    fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['MACD_signal'], name='MACD', line=dict(color='#FFA500')), row=current_row, col=1)
                    fig.update_yaxes(fixedrange=True, row=current_row, col=1)
                elif ind == "RSI":
                    fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['RSI'], name='RSI', line=dict(color='#9932CC')), row=current_row, col=1)
                    fig.add_trace(go.Scatter(x=df_plot.index, y=[70]*len(df_plot), line=dict(color='#FF4B4B', dash='dash'), showlegend=False), row=current_row, col=1)
                    fig.add_trace(go.Scatter(x=df_plot.index, y=[30]*len(df_plot), line=dict(color='#00D26A', dash='dash'), showlegend=False), row=current_row, col=1)
                    fig.update_yaxes(range=[0, 100], fixedrange=True, row=current_row, col=1)
                current_row += 1
                
            fig.update_layout(xaxis_rangeslider_visible=False, height=400 + 150 * len(selected_inds), margin=dict(l=10, r=10, t=80, b=10), legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0.01), dragmode='pan')
            st.plotly_chart(fig, use_container_width=True, config={'scrollZoom': True, 'displayModeBar': False})
        else:
            st.error("找不到該股票資料，可能是代號錯誤或系統連線異常。")

# ----------------------------------------
# 分頁 2：我的投資組合 (損益追蹤)
# ----------------------------------------
with tab2:
    if MY_PORTFOLIO:
        portfolio_data = []
        my_bar = st.progress(0, text="正在為您結算持股最新報價...")
        for i, info in enumerate(MY_PORTFOLIO):
            symbol = info['symbol']
            cost = info['cost']
            shares = info['shares']
            stock_name = info['name']
            category = info['category']
            tick = yf.Ticker(symbol)
            hist = tick.history(period="5d")
            
            if not hist.empty:
                current_price = hist['Close'].iloc[-1]
                if len(hist) >= 2:
                    prev_price = hist['Close'].iloc[-2]
                else:
                    prev_price = current_price
                
                daily_price_diff = current_price - prev_price
                daily_pct_diff = (daily_price_diff / prev_price) * 100 if prev_price > 0 else 0
                daily_profit_diff = daily_price_diff * shares
                
                stock_cost_raw = cost * shares
                stock_value_raw = current_price * shares
                discount = 0.6
                buy_fee = max(20, stock_cost_raw * 0.001425 * discount)
                sell_fee = max(20, stock_value_raw * 0.001425 * discount)
                
                if symbol.startswith("00"):
                    tax = stock_value_raw * 0.001
                    type_label = "ETF"
                else:
                    tax = stock_value_raw * 0.003
                    type_label = "個股"
                
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
            my_bar.progress((i + 1) / len(MY_PORTFOLIO), text="正在為您結算持股最新報價...")
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
            df_portfolio.index = df_portfolio.index + 1
            styled_table = df_portfolio.style.apply(color_tw_col, subset=["淨損益", "報酬率 (%)", "今日漲跌 (%)", "今日獲利增減"]).format({
                "持股數": "{:,.0f}", "平均成本": "{:.2f}", "最新股價": "{:.2f}", "今日漲跌 (%)": "{:.2f}",
                "今日獲利增減": "${:,.0f}", "總成本": "${:,.0f}", "目前市值": "${:,.0f}",
                "淨損益": "${:,.0f}", "報酬率 (%)": "{:.1f}"  
            })
            st.table(styled_table)
            
            csv = df_portfolio.to_csv(index=False, encoding='utf-8-sig')
            st.download_button(label=f"📥 下載【{cat}】持股明細", data=csv, file_name=f"{cat}_明細.csv", mime="text/csv", key=f"dl_{cat}")
            st.divider() 
    else:
        st.info("尚未從試算表讀取到持股資料。請確認您的試算表 A、B、C 欄有正確輸入內容。")

# ----------------------------------------
# 🌟 分頁 3：全市場大盤分析 (新功能)
# ----------------------------------------
with tab3:
    st.subheader("🌍 台灣股市大盤與產業分析")
    st.markdown("*(💡 註：為維持系統極速運算，此區塊追蹤「加權指數」與「台股最具代表性之 30 大權值股及產業 ETF」作為全市場縮影。)*")
    
    @st.cache_data(ttl=300) # 快取 5 分鐘，避免頻繁呼叫
    def get_market_data():
        # 定義大盤、產業代表 ETF、以及重要權值股
        market_tickers = {
            "^TWII": "加權指數 (大盤)",
            "^TWOII": "櫃買指數 (中小型)",
            "0050.TW": "元大台灣50 (大盤縮影)",
            "0056.TW": "元大高股息 (高息代表)",
            "00878.TW": "國泰永續高股息 (ESG)",
            "00881.TW": "國泰台灣5G+ (科技產業)",
            "0055.TW": "元大MSCI金融 (金融產業)",
            "2330.TW": "台積電 (半導體)",
            "2317.TW": "鴻海 (代工)",
            "2454.TW": "聯發科 (IC設計)",
            "2308.TW": "台達電 (電源/綠能)",
            "2881.TW": "富邦金 (金融)",
            "2603.TW": "長榮 (航運)",
            "2382.TW": "廣達 (AI伺服器)",
            "1101.TW": "台泥 (傳產建材)",
            "2002.TW": "中鋼 (傳產鋼鐵)",
            "1216.TW": "統一 (傳產食品)"
        }
        
        symbols = list(market_tickers.keys())
        data_list = []
        
        # 建立進度條
        prog_bar = st.progress(0, text="正在掃描全市場指標股數據...")
        
        for i, sym in enumerate(symbols):
            try:
                t = yf.Ticker(sym)
                hist = t.history(period="5d")
                if len(hist) >= 2:
                    curr = hist['Close'].iloc[-1]
                    prev = hist['Close'].iloc[-2]
                    vol = hist['Volume'].iloc[-1]
                    diff = curr - prev
                    pct = (diff / prev) * 100
                    
                    data_list.append({
                        "代號": sym.replace(".TW", ""),
                        "名稱": market_tickers[sym],
                        "最新報價": round(curr, 2),
                        "漲跌點數": round(diff, 2),
                        "漲跌幅 (%)": round(pct, 2),
                        "成交量 (張)": round(vol / 1000, 0) if sym not in ["^TWII", "^TWOII"] else "大盤總量" 
                    })
            except:
                pass
            prog_bar.progress((i + 1) / len(symbols), text=f"正在解析 {market_tickers[sym]}...")
            
        prog_bar.empty()
        return pd.DataFrame(data_list)
        
    df_market = get_market_data()
    
    if not df_market.empty:
        # --- 區塊 1：大盤指數表現 ---
        st.markdown("### 📊 大盤與櫃買指數表現")
        idx_cols = st.columns(2)
        twii_data = df_market[df_market["代號"] == "^TWII"]
        twoii_data = df_market[df_market["代號"] == "^TWOII"]
        
        if not twii_data.empty:
            twii = twii_data.iloc[0]
            idx_cols[0].metric(label="📈 加權指數 (集中市場)", value=f"{twii['最新報價']:,.2f}", delta=f"{twii['漲跌點數']:.2f} ({twii['漲跌幅 (%)']}%)", delta_color="inverse")
        if not twoii_data.empty:
            twoii = twoii_data.iloc[0]
            idx_cols[1].metric(label="📈 櫃買指數 (中小型股)", value=f"{twoii['最新報價']:,.2f}", delta=f"{twoii['漲跌點數']:.2f} ({twoii['漲跌幅 (%)']}%)", delta_color="inverse")
            
        st.divider()
        
        # 將大盤指數從排行中剔除，只保留個股與 ETF
        df_stocks = df_market[~df_market["代號"].isin(["^TWII", "^TWOII"])].copy()
        
        # --- 區塊 2：產業板塊 (ETF) 表現 ---
        st.markdown("### 🏢 產業板塊與主題表現 (代表性 ETF)")
        df_etf = df_stocks[df_stocks["代號"].str.startswith("00")].copy()
        df_etf = df_etf.sort_values(by="漲跌幅 (%)", ascending=False)
        df_etf.index = range(1, len(df_etf) + 1)
        st.table(df_etf.style.apply(color_tw_col, subset=["漲跌點數", "漲跌幅 (%)"]).format({
            "最新報價": "{:.2f}", "漲跌點數": "{:.2f}", "漲跌幅 (%)": "{:.2f}", "成交量 (張)": "{:,.0f}"
        }))
        
        st.divider()
        
        # --- 區塊 3：權值股排行戰況 ---
        st.markdown("### 🔥 市場焦點權值股戰況")
        df_corp = df_stocks[~df_stocks["代號"].str.startswith("00")].copy()
        
        col_r1, col_r2 = st.columns(2)
        with col_r1:
            st.markdown("#### 🏆 強勢領漲排行 (漲幅 Top 5)")
            top_gainers = df_corp.sort_values(by="漲跌幅 (%)", ascending=False).head(5)
            top_gainers.index = range(1, len(top_gainers) + 1)
            st.table(top_gainers[["名稱", "最新報價", "漲跌幅 (%)"]].style.apply(color_tw_col, subset=["漲跌幅 (%)"]).format({"最新報價": "{:.2f}", "漲跌幅 (%)": "{:.2f}"}))
            
        with col_r2:
            st.markdown("#### 📉 弱勢回檔排行 (跌幅 Top 5)")
            top_losers = df_corp.sort_values(by="漲跌幅 (%)", ascending=True).head(5)
            top_losers.index = range(1, len(top_losers) + 1)
            st.table(top_losers[["名稱", "最新報價", "漲跌幅 (%)"]].style.apply(color_tw_col, subset=["漲跌幅 (%)"]).format({"最新報價": "{:.2f}", "漲跌幅 (%)": "{:.2f}"}))
            
        st.markdown("#### 💥 市場吸金人氣王 (成交量 Top 5)")
        top_vol = df_corp.sort_values(by="成交量 (張)", ascending=False).head(5)
        top_vol.index = range(1, len(top_vol) + 1)
        st.table(top_vol[["名稱", "最新報價", "漲跌幅 (%)", "成交量 (張)"]].style.apply(color_tw_col, subset=["漲跌幅 (%)"]).format({"最新報價": "{:.2f}", "漲跌幅 (%)": "{:.2f}", "成交量 (張)": "{:,.0f}"}))
        
    else:
        st.error("暫時無法取得大盤資料，請稍後再試。")
