import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.trend import MACD
import twstock

# 把網頁標籤也改成帥順的專屬名稱
st.set_page_config(page_title="帥順股市分析與資產管理神器", layout="wide")

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
    """將 DataFrame 直行套用台股紅綠色"""
    return ['color: #FF4B4B' if isinstance(v, (int, float)) and v > 0 
            else 'color: #00D26A' if isinstance(v, (int, float)) and v < 0 
            else '' for v in s]

# ==========================================
# 🚀 正式內容開始 (已暫時關閉密碼鎖功能)
# ==========================================
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

tab1, tab2 = st.tabs(["📈 個股技術分析", "💰 我的投資組合"])

# ----------------------------------------
# 分頁 1：個股技術分析與警示
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
        st.subheader(f"📊 **{display_name}** - 專業技術線圖")
        
        col_ctrl1, col_ctrl2, col_ctrl3 = st.columns(3)
        with col_ctrl1:
            tf_option = st.radio("⏳ K線週期", ["日線", "週線", "月線", "年線"], horizontal=True)
        with col_ctrl2:
            ma_options = ["5", "10", "20", "30", "60", "120", "240"]
            selected_mas = st.multiselect("📈 顯示均線 (可複選)", ma_options, default=["5", "20", "60"])
        with col_ctrl3:
            ind_options = ["成交量", "KD", "MACD", "RSI"]
            selected_inds = st.multiselect("📉 附圖指標 (可複選)", ind_options, default=["成交量", "KD", "MACD"])
            
        show_pe_river = st.checkbox("🌊 疊加本益比河流圖 (僅適用有獲利之個股)", value=False)
        
        ticker_data = yf.Ticker(ticker_symbol)
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
            
            latest_price = df_plot['Close'].iloc[-1]
            
            rows = 1 + len(selected_inds)
            if rows == 1:
                row_heights = [1.0]
            else:
                row_heights = [0.5] + [0.5 / len(selected_inds)] * len(selected_inds)
                
            fig = make_subplots(rows=rows, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=row_heights)
            
            fig.add_trace(go.Candlestick(
                x=df_plot.index, open=df_plot['Open'], high=df_plot['High'], low=df_plot['Low'], close=df_plot['Close'],
                increasing_line_color='#FF4B4B', decreasing_line_color='#00D26A', name='K線'
            ), row=1, col=1)
            
            for ma_col, color in ma_lines.items():
                fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot[ma_col], line=dict(color=color, width=1.5), name=ma_col), row=1, col=1)

            if show_pe_river:
                try:
                    eps = ticker_data.info.get('trailingEps', 0)
                    if eps and eps > 0:
                        pe_ratios = [10, 12, 15, 18, 20, 25]
                        river_colors = ['#c6dbef', '#9ecae1', '#6baed6', '#4292c6', '#2171b5', '#084594']
                        for pe, color in zip(pe_ratios, river_colors):
                            fig.add_trace(go.Scatter(
                                x=df_plot.index, y=[eps * pe]*len(df_plot), 
                                name=f"{pe}X 本益比", line=dict(color=color, dash='dot', width=1.5)
                            ), row=1, col=1)
                    else:
                        st.warning("⚠️ Yahoo財經查無此股票之有效 EPS 資料，無法繪製本益比河流圖。")
                except:
                    pass

            current_row = 2
            for ind in selected_inds:
                if ind == "成交量":
                    vol_colors = ['#FF4B4B' if row['Close'] >= row['Open'] else '#00D26A' for i, row in df_plot.iterrows()]
                    fig.add_trace(go.Bar(x=df_plot.index, y=df_plot['Volume'], marker_color=vol_colors, name='成交量'), row=current_row, col=1)
                elif ind == "KD":
                    fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['K'], name='K值', line=dict(color='#00BFFF')), row=current_row, col=1)
                    fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['D'], name='D值', line=dict(color='#FFA500')), row=current_row, col=1)
                elif ind == "MACD":
                    macd_colors = ['#FF4B4B' if v > 0 else '#00D26A' for v in df_plot['MACD_hist']]
                    fig.add_trace(go.Bar(x=df_plot.index, y=df_plot['MACD_hist'], marker_color=macd_colors, name='柱狀體'), row=current_row, col=1)
                    fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['MACD'], name='MACD', line=dict(color='#00BFFF')), row=current_row, col=1)
                    fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['MACD_signal'], name='Signal', line=dict(color='#FFA500')), row=current_row, col=1)
                elif ind == "RSI":
                    fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['RSI'], name='RSI', line=dict(color='#9932CC')), row=current_row, col=1)
                    fig.add_trace(go.Scatter(x=df_plot.index, y=[70]*len(df_plot), line=dict(color='#FF4B4B', dash='dash'), showlegend=False), row=current_row, col=1)
                    fig.add_trace(go.Scatter(x=df_plot.index, y=[30]*len(df_plot), line=dict(color='#00D26A', dash='dash'), showlegend=False), row=current_row, col=1)
                
                current_row += 1
                
            fig.update_layout(
                xaxis_rangeslider_visible=False, 
                height=400 + 150 * len(selected_inds),
                # 🌟 升級 1：把天花板(t)從 30 挑高到 80，給圖例空間
                margin=dict(l=10, r=10, t=80, b=10),
                legend=dict(
                    orientation="h", 
                    yanchor="bottom", 
                    y=1.01,         # 放在圖表頂部的邊緣
                    xanchor="left", # 統一靠左對齊
                    x=0.01
                ),
                # 🌟 升級 2(a)：把預設的拖曳行為設定為平移 (Pan)，取代原本惱人的框選放大
                dragmode='pan' 
            )
            
            # 🌟 升級 2(b)：注入這行 config 設定，強制解鎖兩指雙縮放(Pinch-to-zoom)的超棒手感
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
            hist = tick.history(period="1d")
            
            if not hist.empty:
                current_price = hist['Close'].iloc[-1]
                
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
                    "category": category, 
                    "股票名稱": stock_name,
                    "股票代號": f"{symbol} ({type_label})",
                    "持股數": shares,
                    "平均成本": cost,
                    "最新股價": round(current_price, 2),
                    "總成本": true_stock_cost,       
                    "目前市值": round(stock_value_raw, 2),
                    "淨損益": round(true_profit, 0),
                    "報酬率 (%)": round(roi, 1) 
                })
            my_bar.progress((i + 1) / len(MY_PORTFOLIO), text="正在為您結算持股最新報價...")
            
        my_bar.empty()
        
        grouped_data = {}
        for p in portfolio_data:
            cat = p["category"]
            if cat not in grouped_data:
                grouped_data[cat] = []
            grouped_data[cat].append(p)
            
        def sort_key(cat):
            if cat in ["本人", "帥順"]: 
                return 0
            return 1
            
        sorted_categories = sorted(grouped_data.keys(), key=sort_key)
        
        for cat in sorted_categories:
            cat_records = grouped_data[cat]
            
            cat_total_cost = sum([p["總成本"] for p in cat_records])
            cat_total_value = sum([p["目前市值"] for p in cat_records])
            cat_total_profit = sum([p["淨損益"] for p in cat_records])
            cat_total_roi = (cat_total_profit / cat_total_cost) * 100 if cat_total_cost > 0 else 0
            
            st.markdown(f"### 👤 【{cat}】的專屬資產")
            
            col1, col2, col3 = st.columns(3)
            col1.metric("總成本 (含手續費)", f"${cat_total_cost:,.0f}")
            col2.metric("目前總市值", f"${cat_total_value:,.0f}")
            col3.metric("總未實現淨利", f"${cat_total_profit:,.0f}", f"{cat_total_roi:.1f}%", delta_color="inverse")
            
            display_list = []
            for p in cat_records:
                display_item = p.copy()
                del display_item["category"]
                display_list.append(display_item)
                
            df_portfolio = pd.DataFrame(display_list)
            df_portfolio.index = df_portfolio.index + 1
            
            styled_table = df_portfolio.style.apply(color_tw_col, subset=["淨損益", "報酬率 (%)"]).format({
                "持股數": "{:,.0f}",
                "平均成本": "{:.2f}",
                "最新股價": "{:.2f}",
                "總成本": "${:,.0f}",          
                "目前市值": "${:,.0f}",
                "淨損益": "${:,.0f}",
                "報酬率 (%)": "{:.1f}"  
            })
            
            st.table(styled_table)
            
            csv = df_portfolio.to_csv(index=False, encoding='utf-8-sig')
            st.download_button(
                label=f"📥 下載【{cat}】持股明細 (CSV/Excel)",
                data=csv,
                file_name=f"{cat}_的持股明細.csv",
                mime="text/csv",
                key=f"download_{cat}" 
            )
            
            st.divider() 
            
        st.caption("💡 想要把完整畫面匯出 PDF？直接使用瀏覽器的「列印 ➔ 另存為 PDF」功能，排版最完美！")
    else:
        st.info("尚未從試算表讀取到持股資料。請確認您的試算表 A、B、C 欄有正確輸入內容。")
