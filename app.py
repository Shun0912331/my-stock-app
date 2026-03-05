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
                
                # 🌟 修復 nan 問題：如果這格是空白 (pd.isna) 或是找不到，自動當作「本人」
                if '分類' in df.columns and pd.notna(row['分類']) and str(row['分類']).strip() != "":
                    category = str(row['分類']).strip()
                else:
                    category = "本人"
                    
                portfolio.append({
                    'symbol': symbol,
                    'cost': float(row['成本']), 
                    'shares': int(row['股數']),
                    'name': stock_name,
                    'category': category
                })
        return portfolio
    except Exception as e:
        return []
