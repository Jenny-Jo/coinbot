# RSI 백테스팅 전략 구현 (수정 버전)
import numpy as np
import pandas as pd
import vectorbt as vbt
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# 데이터 다운로드 파라미터 설정
coin_target = 'BTC'  # 비트코인
coin_refer = 'USDT'  # 테더
end_date = datetime.now()  # 현재 시간까지
start_date = end_date - timedelta(days=30)  # 30일 전부터

# Binance에서 데이터 다운로드
binance_data = vbt.BinanceData.download(
    '%s%s' % (coin_target, coin_refer),
    start=start_date,
    end=end_date,
    interval='1m',
    tqdm_kwargs=dict(ncols='100%')
)
data = binance_data.get()

# 데이터 확인
print("데이터 샘플:")
print(data.head())
print("\n데이터 정보:")
print(data.info())

# RSI 파라미터 설정
rsi_window = 14  # RSI 계산 기간
rsi_oversold = 30  # 과매도 기준점
rsi_overbought = 70  # 과매수 기준점

# RSI 계산
rsi = vbt.RSI.run(data['Close'], window=rsi_window)

# 매수/매도 신호 생성 (수정된 부분)
# crossover 인자 대신 크로스오버 로직을 직접 구현
rsi_below_threshold = rsi.rsi < rsi_oversold
rsi_above_threshold = rsi.rsi > rsi_overbought

# 과매도 상태에서 상승 돌파 (매수 신호)
entries = (rsi_below_threshold.shift(1) & ~rsi_below_threshold)

# 과매수 상태에서 하락 돌파 (매도 신호)
exits = (rsi_above_threshold.shift(1) & ~rsi_above_threshold)

# 포트폴리오 백테스팅 실행
pf = vbt.Portfolio.from_signals(
    data['Close'],
    entries,
    exits,
    init_cash=1000,  # 초기 자본금
    fees=0.001,      # 거래 수수료 (0.1%)
    freq='1m'        # 데이터 주기
)

# 백테스팅 결과 출력
print("\n===== RSI 전략 백테스팅 결과 =====")
print(f"총 수익: ${pf.total_profit:.2f}")
print(f"총 수익률: {pf.total_return * 100:.2f}%")
print(f"최대 낙폭 (MDD): {pf.max_drawdown * 100:.2f}%")
print(f"승률: {pf.win_rate * 100:.2f}%")
print(f"총 거래 횟수: {pf.count}")

# 결과 시각화
fig, axes = plt.subplots(3, 1, figsize=(15, 15), gridspec_kw={'height_ratios': [3, 1, 1]})

# 가격 차트와 매수/매도 신호
pf.plot(ax=axes[0])
axes[0].set_title('가격 차트 및 매수/매도 신호')

# RSI 지표
rsi.plot(ax=axes[1])
axes[1].axhline(y=rsi_oversold, color='g', linestyle='--')
axes[1].axhline(y=rsi_overbought, color='r', linestyle='--')
axes[1].set_title('RSI 지표')

# 자본금 변화
pf.plot_equity(ax=axes[2])
axes[2].set_title('자본금 변화')

plt.tight_layout()
plt.show()

# 월별 수익률 분석
monthly_returns = pf.returns.resample('M').apply(lambda x: (1 + x).prod() - 1)
print("\n===== 월별 수익률 =====")
print(monthly_returns)

# 파라미터 최적화 함수
def optimize_rsi_strategy():
    windows = np.arange(5, 30, 5)  # RSI 기간: 5, 10, 15, 20, 25
    oversolds = np.arange(20, 40, 5)  # 과매도 기준: 20, 25, 30, 35
    overboughts = np.arange(60, 80, 5)  # 과매수 기준: 60, 65, 70, 75
    
    # 파라미터 최적화 실행
    rsi_opt = vbt.RSI.run(
        data['Close'],
        window=windows,
        param_product=True
    )
    
    entries_opt = {}
    exits_opt = {}
    
    for w in windows:
        for os in oversolds:
            for ob in overboughts:
                key = (w, os, ob)
                entries_opt[key] = rsi_opt.rsi_below(os, crossover=True, window=w)
                exits_opt[key] = rsi_opt.rsi_above(ob, crossover=True, window=w)
    
    pf_opt = vbt.Portfolio.from_signal_arrays(
        data['Close'],
        entries_opt,
        exits_opt,
        init_cash=1000,
        fees=0.001,
        freq='1m'
    )
    
    # 최적 파라미터 찾기
    metrics = pf_opt.metrics(['total_return', 'sharpe_ratio', 'max_drawdown'])
    best_params = metrics.sort_values('total_return', ascending=False).index[0]
    
    print("\n===== 최적 파라미터 =====")
    print(f"RSI 기간: {best_params[0]}")
    print(f"과매도 기준: {best_params[1]}")
    print(f"과매수 기준: {best_params[2]}")
    print(f"총 수익률: {metrics.loc[best_params, 'total_return'] * 100:.2f}%")
    
    return best_params

# 파라미터 최적화 실행 (시간이 오래 걸릴 수 있으므로 주석 처리)
# best_params = optimize_rsi_strategy()
