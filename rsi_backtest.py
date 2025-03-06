# RSI 백테스팅 전략 구현 (수정 버전)
import numpy as np
import pandas as pd
import vectorbt as vbt
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from matplotlib import font_manager, rc

class RSIBacktester:
    def __init__(self, coin_target='BTC', coin_refer='USDT', train_days=60, test_days=30,
                 rsi_window=25, rsi_oversold=20, rsi_overbought=70):
        """RSI 백테스팅 클래스 초기화
        
        Args:
            coin_target (str): 대상 코인 (예: 'BTC')
            coin_refer (str): 기준 화폐 (예: 'USDT')
            train_days (int): 학습 기간 (일)
            test_days (int): 테스트 기간 (일)
            rsi_window (int): RSI 계산 기간
            rsi_oversold (int): 과매도 기준점
            rsi_overbought (int): 과매수 기준점
        """
        self.coin_target = coin_target
        self.coin_refer = coin_refer
        self.train_days = train_days
        self.test_days = test_days
        self.rsi_window = rsi_window
        self.rsi_oversold = rsi_oversold
        self.rsi_overbought = rsi_overbought
        
        # 데이터 및 분석 결과 저장용 변수
        self.train_data = None
        self.test_data = None
        self.rsi = None
        self.portfolio = None
        self.optimization_results = None
        
        # 시각화 설정
        plt.rcParams['font.family'] = 'AppleGothic'
        plt.rcParams['axes.unicode_minus'] = False
        
    def fetch_data(self):
        """학습 및 테스트용 데이터 다운로드"""
        end_date = datetime.now()
        test_start = end_date - timedelta(days=self.test_days)
        train_start = test_start - timedelta(days=self.train_days)
        
        # 학습 데이터 다운로드
        train_data = vbt.BinanceData.download(
            f'{self.coin_target}{self.coin_refer}',
            start=train_start,
            end=test_start,
            interval='1m'
        )
        self.train_data = train_data.get()
        
        # 테스트 데이터 다운로드
        test_data = vbt.BinanceData.download(
            f'{self.coin_target}{self.coin_refer}',
            start=test_start,
            end=end_date,
            interval='1m'
        )
        self.test_data = test_data.get()
        
        return self.train_data, self.test_data
    
    def calculate_rsi(self):
        """RSI 지표 계산"""
        if self.test_data is None:
            self.fetch_data()
            
        self.rsi = vbt.indicators.RSI.run(
            self.test_data['Close'],
            window=self.rsi_window,
            short_name='RSI'
        )
        return self.rsi
    
    def run_backtest(self, init_cash=1000, fees=0.001):
        """테스트 데이터로 백테스팅 실행"""
        if self.test_data is None:
            self.fetch_data()
            
        self.rsi = vbt.indicators.RSI.run(
            self.test_data['Close'],
            window=self.rsi_window,
            short_name='RSI'
        )
        
        entries = self.rsi.rsi_crossed_above(self.rsi_oversold)
        exits = self.rsi.rsi_crossed_below(self.rsi_overbought)
        
        self.portfolio = vbt.Portfolio.from_signals(
            close=self.test_data['Close'],
            entries=entries,
            exits=exits,
            init_cash=init_cash,
            fees=fees,
            freq='1m',
            direction='both',
            upon_opposite_entry='ignore',
            upon_long_conflict='ignore',
            log=True
        )
        return self.portfolio
    
    def optimize_parameters(self, init_cash=1000, fees=0.001):
        """학습 데이터로 RSI 전략의 최적 파라미터를 찾습니다."""
        if self.train_data is None:
            self.fetch_data()
            
        # 파라미터 범위 설정
        param_grid = {
            'window': np.arange(5, 30, 5),
            'oversold': np.arange(20, 40, 5),
            'overbought': np.arange(60, 80, 5)
        }
        
        results = []
        
        # 학습 데이터로 최적화
        for window in param_grid['window']:
            for oversold in param_grid['oversold']:
                for overbought in param_grid['overbought']:
                    rsi = vbt.indicators.RSI.run(
                        self.train_data['Close'],
                        window=window,
                        short_name='RSI'
                    )
                    
                    entries = rsi.rsi_crossed_above(oversold)
                    exits = rsi.rsi_crossed_below(overbought)
                    
                    pf = vbt.Portfolio.from_signals(
                        close=self.train_data['Close'],
                        entries=entries,
                        exits=exits,
                        init_cash=init_cash,
                        fees=fees,
                        freq='1m',
                        direction='both'
                    )
                    
                    results.append({
                        'window': window,
                        'oversold': oversold,
                        'overbought': overbought,
                        'total_return': pf.total_return(),
                        'sharpe_ratio': pf.sharpe_ratio(),
                        'max_drawdown': pf.max_drawdown()
                    })
        
        self.optimization_results = pd.DataFrame(results)
        
        best_params = self.optimization_results.loc[
            self.optimization_results['sharpe_ratio'].idxmax()
        ]
        
        self.rsi_window = int(best_params['window'])
        self.rsi_oversold = int(best_params['oversold'])
        self.rsi_overbought = int(best_params['overbought'])
        
        return best_params.to_dict()
    
    def print_optimization_results(self):
        """최적화 결과를 출력합니다."""
        if self.optimization_results is None:
            print("최적화를 먼저 실행해주세요.")
            return
            
        best_params = self.optimization_results.loc[
            self.optimization_results['sharpe_ratio'].idxmax()
        ]
        
        print("\n===== RSI 전략 최적화 결과 =====")
        print(f"최적 RSI 기간: {int(best_params['window'])}")
        print(f"최적 과매도 기준: {int(best_params['oversold'])}")
        print(f"최적 과매수 기준: {int(best_params['overbought'])}")
        print(f"총 수익률: {best_params['total_return'] * 100:.2f}%")
        print(f"샤프 비율: {best_params['sharpe_ratio']:.2f}")
        print(f"최대 낙폭: {best_params['max_drawdown'] * 100:.2f}%")
    
    def print_results(self):
        """백테스팅 결과 출력"""
        if self.portfolio is None:
            print("백테스트를 먼저 실행해주세요.")
            return
            
        print("\n===== RSI 전략 백테스팅 결과 =====")
        print(f"총 수익: ${self.portfolio.total_profit():.2f}")
        print(f"총 수익률: {self.portfolio.total_return() * 100:.2f}%")
        print(f"최대 낙폭 (MDD): {self.portfolio.max_drawdown() * 100:.2f}%")
        print(f"샤프 비율: {self.portfolio.sharpe_ratio():.2f}")
        print(f"총 거래 횟수: {len(self.portfolio.trades)}")
        
        # 승률 계산
        winning_trades = (self.portfolio.trades.returns > 0).sum()
        total_trades = len(self.portfolio.trades)
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
        print(f"승률: {win_rate:.2f}%")
    
    def plot_results(self):
        """결과 시각화"""
        if self.portfolio is None:
            print("백테스트를 먼저 실행해주세요.")
            return
            
        fig, axes = plt.subplots(3, 1, figsize=(15, 15), 
                                gridspec_kw={'height_ratios': [3, 1, 1]})
        
        # 가격 차트와 매수/매도 신호
        entries = self.portfolio.entries
        exits = self.portfolio.exits
        
        axes[0].plot(self.test_data.index, self.test_data['Close'], label='가격', color='blue')
        for idx in entries[entries].index:
            axes[0].axvline(x=idx, color='g', alpha=0.3, linestyle='--')
        for idx in exits[exits].index:
            axes[0].axvline(x=idx, color='r', alpha=0.3, linestyle='--')
        axes[0].set_title('Price Chart and Signals')
        axes[0].legend()
        
        # RSI 지표
        axes[1].plot(self.test_data.index, self.rsi.rsi, label='RSI', color='purple')
        axes[1].axhline(y=self.rsi_oversold, color='g', linestyle='--', label='과매도')
        axes[1].axhline(y=self.rsi_overbought, color='r', linestyle='--', label='과매수')
        axes[1].set_title('RSI Indicator')
        axes[1].legend()
        
        # 자본금 변화
        equity = self.portfolio.value()
        axes[2].plot(equity.index, equity, label='자본금', color='green')
        axes[2].set_title('Portfolio Value')
        axes[2].legend()
        
        # x축 레이블 포맷 설정
        for ax in axes:
            ax.tick_params(axis='x', rotation=45)
            ax.grid(True)
        
        plt.tight_layout()
        plt.show()

def main():
    # RSI 백테스터 인스턴스 생성
    backtest = RSIBacktester(
        coin_target='BTC',
        coin_refer='USDT',
        train_days=60,  # 60일 학습
        test_days=30,   # 30일 테스트
        rsi_window=25,
        rsi_oversold=20,
        rsi_overbought=70
    )
    
    # 학습 데이터로 최적화 실행
    print("학습 데이터로 파라미터 최적화 중...")
    backtest.optimize_parameters()
    backtest.print_optimization_results()
    
    # 테스트 데이터로 백테스팅 실행
    print("\n최적화된 파라미터로 테스트 데이터 백테스팅 실행 중...")
    backtest.run_backtest(init_cash=1000, fees=0.001)
    
    # 결과 출력
    backtest.print_results()
    
    # 결과 시각화
    backtest.plot_results()

if __name__ == "__main__":
    main()
