import numpy as np
import pandas as pd
import vectorbt as vbt
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from tqdm import tqdm
import platform

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
        self.symbol = f'{coin_target}{coin_refer}'
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
        self._setup_plotting()
        
    def _setup_plotting(self):
        """시각화 설정"""
        # 한글 폰트 지원 설정 (Windows 환경에 맞게 수정)
        if platform.system() == 'Windows':
            plt.rcParams['font.family'] = 'Malgun Gothic'  # 윈도우의 경우 맑은 고딕 사용
        elif platform.system() == 'Darwin':  # macOS
            plt.rcParams['font.family'] = 'AppleGothic'
        else:  # Linux 등
            plt.rcParams['font.family'] = 'NanumGothic'
        
        plt.rcParams['axes.unicode_minus'] = False
        
    def fetch_data(self):
        """학습 및 테스트용 데이터 다운로드"""
        end_date = datetime.now()
        test_start = end_date - timedelta(days=self.test_days)
        train_start = test_start - timedelta(days=self.train_days)
        
        # 학습 데이터와 테스트 데이터 다운로드 (vectorbt의 간결한 API 사용)
        train_data = vbt.BinanceData.download(
            self.symbol,
            start=train_start,
            end=test_start,
            interval='1m'
        ).get()
        
        test_data = vbt.BinanceData.download(
            self.symbol,
            start=test_start,
            end=end_date,
            interval='1m'
        ).get()
        
        self.train_data = train_data
        self.test_data = test_data
        
        return train_data, test_data
    
    def calculate_rsi(self, data=None):
        """RSI 지표 계산"""
        if data is None:
            if self.test_data is None:
                self.fetch_data()
            data = self.test_data
            
        # vectorbt의 RSI 지표 계산
        rsi = vbt.indicators.RSI.run(
            data['Close'],
            window=self.rsi_window,
            short_name='RSI'
        )
        return rsi
    
    def run_backtest(self, data=None, init_cash=1000, fees=0.001):
        """데이터로 백테스팅 실행"""
        if data is None:
            if self.test_data is None:
                self.fetch_data()
            data = self.test_data
            
        # RSI 계산
        self.rsi = self.calculate_rsi(data)
        
        # 매수/매도 신호 생성
        entries = self.rsi.rsi_crossed_above(self.rsi_oversold)
        exits = self.rsi.rsi_crossed_below(self.rsi_overbought)
        
        # 포트폴리오 백테스팅 실행
        self.portfolio = vbt.Portfolio.from_signals(
            close=data['Close'],
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
        """학습 데이터로 RSI 전략의 최적 파라미터 탐색"""
        if self.train_data is None:
            self.fetch_data()
            
        # 파라미터 범위 설정
        windows = np.arange(5, 21, 3)
        oversolds = np.arange(25, 35, 2)
        overboughts = np.arange(65, 75, 2)
        
        # RSI 계산 - run_combs 사용
        rsi = vbt.RSI.run_combs(
            close=self.train_data.Close,
            window=windows,
            param_product=True
        )
        
        # 모든 파라미터 조합 생성
        param_product = np.array([
            (w, os, ob)
            for w in windows
            for os in oversolds
            for ob in overboughts
        ])
        
        # 진입/퇴출 시그널 생성
        entries = pd.DataFrame(False, index=self.train_data.index, columns=range(len(param_product)))
        exits = pd.DataFrame(False, index=self.train_data.index, columns=range(len(param_product)))
        
<<<<<<< HEAD
        # 결과를 DataFrame으로 변환
        self.optimization_results = pd.DataFrame(results)
        self.optimization_results = pd.DataFrame(results)
=======
        for i, (w, os, ob) in enumerate(param_product):
            rsi_values = rsi.rsi[w]
            entries.iloc[:, i] = rsi_values.vbt.crossed_above(os)
            exits.iloc[:, i] = rsi_values.vbt.crossed_below(ob)
        
        # 포트폴리오 백테스팅
        portfolio = vbt.Portfolio.from_signals(
            close=self.train_data.Close,
            entries=entries,
            exits=exits,
            init_cash=init_cash,
            fees=fees,
            freq='1m'
        )
        
        # 결과 데이터프레임 생성
        self.optimization_results = pd.DataFrame({
            'window': param_product[:, 0],
            'oversold': param_product[:, 1],
            'overbought': param_product[:, 2],
            'total_return': portfolio.total_return(),
            'sharpe_ratio': portfolio.sharpe_ratio(),
            'max_drawdown': portfolio.max_drawdown(),
            'num_trades': portfolio.trades.count(),
            'win_rate': portfolio.trades.win_rate() * 100
        })
>>>>>>> bf0f0f7 (rsi backtest)
        
        # 평가 기준: 거래횟수가 최소 10회 이상인 것 중에서 샤프 비율이 가장 높은 것 선택
        valid_results = self.optimization_results[self.optimization_results['num_trades'] >= 10]
        
        if len(valid_results) > 0:
            best_idx = valid_results['sharpe_ratio'].idxmax()
            best_params = valid_results.loc[best_idx]
        else:
            best_idx = self.optimization_results['sharpe_ratio'].idxmax()
            best_params = self.optimization_results.loc[best_idx]
        
        # 최적 파라미터 업데이트
        self.rsi_window = int(best_params['window'])
        self.rsi_oversold = int(best_params['oversold'])
        self.rsi_overbought = int(best_params['overbought'])
        
        return best_params.to_dict()
    
    def print_optimization_results(self):
        """최적화 결과를 출력합니다."""
        if self.optimization_results is None:
            print("최적화를 먼저 실행해주세요.")
            return
            
        best_idx = self.optimization_results['sharpe_ratio'].idxmax()
        best_params = self.optimization_results.loc[best_idx]
        
        print("\n===== RSI 전략 최적화 결과 =====")
        print(f"최적 RSI 기간: {int(best_params['window'])}")
        print(f"최적 과매도 기준: {int(best_params['oversold'])}")
        print(f"최적 과매수 기준: {int(best_params['overbought'])}")
        print(f"총 수익률: {best_params['total_return'] * 100:.2f}%")
        print(f"샤프 비율: {best_params['sharpe_ratio']:.2f}")
        print(f"최대 낙폭: {best_params['max_drawdown'] * 100:.2f}%")
        print(f"총 거래 횟수: {int(best_params['num_trades'])}")
        print(f"승률: {best_params['win_rate']:.2f}%")
        
        # 상위 5개 파라미터 조합 출력
        print("\n----- 상위 5개 파라미터 조합 -----")
        top5 = self.optimization_results.sort_values('sharpe_ratio', ascending=False).head(5)
        for idx, row in top5.iterrows():
            print(f"RSI 기간: {int(row['window'])}, 과매도: {int(row['oversold'])}, 과매수: {int(row['overbought'])}")
            print(f"  수익률: {row['total_return']*100:.2f}%, 샤프비율: {row['sharpe_ratio']:.2f}, 거래수: {int(row['num_trades'])}\n")
    
    def print_backtest_results(self):
        """백테스팅 결과 출력"""
        if self.portfolio is None:
            print("백테스트를 먼저 실행해주세요.")
            return
        
        # vectorbt의 백테스팅 결과 통계 활용
        stats = self.portfolio.stats()
        trades = self.portfolio.trades
        
        print("\n===== RSI 전략 백테스팅 결과 =====")
        print(f"시작일: {self.test_data.index[0]}")
        print(f"종료일: {self.test_data.index[-1]}")
        print(f"총 수익: ${self.portfolio.total_profit():.2f}")
        print(f"총 수익률: {self.portfolio.total_return() * 100:.2f}%")
        
        # annual_returns()는 Series를 반환하므로 적절히 처리
        annual_returns = self.portfolio.annual_returns()
        if isinstance(annual_returns, pd.Series):
            # Series인 경우 첫 번째 값 또는 평균 사용
            annual_returns_value = annual_returns.iloc[0] if len(annual_returns) > 0 else 0
        else:
            annual_returns_value = annual_returns
        print(f"연간 수익률: {annual_returns_value * 100:.2f}%")
        
        print(f"최대 낙폭 (MDD): {self.portfolio.max_drawdown() * 100:.2f}%")
        print(f"샤프 비율: {self.portfolio.sharpe_ratio():.2f}")
        print(f"칼마 비율: {self.portfolio.calmar_ratio():.2f}")
        print(f"총 거래 횟수: {len(trades)}")
        
        # 승률 및 추가 통계
        if len(trades) > 0:
            win_rate = trades.win_rate() * 100
            avg_win = trades.winning.pnl.mean() if len(trades.winning) > 0 else 0
            avg_loss = trades.losing.pnl.mean() if len(trades.losing) > 0 else 0
            
            print(f"승률: {win_rate:.2f}%")
            print(f"평균 수익 거래: ${avg_win:.2f}")
            print(f"평균 손실 거래: ${avg_loss:.2f}")
            
            # 연속 승리/손실 스트릭 속성 확인 및 안전하게 접근
            try:
                # 속성 이름이 변경되었을 수 있으므로 다양한 가능성 시도
                if hasattr(trades, 'winning_streak'):
                    print(f"최대 연속 승리: {trades.winning_streak.max()}")
                elif hasattr(trades, 'win_streak'):
                    print(f"최대 연속 승리: {trades.win_streak.max()}")
                else:
                    print("최대 연속 승리: 정보 없음")
                    
                if hasattr(trades, 'losing_streak'):
                    print(f"최대 연속 손실: {trades.losing_streak.max()}")
                elif hasattr(trades, 'loss_streak'):
                    print(f"최대 연속 손실: {trades.loss_streak.max()}")
                else:
                    print("최대 연속 손실: 정보 없음")
            except Exception as e:
                print(f"연속 거래 통계 계산 중 오류 발생: {e}")
    
    def plot_rsi_distribution(self):
        """RSI 분포 시각화"""
        if self.rsi is None:
            print("RSI를 먼저 계산해주세요.")
            return
            
        plt.figure(figsize=(10, 6))
        self.rsi.rsi.plot.hist(bins=50, alpha=0.7)
        plt.axvline(x=self.rsi_oversold, color='g', linestyle='--', label=f'과매도 ({self.rsi_oversold})')
        plt.axvline(x=self.rsi_overbought, color='r', linestyle='--', label=f'과매수 ({self.rsi_overbought})')
        plt.title('RSI 분포')
        plt.xlabel('RSI 값')
        plt.ylabel('빈도')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
    
    def plot_results(self, subplots=True):
        """결과 시각화 (개선된 버전)"""
        if self.portfolio is None:
            print("백테스트를 먼저 실행해주세요.")
            return
        
        try:
            if subplots:
                # 서브플롯으로 전체 시각화 (가격, RSI, 자본금)
                fig, axes = plt.subplots(3, 1, figsize=(15, 15), 
                                        gridspec_kw={'height_ratios': [3, 1, 1]})
                
                # 가격 차트와 매수/매도 신호
                self._plot_price_with_signals(axes[0])
                
                # RSI 지표
                self._plot_rsi(axes[1])
                
                # 자본금 변화
                self._plot_equity(axes[2])
                
                # x축 레이블 포맷 설정
                for ax in axes:
                    ax.tick_params(axis='x', rotation=45)
                    ax.grid(True, alpha=0.3)
                
                plt.tight_layout()
                plt.show()
            else:
                # vectorbt의 내장 시각화 기능 활용
                self.portfolio.plot()
                plt.tight_layout()
                plt.show()
        except Exception as e:
            print(f"결과 시각화 중 오류 발생: {e}")
            print("기본 통계 정보만 출력합니다.")
            # 기본 통계 정보 출력
            print("\n----- 기본 통계 정보 -----")
            print(f"총 수익: ${self.portfolio.total_profit():.2f}")
            print(f"총 수익률: {self.portfolio.total_return() * 100:.2f}%")
            print(f"최대 낙폭: {self.portfolio.max_drawdown() * 100:.2f}%")
            print(f"샤프 비율: {self.portfolio.sharpe_ratio():.2f}")
    
    def _plot_price_with_signals(self, ax):
        """가격 차트와 매수/매도 신호 표시"""
        # 기본 가격 차트
        ax.plot(self.test_data.index, self.test_data['Close'], label='가격', color='blue')
        
        # 매수/매도 시그널 추출 (수정된 부분)
        # vectorbt의 최신 버전에서는 다른 방식으로 접근해야 함
        try:
            # 방법 1: trades를 통해 진입/종료 시점 접근
            if len(self.portfolio.trades) > 0:
                entry_times = self.portfolio.trades.entry_time
                exit_times = self.portfolio.trades.exit_time
                
                # 매수/매도 포인트 표시
                entry_prices = self.test_data['Close'].loc[entry_times]
                exit_prices = self.test_data['Close'].loc[exit_times]
                
                ax.scatter(entry_times, entry_prices, marker='^', color='g', s=100, label='매수 신호')
                ax.scatter(exit_times, exit_prices, marker='v', color='r', s=100, label='매도 신호')
        except Exception as e:
            # 방법 2: 원래 생성한 entries/exits 신호 사용
            print(f"거래 시그널 시각화 중 오류 발생: {e}")
            print("대체 방법으로 시각화 시도...")
            
            # 원래 생성한 entries/exits 신호 사용
            entries = self.rsi.rsi_crossed_above(self.rsi_oversold)
            exits = self.rsi.rsi_crossed_below(self.rsi_overbought)
            
            entry_points = self.test_data['Close'][entries]
            exit_points = self.test_data['Close'][exits]
            
            if not entry_points.empty:
                ax.scatter(entry_points.index, entry_points, marker='^', color='g', s=100, label='매수 신호')
            if not exit_points.empty:
                ax.scatter(exit_points.index, exit_points, marker='v', color='r', s=100, label='매도 신호')
        
        ax.set_title('가격 차트 및 매매 신호', fontsize=14)
        ax.set_ylabel('가격', fontsize=12)
        ax.legend(loc='upper left')
    
    def _plot_rsi(self, ax):
        """RSI 지표 및 기준선 표시"""
        ax.plot(self.test_data.index, self.rsi.rsi, label='RSI', color='purple')
        ax.axhline(y=self.rsi_oversold, color='g', linestyle='--', label=f'과매도 ({self.rsi_oversold})')
        ax.axhline(y=self.rsi_overbought, color='r', linestyle='--', label=f'과매수 ({self.rsi_overbought})')
        ax.axhline(y=50, color='gray', linestyle='-', alpha=0.5)
        
        # RSI 영역 색상 표시
        ax.fill_between(self.test_data.index, self.rsi_oversold, self.rsi.rsi, 
                      where=(self.rsi.rsi <= self.rsi_oversold), 
                      color='green', alpha=0.3)
        ax.fill_between(self.test_data.index, self.rsi.rsi, self.rsi_overbought, 
                      where=(self.rsi.rsi >= self.rsi_overbought), 
                      color='red', alpha=0.3)
        
        ax.set_title('RSI 지표', fontsize=14)
        ax.set_ylabel('RSI 값', fontsize=12)
        ax.set_ylim(0, 100)
        ax.legend(loc='upper left')
    
    def _plot_equity(self, ax):
        """자본금 변화 및 드로다운 표시"""
        # 자본금 변화
        equity = self.portfolio.value()
        underwater = self.portfolio.drawdown()
        
        ax.plot(equity.index, equity, label='자본금', color='green')
        
        # 드로다운 표시
        ax2 = ax.twinx()
        ax2.fill_between(underwater.index, 0, underwater, color='red', alpha=0.3, label='드로다운')
        ax2.set_ylim(1, 0)  # 드로다운은 역순으로 표시 (0이 최상단)
        ax2.set_ylabel('드로다운 (%)', fontsize=12)
        
        # 범례 설정
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
        
        ax.set_title('포트폴리오 가치 및 드로다운', fontsize=14)
        ax.set_ylabel('포트폴리오 가치 ($)', fontsize=12)
    
    def analyze_trade_statistics(self):
        """거래 통계 분석"""
        if self.portfolio is None:
            print("백테스트를 먼저 실행해주세요.")
            return
            
        trades = self.portfolio.trades
        
        if len(trades) == 0:
            print("거래 내역이 없습니다.")
            return
            
        # 전체 거래 통계
        print("\n===== 거래 통계 분석 =====")
        print(f"총 거래 횟수: {len(trades)}")
        print(f"승리 거래: {len(trades.winning)}")
        print(f"손실 거래: {len(trades.losing)}")
        print(f"승률: {trades.win_rate() * 100:.2f}%")
        
        # 수익/손실 거래 분석
        if len(trades.winning) > 0 and len(trades.losing) > 0:
            print("\n----- 수익/손실 통계 -----")
            print(f"평균 수익 거래: ${trades.winning.pnl.mean():.2f}")
            print(f"평균 손실 거래: ${trades.losing.pnl.mean():.2f}")
            print(f"최대 수익 거래: ${trades.pnl.max():.2f}")
            print(f"최대 손실 거래: ${trades.pnl.min():.2f}")
            print(f"수익/손실 비율: {abs(trades.winning.pnl.mean() / trades.losing.pnl.mean()):.2f}")
            
            # 거래 기간 분석
            print("\n----- 거래 기간 분석 -----")
            print(f"평균 거래 유지 기간: {trades.duration.mean()} 분")
            print(f"평균 승리 거래 유지 기간: {trades.winning.duration.mean()} 분")
            print(f"평균 손실 거래 유지 기간: {trades.losing.duration.mean()} 분")
        
        # 시각화: 수익 분포
        try:
            # MappedArray를 pandas Series로 변환하여 시각화
            pnl_series = pd.Series(trades.pnl.values, index=range(len(trades.pnl)))
            plt.figure(figsize=(12, 6))
            pnl_series.hist(bins=30, alpha=0.7)
            plt.axvline(x=0, color='r', linestyle='--')
            plt.title('거래별 수익 분포')
            plt.xlabel('수익/손실 ($)')
            plt.ylabel('거래 수')
            plt.grid(True, alpha=0.3)
            plt.show()
            
            # 시각화: 거래 기간 vs 수익
            plt.figure(figsize=(12, 6))
            duration_series = pd.Series(trades.duration.values, index=range(len(trades.duration)))
            plt.scatter(duration_series, pnl_series, alpha=0.7, c=pnl_series, cmap='coolwarm')
            plt.axhline(y=0, color='r', linestyle='--')
            plt.title('거래 기간 vs 수익')
            plt.xlabel('거래 기간 (분)')
            plt.ylabel('수익/손실 ($)')
            plt.colorbar(label='수익/손실 ($)')
            plt.grid(True, alpha=0.3)
            plt.show()
        except Exception as e:
            print(f"시각화 중 오류 발생: {e}")
            print("대체 통계 정보:")
            print(f"PNL 범위: ${trades.pnl.min():.2f} ~ ${trades.pnl.max():.2f}")
            print(f"PNL 평균: ${trades.pnl.mean():.2f}")
            print(f"거래 기간 범위: {trades.duration.min()} ~ {trades.duration.max()} 분")

def main():
    # RSI 백테스터 인스턴스 생성
    backtest = RSIBacktester(
        coin_target='BTC',
        coin_refer='USDT',
        train_days=60,
        test_days=30,
        rsi_window=14,
        rsi_oversold=30,
        rsi_overbought=70
    )
    
    # 학습 데이터로 최적화 실행
    print("학습 데이터로 파라미터 최적화 중...")
    backtest.fetch_data()  # 데이터 준비
    backtest.optimize_parameters()
    backtest.print_optimization_results()
    
    # 테스트 데이터로 백테스팅 실행
    print("\n최적화된 파라미터로 테스트 데이터 백테스팅 실행 중...")
    backtest.run_backtest(init_cash=1000, fees=0.001)
    
    # 기본 결과 출력
    backtest.print_backtest_results()
    
    # 거래 통계 분석
    backtest.analyze_trade_statistics()
    
    # RSI 분포 시각화
    backtest.plot_rsi_distribution()
    
    # 결과 시각화
    backtest.plot_results()
    
    # vectorBT 내장 시각화도 활용
    print("\nvectorBT 내장 시각화:")
    backtest.plot_results(subplots=False)

if __name__ == "__main__":
    main()