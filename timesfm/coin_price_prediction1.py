# timesfm/model.py
import pandas as pd
import numpy as np
import timesfm
from typing import Dict, List, Optional, Tuple
import yfinance as yf
from datetime import datetime, timedelta
import torch
from transformers import AutoTokenizer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch.utils.data import Dataset
from timesfm.pytorch_patched_decoder import PatchedTimeSeriesDecoder
from finetuning.finetuning_torch import FinetuningConfig, TimesFMFinetuner
import matplotlib.pyplot as plt
from huggingface_hub import snapshot_download
from os import path

class TimeSeriesDataset(Dataset):
    def __init__(self, series: np.ndarray, context_length: int, horizon_length: int, freq_type: int = 0):
        if freq_type not in [0, 1, 2]:
            raise ValueError("freq_type must be 0, 1, or 2")
            
        self.series = series
        self.context_length = context_length
        self.horizon_length = horizon_length  
        self.freq_type = freq_type
        self._prepare_samples()
        
    def _prepare_samples(self) -> None:
        self.samples = []
        total_length = self.context_length + self.horizon_length
        
        for start_idx in range(0, len(self.series) - total_length + 1):
            end_idx = start_idx + self.context_length
            x_context = self.series[start_idx:end_idx]
            x_future = self.series[end_idx:end_idx + self.horizon_length]
            self.samples.append((x_context, x_future))
            
    def __len__(self) -> int:
        return len(self.samples)
        
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x_context, x_future = self.samples[index]
        
        x_context = torch.tensor(x_context, dtype=torch.float32)
        x_future = torch.tensor(x_future, dtype=torch.float32)
        
        input_padding = torch.zeros_like(x_context)
        freq = torch.tensor([self.freq_type], dtype=torch.long)
        
        return x_context, input_padding, freq, x_future

class CoinPricePredictor:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.target_horizon = 30  # 실제로 사용할 예측 기간
        self.horizon_len = 128   # TimesFM 요구사항
        self.context_len = 192   # Must be multiple of 32
        
        # TimesFM 모델 초기화
        self.model, self.hparams, self.model_config = self._initialize_model()
        self.scaler = StandardScaler()
        
    def _initialize_model(self):
        repo_id = "google/timesfm-2.0-500m-pytorch"
        hparams = timesfm.TimesFmHparams(
            backend=self.device,
            per_core_batch_size=32,
            horizon_len=self.horizon_len,
            num_layers=50,
            use_positional_embedding=False,
            context_len=self.context_len
        )
        
        tfm = timesfm.TimesFm(
            hparams=hparams,
            checkpoint=timesfm.TimesFmCheckpoint(
                huggingface_repo_id=repo_id
            )
        )
        
        # 체크포인트 다운로드 및 로딩
        checkpoint_path = path.join(snapshot_download(repo_id), "torch_model.ckpt")
        model = PatchedTimeSeriesDecoder(tfm._model_config)
        loaded_checkpoint = torch.load(checkpoint_path, map_location=self.device)
        model.load_state_dict(loaded_checkpoint)
        model = model.to(self.device)
        
        return model, hparams, tfm._model_config
        
    def get_market_data(self) -> pd.DataFrame:
        """시장 데이터 수집"""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365*2)
        
        tickers = {
            'BTC-USD': 'Bitcoin',
            '^IXIC': 'NASDAQ',
            '^GSPC': 'SP500',
            'GC=F': 'Gold',
            'CL=F': 'Oil',
            '^RUT': 'Russell2000',
            '^TNX': 'Treasury_Yield',
            '^VIX': 'VIX'
        }
        
        df = pd.DataFrame()
        for ticker, name in tickers.items():
            try:
                data = yf.download(ticker, start=start_date, end=end_date)
                if len(data) > 0:
                    df[f'{name}_Close'] = data['Close']
                    if ticker == 'BTC-USD':
                        df[f'{name}_Volume'] = data['Volume']
                        df[f'{name}_High'] = data['High']
                        df[f'{name}_Low'] = data['Low']
            except Exception as e:
                print(f"Error downloading {ticker}: {e}")
        
        # 결측치 처리
        df = df.fillna(method='ffill').fillna(method='bfill')
        
        # 충분한 데이터가 있는지 확인
        min_required = self.context_len + 128
        if len(df) < min_required:
            raise ValueError(f"충분한 데이터를 수집하지 못했습니다. 최소 {min_required}일의 데이터가 필요합니다.")
        
        return df

    def prepare_datasets(self, df: pd.DataFrame, train_split: float = 0.8) -> Tuple[Dataset, Dataset]:
        """데이터셋 준비"""
        # 데이터 정규화
        series = df['Bitcoin_Close'].values.reshape(-1, 1)
        series = self.scaler.fit_transform(series).flatten()
        
        # 학습/검증 데이터 분할
        train_size = int(len(series) * train_split)
        train_data = series[:train_size]
        val_data = series[train_size:]
        
        # 데이터셋 생성 - horizon_length를 128로 고정
        train_dataset = TimeSeriesDataset(
            train_data,
            context_length=self.context_len,
            horizon_length=128,  # TimesFM 요구사항
            freq_type=0
        )
        
        val_dataset = TimeSeriesDataset(
            val_data,
            context_length=self.context_len,
            horizon_length=128,  # TimesFM 요구사항
            freq_type=0
        )
        
        print(f"학습 샘플 수: {len(train_dataset)}")
        print(f"검증 샘플 수: {len(val_dataset)}")
        
        return train_dataset, val_dataset

    def train(self, train_dataset, val_dataset):
        """모델 파인튜닝"""
        config = FinetuningConfig(
            batch_size=32,
            num_epochs=10,
            learning_rate=1e-4,
            use_wandb=False,
            freq_type=0,
            log_every_n_steps=10,
            val_check_interval=0.5,
            use_quantile_loss=True
        )
        
        finetuner = TimesFMFinetuner(self.model, config)
        results = finetuner.finetune(
            train_dataset=train_dataset,
            val_dataset=val_dataset
        )
        
        return results

    def plot_predictions(self, val_dataset: Dataset, save_path: str = "predictions.png"):
        """예측 결과 시각화"""
        self.model.eval()
        
        x_context, x_padding, freq, x_future = val_dataset[0]
        x_context = x_context.unsqueeze(0).to(self.device)
        x_padding = x_padding.unsqueeze(0).to(self.device)
        freq = freq.unsqueeze(0).to(self.device)
        x_future = x_future.unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            predictions = self.model(x_context, x_padding.float(), freq)
            predictions_mean = predictions[..., 0]
            last_patch_pred = predictions_mean[:, -1, :]
            
        # 역정규화
        context_vals = self.scaler.inverse_transform(x_context[0].cpu().numpy().reshape(-1, 1))
        future_vals = self.scaler.inverse_transform(x_future[0].cpu().numpy().reshape(-1, 1))
        pred_vals = self.scaler.inverse_transform(last_patch_pred[0].cpu().numpy().reshape(-1, 1))
        
        plt.figure(figsize=(12, 6))
        plt.plot(range(len(context_vals)), context_vals, label="Historical", color="blue")
        plt.plot(range(len(context_vals), len(context_vals) + len(future_vals)), 
                future_vals, label="Actual", color="green", linestyle="--")
        plt.plot(range(len(context_vals), len(context_vals) + len(pred_vals)), 
                pred_vals, label="Predicted", color="red")
        
        plt.title("Bitcoin Price Prediction")
        plt.xlabel("Time")
        plt.ylabel("Price")
        plt.legend()
        plt.savefig(save_path)
        plt.close()

    def predict(self, context_data: pd.DataFrame, dynamic_covariates: Dict) -> np.ndarray:
        """예측 수행"""
        scaled_context = self.scaler.transform(context_data[['Bitcoin_Close']])
        
        # 128일 예측 수행
        point_forecast, _ = self.model.forecast(
            scaled_context,
            freq=0,
            dynamic_covariates=dynamic_covariates
        )
        
        # 필요한 30일만 선택
        predictions = self.scaler.inverse_transform(point_forecast)
        return predictions[:, :self.target_horizon]

def main():
    predictor = CoinPricePredictor()
    print("모델 초기화 완료")
    
    # 데이터 수집
    market_data = predictor.get_market_data()
    print("데이터 수집 완료")
    
    # 데이터셋 준비
    train_dataset, val_dataset = predictor.prepare_datasets(market_data)
    print(f"데이터셋 준비 완료 - 학습: {len(train_dataset)}, 검증: {len(val_dataset)}")
    
    # 모델 학습
    print("학습 시작...")
    results = predictor.train(train_dataset, val_dataset)
    print("학습 완료")
    
    # 예측 결과 시각화
    predictor.plot_predictions(val_dataset, "bitcoin_predictions.png")
    print("예측 결과가 bitcoin_predictions.png에 저장되었습니다.")

if __name__ == "__main__":
    main()