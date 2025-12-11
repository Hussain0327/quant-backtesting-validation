from data.fetcher import fetch_data
from strategies import MovingAverageCrossover, RSIStrategy
from backtest.engine import BacktestEngine
from analytics.metrics import calculate_metrics


def main():
    print('fetching data...')
    data = fetch_data('AAPL', '2022-01-01', '2024-01-01')

    strategy = MovingAverageCrossover(short_window=20, long_window=50)
    engine = BacktestEngine(initial_capital=10000)

    print(f'running backtest with {strategy.name}...')
    results = engine.run(data, strategy)

    train_metrics = calculate_metrics(results['train'])
    test_metrics = calculate_metrics(results['test'])

    print('\n--- train results ---')
    print(f"return: {train_metrics['total_return']:.2f}%")
    print(f"sharpe: {train_metrics['sharpe']:.2f}")
    print(f"max drawdown: {train_metrics['max_drawdown']:.2f}%")
    print(f"trades: {train_metrics['num_trades']}")

    print('\n--- test results ---')
    print(f"return: {test_metrics['total_return']:.2f}%")
    print(f"sharpe: {test_metrics['sharpe']:.2f}")
    print(f"max drawdown: {test_metrics['max_drawdown']:.2f}%")
    print(f"trades: {test_metrics['num_trades']}")


if __name__ == '__main__':
    main()
