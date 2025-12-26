import { MetricCard } from './metric-card';
import { FORMATTERS } from '@/lib/utils/constants';

interface Metrics {
  totalReturn: number;
  sharpe: number;
  maxDrawdown: number;
  winRate: number;
  numTrades: number;
  finalEquity: number;
}

interface MetricsGridProps {
  metrics: Metrics;
  initialCapital?: number;
}

export function MetricsGrid({ metrics, initialCapital = 10000 }: MetricsGridProps) {
  const profit = metrics.finalEquity - initialCapital;

  return (
    <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-4">
      <MetricCard
        label="Test Return"
        value={`${metrics.totalReturn >= 0 ? '+' : ''}${metrics.totalReturn.toFixed(1)}%`}
        deltaType={metrics.totalReturn >= 0 ? 'positive' : 'negative'}
      />
      <MetricCard
        label="Sharpe Ratio"
        value={metrics.sharpe.toFixed(2)}
        deltaType={metrics.sharpe >= 0.5 ? 'positive' : metrics.sharpe >= 0 ? 'neutral' : 'negative'}
      />
      <MetricCard
        label="Max Drawdown"
        value={`${metrics.maxDrawdown.toFixed(1)}%`}
        deltaType={metrics.maxDrawdown >= -10 ? 'neutral' : 'negative'}
      />
      <MetricCard
        label="Win Rate"
        value={`${metrics.winRate.toFixed(0)}%`}
        deltaType={metrics.winRate >= 50 ? 'positive' : 'neutral'}
      />
      <MetricCard
        label="Trades"
        value={metrics.numTrades.toString()}
      />
      <MetricCard
        label="Final Equity"
        value={FORMATTERS.currency.format(metrics.finalEquity)}
        delta={FORMATTERS.currency.format(profit)}
        deltaType={profit >= 0 ? 'positive' : 'negative'}
      />
    </div>
  );
}
