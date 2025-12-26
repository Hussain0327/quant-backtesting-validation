import { cn } from '@/lib/utils/cn';

interface MetricCardProps {
  label: string;
  value: string | number;
  delta?: string | number;
  deltaType?: 'positive' | 'negative' | 'neutral';
  className?: string;
}

export function MetricCard({
  label,
  value,
  delta,
  deltaType = 'neutral',
  className,
}: MetricCardProps) {
  const deltaColors = {
    positive: 'text-success',
    negative: 'text-error',
    neutral: 'text-text-secondary',
  };

  return (
    <div className={cn('card p-4', className)}>
      <div className="metric-label mb-1">{label}</div>
      <div className="metric-value">{value}</div>
      {delta !== undefined && (
        <div className={cn('text-sm mt-1', deltaColors[deltaType])}>
          {deltaType === 'positive' && '+'}
          {delta}
        </div>
      )}
    </div>
  );
}
