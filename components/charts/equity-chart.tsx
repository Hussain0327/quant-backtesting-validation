'use client';

import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  ReferenceLine,
} from 'recharts';
import { CHART_COLORS } from '@/lib/utils/constants';
import { format } from 'date-fns';

interface EquityPoint {
  date: Date | string;
  equity: number;
  price: number;
}

interface EquityChartProps {
  equityCurve: EquityPoint[];
  buyHoldEquity?: EquityPoint[];
  initialCapital?: number;
}

export function EquityChart({
  equityCurve,
  buyHoldEquity,
  initialCapital = 10000,
}: EquityChartProps) {
  // Combine data for chart
  const chartData = equityCurve.map((point, i) => {
    const buyHold = buyHoldEquity?.[i];
    return {
      date: typeof point.date === 'string' ? point.date : format(point.date, 'MMM d'),
      Strategy: point.equity,
      'Buy & Hold': buyHold?.equity ?? initialCapital,
    };
  });

  return (
    <div className="h-[300px] w-full">
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={chartData} margin={{ top: 5, right: 30, left: 20, bottom: 5 }}>
          <CartesianGrid strokeDasharray="3 3" stroke={CHART_COLORS.grid} />
          <XAxis
            dataKey="date"
            stroke="#94a3b8"
            tick={{ fill: '#94a3b8', fontSize: 12 }}
            tickLine={{ stroke: '#94a3b8' }}
          />
          <YAxis
            stroke="#94a3b8"
            tick={{ fill: '#94a3b8', fontSize: 12 }}
            tickLine={{ stroke: '#94a3b8' }}
            tickFormatter={(value) => `$${(value / 1000).toFixed(0)}k`}
          />
          <Tooltip
            contentStyle={{
              backgroundColor: '#1e293b',
              border: '1px solid #334155',
              borderRadius: '8px',
            }}
            labelStyle={{ color: '#e2e8f0' }}
            formatter={(value: number) => [`$${value.toFixed(0)}`, '']}
          />
          <Legend />
          <ReferenceLine y={initialCapital} stroke="#64748b" strokeDasharray="3 3" />
          <Line
            type="monotone"
            dataKey="Strategy"
            stroke={CHART_COLORS.equity}
            strokeWidth={2}
            dot={false}
          />
          {buyHoldEquity && (
            <Line
              type="monotone"
              dataKey="Buy & Hold"
              stroke={CHART_COLORS.benchmark}
              strokeWidth={2}
              strokeDasharray="5 5"
              dot={false}
            />
          )}
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
