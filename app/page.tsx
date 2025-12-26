'use client';

import { useState } from 'react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Slider } from '@/components/ui/slider';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { MetricsGrid } from '@/components/metrics/metrics-grid';
import { EquityChart } from '@/components/charts/equity-chart';
import { STRATEGIES, StrategyKey, FORMATTERS } from '@/lib/utils/constants';
import { Loader2, TrendingUp, AlertTriangle, CheckCircle2, XCircle } from 'lucide-react';

interface BacktestResult {
  success: boolean;
  ticker: string;
  strategy: string;
  params: Record<string, unknown>;
  train: {
    equityCurve: Array<{ date: string; equity: number; price: number }>;
    trades: Array<{ type: string; price: number; shares: number; date: string }>;
    metrics: {
      totalReturn: number;
      sharpe: number;
      maxDrawdown: number;
      winRate: number;
      numTrades: number;
      finalEquity: number;
    };
  };
  test: {
    equityCurve: Array<{ date: string; equity: number; price: number }>;
    trades: Array<{ type: string; price: number; shares: number; date: string }>;
    buyHoldEquity: Array<{ date: string; equity: number; price: number }>;
    metrics: {
      totalReturn: number;
      sharpe: number;
      maxDrawdown: number;
      winRate: number;
      numTrades: number;
      finalEquity: number;
    };
  };
  significance?: {
    sharpeConfidence: {
      sharpe: number;
      ciLower: number;
      ciUpper: number;
      ciIncludesZero: boolean;
    };
    vsBenchmark: {
      observedDiff: number;
      pValue: number;
      significantAt05: boolean;
    };
    vsRandom: {
      pValue: number;
      percentileRank: number;
      significantAt05: boolean;
    };
    returnDistribution: {
      skewness: number;
      excessKurtosis: number;
      isFatTailed: boolean;
      isNegativelySkewed: boolean;
      var95: number;
    };
  };
  summary?: {
    sharpeStatisticallySignificant: boolean;
    beatsBenchmarkSignificantly: boolean;
    beatsRandomTrading: boolean;
    overallEvidence: string;
    testsPassedCount: number;
  };
}

export default function TradingPlatform() {
  // Form state
  const [ticker, setTicker] = useState('AAPL');
  const [startDate, setStartDate] = useState('2023-01-01');
  const [endDate, setEndDate] = useState('2024-12-01');
  const [strategyType, setStrategyType] = useState<StrategyKey>('ma-crossover');
  const [shortWindow, setShortWindow] = useState(20);
  const [longWindow, setLongWindow] = useState(50);
  const [rsiPeriod, setRsiPeriod] = useState(14);
  const [initialCapital, setInitialCapital] = useState(10000);
  const [trainSplit, setTrainSplit] = useState(0.7);

  // Results state
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<BacktestResult | null>(null);

  const selectedStrategy = STRATEGIES[strategyType];

  const runBacktest = async () => {
    setIsLoading(true);
    setError(null);

    try {
      // Build strategy params based on type
      let strategyParams: Record<string, number> = {};
      switch (strategyType) {
        case 'ma-crossover':
          strategyParams = { shortWindow, longWindow };
          break;
        case 'rsi':
          strategyParams = { period: rsiPeriod, oversold: 30, overbought: 70 };
          break;
        case 'momentum':
          strategyParams = { lookback: 20 };
          break;
        case 'pairs-trading':
          strategyParams = { lookback: 20, entryThreshold: 2.0, exitThreshold: 0.5 };
          break;
        case 'bollinger-bands':
          strategyParams = { lookback: 20, numStd: 2.0 };
          break;
      }

      const response = await fetch('/api/backtest', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          ticker,
          startDate,
          endDate,
          strategyType,
          strategyParams,
          initialCapital,
          trainSplit,
          runSignificanceTests: true,
          nBootstrap: 3000,
        }),
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.error || 'Backtest failed');
      }

      setResult(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'An error occurred');
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-background">
      {/* Header */}
      <header className="border-b border-surface bg-background-secondary">
        <div className="container mx-auto px-6 py-4">
          <div className="flex items-center gap-3">
            <TrendingUp className="h-8 w-8 text-primary" />
            <div>
              <h1 className="text-xl font-bold gradient-text">Trading Platform</h1>
              <p className="text-sm text-text-secondary">
                Algorithmic Trading Research & Backtesting
              </p>
            </div>
          </div>
        </div>
      </header>

      {/* Disclaimer */}
      <div className="bg-warning/10 border-b border-warning/30">
        <div className="container mx-auto px-6 py-3">
          <p className="text-sm text-warning flex items-center gap-2">
            <AlertTriangle className="h-4 w-4" />
            <span>
              <strong>Research Framework:</strong> This platform is for educational purposes only.
              Backtest results do not guarantee future performance.
            </span>
          </p>
        </div>
      </div>

      {/* Main Content */}
      <div className="flex">
        {/* Sidebar */}
        <aside className="w-80 border-r border-surface bg-background-secondary min-h-[calc(100vh-120px)] p-6">
          <div className="space-y-6">
            {/* Strategy Selection */}
            <div>
              <h3 className="section-header">Strategy</h3>
              <Select value={strategyType} onValueChange={(v) => setStrategyType(v as StrategyKey)}>
                <SelectTrigger>
                  <SelectValue placeholder="Select strategy" />
                </SelectTrigger>
                <SelectContent>
                  {Object.entries(STRATEGIES).map(([key, strategy]) => (
                    <SelectItem key={key} value={key}>
                      {strategy.name}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
              <p className="text-xs text-text-secondary mt-2">{selectedStrategy.description}</p>
            </div>

            {/* Market Data */}
            <div>
              <h3 className="section-header">Market Data</h3>
              <div className="space-y-4">
                <div>
                  <Label htmlFor="ticker">Ticker Symbol</Label>
                  <Input
                    id="ticker"
                    value={ticker}
                    onChange={(e) => setTicker(e.target.value.toUpperCase())}
                    placeholder="AAPL"
                    className="mt-1"
                  />
                </div>
                <div className="grid grid-cols-2 gap-2">
                  <div>
                    <Label htmlFor="start">Start Date</Label>
                    <Input
                      id="start"
                      type="date"
                      value={startDate}
                      onChange={(e) => setStartDate(e.target.value)}
                      className="mt-1"
                    />
                  </div>
                  <div>
                    <Label htmlFor="end">End Date</Label>
                    <Input
                      id="end"
                      type="date"
                      value={endDate}
                      onChange={(e) => setEndDate(e.target.value)}
                      className="mt-1"
                    />
                  </div>
                </div>
              </div>
            </div>

            {/* Strategy Parameters */}
            <div>
              <h3 className="section-header">Parameters</h3>
              <div className="space-y-4">
                {strategyType === 'ma-crossover' && (
                  <>
                    <div>
                      <Label>Short MA: {shortWindow}</Label>
                      <Slider
                        value={[shortWindow]}
                        onValueChange={(v) => setShortWindow(v[0])}
                        min={5}
                        max={50}
                        step={1}
                        className="mt-2"
                      />
                    </div>
                    <div>
                      <Label>Long MA: {longWindow}</Label>
                      <Slider
                        value={[longWindow]}
                        onValueChange={(v) => setLongWindow(v[0])}
                        min={20}
                        max={200}
                        step={5}
                        className="mt-2"
                      />
                    </div>
                  </>
                )}
                {strategyType === 'rsi' && (
                  <div>
                    <Label>RSI Period: {rsiPeriod}</Label>
                    <Slider
                      value={[rsiPeriod]}
                      onValueChange={(v) => setRsiPeriod(v[0])}
                      min={5}
                      max={30}
                      step={1}
                      className="mt-2"
                    />
                  </div>
                )}
              </div>
            </div>

            {/* Backtest Settings */}
            <div>
              <h3 className="section-header">Settings</h3>
              <div className="space-y-4">
                <div>
                  <Label>Initial Capital: {FORMATTERS.currency.format(initialCapital)}</Label>
                  <Slider
                    value={[initialCapital]}
                    onValueChange={(v) => setInitialCapital(v[0])}
                    min={1000}
                    max={100000}
                    step={1000}
                    className="mt-2"
                  />
                </div>
                <div>
                  <Label>Train/Test Split: {(trainSplit * 100).toFixed(0)}%/{((1 - trainSplit) * 100).toFixed(0)}%</Label>
                  <Slider
                    value={[trainSplit]}
                    onValueChange={(v) => setTrainSplit(v[0])}
                    min={0.5}
                    max={0.9}
                    step={0.05}
                    className="mt-2"
                  />
                </div>
              </div>
            </div>

            {/* Run Button */}
            <Button
              onClick={runBacktest}
              disabled={isLoading}
              className="w-full"
              size="lg"
            >
              {isLoading ? (
                <>
                  <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                  Running...
                </>
              ) : (
                'Run Backtest'
              )}
            </Button>

            {error && (
              <div className="p-3 bg-error/10 border border-error/30 rounded-md">
                <p className="text-sm text-error">{error}</p>
              </div>
            )}
          </div>
        </aside>

        {/* Results Area */}
        <main className="flex-1 p-6">
          {!result ? (
            <Card className="animate-fade-in">
              <CardContent className="text-center py-16">
                <div className="text-6xl mb-4">📊</div>
                <h2 className="text-2xl font-bold text-text-primary mb-2">
                  Welcome to Trading Platform
                </h2>
                <p className="text-text-secondary max-w-md mx-auto">
                  Configure your strategy in the sidebar and click "Run Backtest"
                  to see comprehensive results with statistical significance testing.
                </p>

                <div className="grid grid-cols-3 gap-4 mt-8 max-w-lg mx-auto">
                  <Card>
                    <CardContent className="p-4 text-center">
                      <div className="metric-value text-primary">5</div>
                      <div className="metric-label">Strategies</div>
                    </CardContent>
                  </Card>
                  <Card>
                    <CardContent className="p-4 text-center">
                      <div className="metric-value text-accent">4</div>
                      <div className="metric-label">Statistical Tests</div>
                    </CardContent>
                  </Card>
                  <Card>
                    <CardContent className="p-4 text-center">
                      <div className="metric-value text-success">70/30</div>
                      <div className="metric-label">Train/Test</div>
                    </CardContent>
                  </Card>
                </div>
              </CardContent>
            </Card>
          ) : (
            <div className="space-y-6 animate-fade-in">
              {/* Results Header */}
              <div className="flex items-center justify-between">
                <div>
                  <h2 className="text-2xl font-bold text-text-primary">
                    {result.ticker} - {result.strategy}
                  </h2>
                  <p className="text-text-secondary">
                    Test period results with {(trainSplit * 100).toFixed(0)}% train / {((1 - trainSplit) * 100).toFixed(0)}% test split
                  </p>
                </div>
                {result.summary && (
                  <div className={`px-4 py-2 rounded-full font-medium ${
                    result.summary.testsPassedCount >= 2
                      ? 'bg-success/20 text-success'
                      : result.summary.testsPassedCount === 1
                      ? 'bg-warning/20 text-warning'
                      : 'bg-error/20 text-error'
                  }`}>
                    {result.summary.testsPassedCount}/3 Tests Passed
                  </div>
                )}
              </div>

              {/* Metrics Grid */}
              <MetricsGrid metrics={result.test.metrics} initialCapital={initialCapital} />

              {/* Tabs */}
              <Tabs defaultValue="charts" className="w-full">
                <TabsList>
                  <TabsTrigger value="charts">Charts</TabsTrigger>
                  <TabsTrigger value="analysis">Analysis</TabsTrigger>
                  <TabsTrigger value="trades">Trade Log</TabsTrigger>
                  <TabsTrigger value="stats">Statistics</TabsTrigger>
                </TabsList>

                <TabsContent value="charts">
                  <Card>
                    <CardHeader>
                      <CardTitle>Portfolio Equity vs Buy & Hold</CardTitle>
                    </CardHeader>
                    <CardContent>
                      <EquityChart
                        equityCurve={result.test.equityCurve}
                        buyHoldEquity={result.test.buyHoldEquity}
                        initialCapital={initialCapital}
                      />
                    </CardContent>
                  </Card>
                </TabsContent>

                <TabsContent value="analysis">
                  <div className="grid grid-cols-2 gap-4">
                    <Card>
                      <CardHeader>
                        <CardTitle>Training Period</CardTitle>
                      </CardHeader>
                      <CardContent>
                        <div className="space-y-2">
                          <div className="flex justify-between">
                            <span className="text-text-secondary">Return</span>
                            <span className={result.train.metrics.totalReturn >= 0 ? 'text-success' : 'text-error'}>
                              {result.train.metrics.totalReturn.toFixed(2)}%
                            </span>
                          </div>
                          <div className="flex justify-between">
                            <span className="text-text-secondary">Sharpe</span>
                            <span>{result.train.metrics.sharpe.toFixed(2)}</span>
                          </div>
                          <div className="flex justify-between">
                            <span className="text-text-secondary">Max DD</span>
                            <span className="text-error">{result.train.metrics.maxDrawdown.toFixed(2)}%</span>
                          </div>
                          <div className="flex justify-between">
                            <span className="text-text-secondary">Win Rate</span>
                            <span>{result.train.metrics.winRate.toFixed(1)}%</span>
                          </div>
                          <div className="flex justify-between">
                            <span className="text-text-secondary">Trades</span>
                            <span>{result.train.metrics.numTrades}</span>
                          </div>
                        </div>
                      </CardContent>
                    </Card>
                    <Card>
                      <CardHeader>
                        <CardTitle>Test Period</CardTitle>
                      </CardHeader>
                      <CardContent>
                        <div className="space-y-2">
                          <div className="flex justify-between">
                            <span className="text-text-secondary">Return</span>
                            <span className={result.test.metrics.totalReturn >= 0 ? 'text-success' : 'text-error'}>
                              {result.test.metrics.totalReturn.toFixed(2)}%
                            </span>
                          </div>
                          <div className="flex justify-between">
                            <span className="text-text-secondary">Sharpe</span>
                            <span>{result.test.metrics.sharpe.toFixed(2)}</span>
                          </div>
                          <div className="flex justify-between">
                            <span className="text-text-secondary">Max DD</span>
                            <span className="text-error">{result.test.metrics.maxDrawdown.toFixed(2)}%</span>
                          </div>
                          <div className="flex justify-between">
                            <span className="text-text-secondary">Win Rate</span>
                            <span>{result.test.metrics.winRate.toFixed(1)}%</span>
                          </div>
                          <div className="flex justify-between">
                            <span className="text-text-secondary">Trades</span>
                            <span>{result.test.metrics.numTrades}</span>
                          </div>
                        </div>
                      </CardContent>
                    </Card>
                  </div>
                </TabsContent>

                <TabsContent value="trades">
                  <Card>
                    <CardHeader>
                      <CardTitle>Trade History ({result.test.trades.length} trades)</CardTitle>
                    </CardHeader>
                    <CardContent>
                      <div className="max-h-96 overflow-y-auto">
                        <table className="w-full text-sm">
                          <thead className="sticky top-0 bg-background-secondary">
                            <tr className="text-text-secondary border-b border-surface">
                              <th className="text-left py-2">Type</th>
                              <th className="text-left py-2">Date</th>
                              <th className="text-right py-2">Price</th>
                              <th className="text-right py-2">Shares</th>
                            </tr>
                          </thead>
                          <tbody>
                            {result.test.trades.map((trade, i) => (
                              <tr key={i} className="border-b border-surface/50">
                                <td className={`py-2 ${trade.type === 'buy' ? 'text-success' : 'text-error'}`}>
                                  {trade.type.toUpperCase()}
                                </td>
                                <td className="py-2">{new Date(trade.date).toLocaleDateString()}</td>
                                <td className="text-right py-2">${trade.price.toFixed(2)}</td>
                                <td className="text-right py-2">{trade.shares.toFixed(2)}</td>
                              </tr>
                            ))}
                          </tbody>
                        </table>
                      </div>
                    </CardContent>
                  </Card>
                </TabsContent>

                <TabsContent value="stats">
                  {result.significance ? (
                    <div className="grid grid-cols-2 gap-4">
                      <Card>
                        <CardHeader>
                          <CardTitle className="flex items-center gap-2">
                            Sharpe CI
                            {!result.significance.sharpeConfidence.ciIncludesZero ? (
                              <CheckCircle2 className="h-4 w-4 text-success" />
                            ) : (
                              <XCircle className="h-4 w-4 text-error" />
                            )}
                          </CardTitle>
                        </CardHeader>
                        <CardContent>
                          <div className="space-y-2">
                            <div className="flex justify-between">
                              <span className="text-text-secondary">Point Estimate</span>
                              <span>{result.significance.sharpeConfidence.sharpe.toFixed(3)}</span>
                            </div>
                            <div className="flex justify-between">
                              <span className="text-text-secondary">95% CI</span>
                              <span>
                                [{result.significance.sharpeConfidence.ciLower.toFixed(3)},
                                {result.significance.sharpeConfidence.ciUpper.toFixed(3)}]
                              </span>
                            </div>
                          </div>
                        </CardContent>
                      </Card>

                      <Card>
                        <CardHeader>
                          <CardTitle className="flex items-center gap-2">
                            vs Buy & Hold
                            {result.significance.vsBenchmark.significantAt05 ? (
                              <CheckCircle2 className="h-4 w-4 text-success" />
                            ) : (
                              <XCircle className="h-4 w-4 text-error" />
                            )}
                          </CardTitle>
                        </CardHeader>
                        <CardContent>
                          <div className="space-y-2">
                            <div className="flex justify-between">
                              <span className="text-text-secondary">Observed Diff</span>
                              <span>{result.significance.vsBenchmark.observedDiff.toFixed(4)}</span>
                            </div>
                            <div className="flex justify-between">
                              <span className="text-text-secondary">p-value</span>
                              <span>{result.significance.vsBenchmark.pValue.toFixed(4)}</span>
                            </div>
                          </div>
                        </CardContent>
                      </Card>

                      <Card>
                        <CardHeader>
                          <CardTitle className="flex items-center gap-2">
                            vs Random Trading
                            {result.significance.vsRandom.significantAt05 ? (
                              <CheckCircle2 className="h-4 w-4 text-success" />
                            ) : (
                              <XCircle className="h-4 w-4 text-error" />
                            )}
                          </CardTitle>
                        </CardHeader>
                        <CardContent>
                          <div className="space-y-2">
                            <div className="flex justify-between">
                              <span className="text-text-secondary">Percentile Rank</span>
                              <span>{result.significance.vsRandom.percentileRank?.toFixed(1) ?? 'N/A'}%</span>
                            </div>
                            <div className="flex justify-between">
                              <span className="text-text-secondary">p-value</span>
                              <span>{result.significance.vsRandom.pValue?.toFixed(4) ?? 'N/A'}</span>
                            </div>
                          </div>
                        </CardContent>
                      </Card>

                      <Card>
                        <CardHeader>
                          <CardTitle>Distribution</CardTitle>
                        </CardHeader>
                        <CardContent>
                          <div className="space-y-2">
                            <div className="flex justify-between">
                              <span className="text-text-secondary">Skewness</span>
                              <span className={result.significance.returnDistribution.isNegativelySkewed ? 'text-warning' : ''}>
                                {result.significance.returnDistribution.skewness.toFixed(3)}
                              </span>
                            </div>
                            <div className="flex justify-between">
                              <span className="text-text-secondary">Kurtosis</span>
                              <span className={result.significance.returnDistribution.isFatTailed ? 'text-warning' : ''}>
                                {result.significance.returnDistribution.excessKurtosis.toFixed(3)}
                              </span>
                            </div>
                            <div className="flex justify-between">
                              <span className="text-text-secondary">95% VaR</span>
                              <span className="text-error">
                                {(result.significance.returnDistribution.var95 * 100).toFixed(2)}%
                              </span>
                            </div>
                          </div>
                        </CardContent>
                      </Card>
                    </div>
                  ) : (
                    <Card>
                      <CardContent className="py-8 text-center text-text-secondary">
                        No statistical tests were run. Enable them in settings.
                      </CardContent>
                    </Card>
                  )}

                  {result.summary && (
                    <Card className="mt-4">
                      <CardContent className="py-4">
                        <div className="text-center">
                          <div className={`text-lg font-medium ${
                            result.summary.testsPassedCount >= 2
                              ? 'text-success'
                              : result.summary.testsPassedCount === 1
                              ? 'text-warning'
                              : 'text-error'
                          }`}>
                            {result.summary.overallEvidence}
                          </div>
                        </div>
                      </CardContent>
                    </Card>
                  )}
                </TabsContent>
              </Tabs>
            </div>
          )}
        </main>
      </div>
    </div>
  );
}
