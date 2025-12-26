/**
 * Backtest API Route
 * Runs backtest with specified strategy and returns results
 */

import { NextRequest, NextResponse } from 'next/server';
import yahooFinance from 'yahoo-finance2';
import { createStrategy, StrategyType, OHLCV } from '@/lib/strategies';
import { BacktestEngine } from '@/lib/backtest';
import { calculateMetrics, calculateReturns, buyAndHoldEquity } from '@/lib/analytics/metrics';
import { bootstrapSharpeCI, permutationTest, monteCarloNull, analyzeDistribution } from '@/lib/analytics/significance';
import { pctChange, dropNaN } from '@/lib/math/rolling';

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const {
      ticker,
      startDate,
      endDate,
      strategyType,
      strategyParams,
      initialCapital = 10000,
      trainSplit = 0.7,
      runSignificanceTests = true,
      nBootstrap = 5000,
    } = body;

    // Validate required fields
    if (!ticker || !startDate || !endDate || !strategyType) {
      return NextResponse.json(
        { error: 'Missing required fields: ticker, startDate, endDate, strategyType' },
        { status: 400 }
      );
    }

    // Fetch market data
    const result = await yahooFinance.historical(ticker.toUpperCase(), {
      period1: new Date(startDate),
      period2: new Date(endDate),
    });

    if (!result || result.length === 0) {
      return NextResponse.json(
        { error: `No data found for ticker ${ticker.toUpperCase()}` },
        { status: 404 }
      );
    }

    // Transform to OHLCV format
    const data: OHLCV[] = result.map((row) => ({
      date: row.date,
      open: row.open,
      high: row.high,
      low: row.low,
      close: row.adjClose ?? row.close,
      volume: row.volume,
    }));

    // Create strategy
    const strategy = createStrategy(strategyType as StrategyType, strategyParams || {});

    // Run backtest
    const engine = new BacktestEngine({ initialCapital });
    const backtestResult = engine.run(data, strategy, trainSplit);

    // Calculate metrics
    const trainMetrics = calculateMetrics(backtestResult.train);
    const testMetrics = calculateMetrics(backtestResult.test);

    // Calculate buy-and-hold for comparison
    const testEquity = backtestResult.test.equityCurve;
    const buyHoldEquity = buyAndHoldEquity(testEquity, initialCapital);

    // Prepare response
    const response: Record<string, unknown> = {
      success: true,
      ticker: ticker.toUpperCase(),
      strategy: backtestResult.strategy,
      params: backtestResult.params,
      train: {
        ...backtestResult.train,
        metrics: trainMetrics,
      },
      test: {
        ...backtestResult.test,
        metrics: testMetrics,
        buyHoldEquity,
      },
    };

    // Run significance tests if requested
    if (runSignificanceTests && testEquity.length > 20) {
      const testReturns = calculateReturns(testEquity);
      const prices = testEquity.map((e) => e.price);
      const benchmarkReturns = dropNaN(pctChange(prices, 1));

      // Store significance results in typed local variables
      const sharpeConfidence = bootstrapSharpeCI(testReturns, { nBootstrap });
      const vsBenchmark = permutationTest(testReturns, benchmarkReturns);
      const vsRandom = monteCarloNull(prices, {
        strategyReturn: testMetrics.totalReturn,
        nTradesObserved: testMetrics.numTrades,
      });
      const returnDistribution = analyzeDistribution(testReturns);

      response.significance = {
        sharpeConfidence,
        vsBenchmark,
        vsRandom,
        returnDistribution,
      };

      // Summary - use local variables directly instead of accessing through response
      const sharpeSig = !sharpeConfidence.ciIncludesZero;
      const beatsBench = vsBenchmark.significantAt05;
      const beatsRandom = vsRandom.significantAt05 ?? false;

      const score = [sharpeSig, beatsBench, beatsRandom].filter(Boolean).length;
      let overallEvidence: string;

      if (score === 3) overallEvidence = 'Strong evidence of genuine edge';
      else if (score === 2) overallEvidence = 'Moderate evidence - warrants further investigation';
      else if (score === 1) overallEvidence = 'Weak evidence - likely noise or overfitting';
      else overallEvidence = 'No statistical evidence of edge over random';

      response.summary = {
        sharpeStatisticallySignificant: sharpeSig,
        beatsBenchmarkSignificantly: beatsBench,
        beatsRandomTrading: beatsRandom,
        overallEvidence,
        testsPassedCount: score,
      };
    }

    return NextResponse.json(response);
  } catch (error) {
    console.error('Backtest error:', error);
    const errorMessage = error instanceof Error ? error.message : 'Unknown error';
    return NextResponse.json(
      { error: `Backtest failed: ${errorMessage}` },
      { status: 500 }
    );
  }
}
