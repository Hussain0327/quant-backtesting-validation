/**
 * Market Data API Route
 * Fetches historical OHLCV data from Yahoo Finance
 */

import { NextRequest, NextResponse } from 'next/server';
import yahooFinance from 'yahoo-finance2';

export async function GET(request: NextRequest) {
  const searchParams = request.nextUrl.searchParams;
  const ticker = searchParams.get('ticker');
  const start = searchParams.get('start');
  const end = searchParams.get('end');

  // Validate parameters
  if (!ticker || !start || !end) {
    return NextResponse.json(
      { error: 'Missing required parameters: ticker, start, end' },
      { status: 400 }
    );
  }

  // Validate ticker format
  if (!/^[A-Z]{1,5}$/i.test(ticker)) {
    return NextResponse.json(
      { error: 'Invalid ticker format. Use 1-5 letters (e.g., AAPL, MSFT)' },
      { status: 400 }
    );
  }

  try {
    const startDate = new Date(start);
    const endDate = new Date(end);

    // Validate dates
    if (isNaN(startDate.getTime()) || isNaN(endDate.getTime())) {
      return NextResponse.json(
        { error: 'Invalid date format. Use YYYY-MM-DD' },
        { status: 400 }
      );
    }

    if (startDate >= endDate) {
      return NextResponse.json(
        { error: 'Start date must be before end date' },
        { status: 400 }
      );
    }

    // Fetch from Yahoo Finance
    const result = await yahooFinance.historical(ticker.toUpperCase(), {
      period1: startDate,
      period2: endDate,
    });

    if (!result || result.length === 0) {
      return NextResponse.json(
        { error: `No data found for ticker ${ticker.toUpperCase()}` },
        { status: 404 }
      );
    }

    // Transform to our format
    const data = result.map((row) => ({
      date: row.date.toISOString(),
      open: row.open,
      high: row.high,
      low: row.low,
      close: row.adjClose ?? row.close, // Use adjusted close
      volume: row.volume,
    }));

    return NextResponse.json({
      data,
      ticker: ticker.toUpperCase(),
      startDate: start,
      endDate: end,
      dataPoints: data.length,
    });
  } catch (error) {
    console.error('Yahoo Finance error:', error);

    // Handle specific errors
    const errorMessage = error instanceof Error ? error.message : 'Unknown error';

    if (errorMessage.includes('Not Found') || errorMessage.includes('404')) {
      return NextResponse.json(
        { error: `Ticker ${ticker.toUpperCase()} not found` },
        { status: 404 }
      );
    }

    return NextResponse.json(
      { error: `Failed to fetch data: ${errorMessage}` },
      { status: 500 }
    );
  }
}
