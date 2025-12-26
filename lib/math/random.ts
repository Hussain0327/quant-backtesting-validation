/**
 * Random number utilities
 * Ported from numpy.random
 */

/**
 * Generate a random integer in range [min, max) (exclusive of max)
 * Equivalent to np.random.randint(min, max)
 */
export function randInt(min: number, max: number): number {
  return Math.floor(Math.random() * (max - min)) + min;
}

/**
 * Generate array of random integers
 */
export function randInts(min: number, max: number, size: number): number[] {
  const result: number[] = [];
  for (let i = 0; i < size; i++) {
    result.push(randInt(min, max));
  }
  return result;
}

/**
 * Randomly select n items from array (with replacement)
 * Equivalent to np.random.choice(arr, size=n, replace=True)
 */
export function choice<T>(arr: T[], size: number): T[] {
  const result: T[] = [];
  for (let i = 0; i < size; i++) {
    const idx = randInt(0, arr.length);
    result.push(arr[idx]);
  }
  return result;
}

/**
 * Randomly select n items from array (without replacement)
 */
export function sample<T>(arr: T[], size: number): T[] {
  if (size >= arr.length) return [...arr];

  const copy = [...arr];
  const result: T[] = [];

  for (let i = 0; i < size; i++) {
    const idx = randInt(0, copy.length);
    result.push(copy[idx]);
    copy.splice(idx, 1);
  }

  return result;
}

/**
 * Shuffle array in-place (Fisher-Yates algorithm)
 * Equivalent to np.random.shuffle
 */
export function shuffle<T>(arr: T[]): T[] {
  const result = [...arr];
  for (let i = result.length - 1; i > 0; i--) {
    const j = randInt(0, i + 1);
    [result[i], result[j]] = [result[j], result[i]];
  }
  return result;
}

/**
 * Generate a random float in range [0, 1)
 */
export function random(): number {
  return Math.random();
}

/**
 * Generate array of random floats in range [0, 1)
 */
export function randoms(size: number): number[] {
  const result: number[] = [];
  for (let i = 0; i < size; i++) {
    result.push(Math.random());
  }
  return result;
}

/**
 * Generate a random float in range [min, max)
 */
export function uniform(min: number, max: number): number {
  return Math.random() * (max - min) + min;
}

/**
 * Generate random samples from standard normal distribution
 * Using Box-Muller transform
 */
export function randn(): number {
  const u1 = Math.random();
  const u2 = Math.random();
  return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
}

/**
 * Generate array of random normal samples
 */
export function randns(size: number): number[] {
  const result: number[] = [];
  for (let i = 0; i < size; i++) {
    result.push(randn());
  }
  return result;
}

/**
 * Seeded random number generator (for reproducibility)
 * Uses xorshift128+ algorithm
 */
export class SeededRandom {
  private state: [number, number];

  constructor(seed: number = Date.now()) {
    // Initialize state from seed
    this.state = [seed, seed ^ 0x49616e42];
  }

  private next(): number {
    let [s0, s1] = this.state;
    const result = s0 + s1;

    s1 ^= s0;
    this.state[0] = ((s0 << 55) | (s0 >>> 9)) ^ s1 ^ (s1 << 14);
    this.state[1] = (s1 << 36) | (s1 >>> 28);

    // Convert to [0, 1) range
    return (result >>> 0) / 0xffffffff;
  }

  random(): number {
    return this.next();
  }

  randInt(min: number, max: number): number {
    return Math.floor(this.next() * (max - min)) + min;
  }

  shuffle<T>(arr: T[]): T[] {
    const result = [...arr];
    for (let i = result.length - 1; i > 0; i--) {
      const j = this.randInt(0, i + 1);
      [result[i], result[j]] = [result[j], result[i]];
    }
    return result;
  }
}
