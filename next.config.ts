import type { NextConfig } from 'next';

const nextConfig: NextConfig = {
  reactStrictMode: true,
  // Optimize for Vercel deployment
  poweredByHeader: false,
  // Enable experimental features
  experimental: {
    // Optimize package imports
    optimizePackageImports: ['recharts', 'lucide-react', 'date-fns'],
  },
};

export default nextConfig;
