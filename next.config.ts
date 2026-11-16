import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  /* config options here */
  cacheComponents: true,
  partialPrefetching: true,
  async rewrites() {
    if (process.env.NODE_ENV === "development") {
      return [
        {
          source: "/api/:path*",
          destination: "http://127.0.0.1:8001/api/:path*",
        },
        {
          source: "/docs",
          destination: "http://127.0.0.1:8001/docs",
        },
        {
          source: "/openapi.json",
          destination: "http://127.0.0.1:8001/openapi.json",
        },
      ];
    }
    return [];
  },
};

export default nextConfig;
