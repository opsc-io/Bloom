// Only enable autoprefixer for non-test environments to avoid requiring
// the dev dependency during isolated unit test runs (Vitest + Vite).
module.exports = () => {
  const plugins = { tailwindcss: {} };
  if (process.env.NODE_ENV !== 'test') {
    // In normal dev/build flows autoprefixer should be present in devDeps.
    plugins.autoprefixer = {};
  }
  return { plugins };
};
