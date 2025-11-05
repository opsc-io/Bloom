#!/usr/bin/env node
const fs = require('fs');
const path = require('path');

// Simple checker: scan .tsx files under app/ and pages/ for React hooks or next/navigation
// and warn if they are missing the "use client" directive at top.

const root = process.cwd();
const scanDirs = ['app', 'pages'];
const hookPatterns = ['useState(', 'useEffect(', 'useRouter(', 'useSearchParams(', 'usePathname(', 'useParams('];

function readFile(p) {
  try { return fs.readFileSync(p, 'utf8'); } catch (e) { return null; }
}

let problems = [];

for (const dir of scanDirs) {
  const full = path.join(root, dir);
  if (!fs.existsSync(full)) continue;
  const files = [];
  function walk(d) {
    for (const name of fs.readdirSync(d)) {
      const p = path.join(d, name);
      const stat = fs.statSync(p);
      if (stat.isDirectory()) walk(p);
      else if (p.endsWith('.tsx')) files.push(p);
    }
  }
  walk(full);
  for (const f of files) {
    const code = readFile(f);
    if (!code) continue;
    const usesHook = /useState\(|useEffect\(|useRouter\(|useSearchParams\(|usePathname\(|useParams\(/.test(code);
    if (usesHook) {
      const firstNonEmpty = code.split(/\r?\n/).find(line => line.trim().length>0);
      if (!code.trim().startsWith('"use client"') && !code.trim().startsWith("'use client'")) {
        problems.push(f);
      }
    }
  }
}

if (problems.length) {
  console.error('Files that use client-only hooks but are missing "use client" directive:');
  for (const p of problems) console.error(' -', path.relative(root, p));
  process.exitCode = 2;
} else {
  console.log('No missing "use client" directives found.');
}
