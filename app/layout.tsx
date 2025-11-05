import './globals.css';
import React from 'react';

export const metadata = {
  title: 'Therapy Practice Platform',
  description: 'Platform for therapists to manage practices',
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>
        <div className="min-h-screen bg-base-200">
          <main className="container mx-auto p-4">{children}</main>
        </div>
      </body>
    </html>
  );
}
