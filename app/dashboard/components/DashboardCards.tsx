"use client";
import React from 'react';

export default function DashboardCards() {
  return (
    <div className="space-y-4">
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
        <div className="card bg-base-100 shadow">
          <div className="card-body">
            <h3 className="card-title">Upcoming appointments</h3>
            <p className="text-sm text-muted">No appointments yet.</p>
          </div>
        </div>
        <div className="card bg-base-100 shadow">
          <div className="card-body">
            <h3 className="card-title">Recent activity</h3>
            <p className="text-sm text-muted">No activity yet.</p>
          </div>
        </div>
      </div>
      <div className="card bg-base-100 shadow">
        <div className="card-body">
          <h3 className="card-title">Quick actions</h3>
          <div className="flex gap-2 mt-2">
            <button className="btn btn-primary">New appointment</button>
            <button className="btn btn-outline">Invite provider</button>
          </div>
        </div>
      </div>
    </div>
  );
}
