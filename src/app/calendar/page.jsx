"use client";
import { Calendar } from "@/components/ui/calendar";
import * as React from "react";

export default function Page() {
  const [date, setDate] = React.useState(() => {
    return new Date(2025, 5, 12);
  });

  return (
    <div> {/* Parent with explicit height; adjust to your layout */}
      <Calendar
        mode="single"
        selected={date}
        onSelect={setDate}
        className="h-full w-full rounded-lg border [--cell-size:2rem] md:[--cell-size:2.5rem]" 
        buttonVariant="ghost"
      />
    </div>
  );
}