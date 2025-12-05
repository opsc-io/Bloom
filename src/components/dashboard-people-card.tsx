"use client";

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { User, Users, MessageCircle } from "lucide-react";

interface Person {
  id: string;
  name: string;
  email: string;
  image?: string | null;
  role?: string;
}

interface DashboardPeopleCardProps {
  people: Person[];
  userRole: string;
}

export function DashboardPeopleCard({ people, userRole }: DashboardPeopleCardProps) {
  const isTherapist = userRole === "THERAPIST";
  const title = isTherapist ? "My Patients" : "My Therapist";
  const description = isTherapist 
    ? "Active patients under your care" 
    : "Your assigned therapist";
  const Icon = isTherapist ? Users : User;

  return (
    <Card className="bg-card/50 backdrop-blur-sm border-border/50 hover:bg-card/60 transition-all hover:shadow-lg flex flex-col h-full">
      <CardHeader className="pb-3">
        <div className="flex items-center justify-between">
          <div className="space-y-1">
            <CardTitle className="text-base font-medium flex items-center gap-2">
              <Icon className="h-4 w-4" />
              {title}
            </CardTitle>
            <CardDescription className="text-xs">{description}</CardDescription>
          </div>
          <Badge variant="secondary" className="text-xs">
            {people.length}
          </Badge>
        </div>
      </CardHeader>
      <CardContent className="flex-1 overflow-hidden">
        {people.length === 0 ? (
          <div className="flex items-center justify-center h-full text-sm text-muted-foreground">
            {isTherapist ? "No patients assigned" : "No therapist assigned"}
          </div>
        ) : (
          <div className="h-full overflow-y-auto pr-2 space-y-2 scrollbar-thin scrollbar-thumb-muted scrollbar-track-transparent">
            {people.map((person) => (
              <div
                key={person.id}
                className="flex items-center gap-3 p-2 rounded-lg hover:bg-muted/50 transition-colors group"
              >
                <Avatar className="h-9 w-9">
                  <AvatarImage src={person.image || ""} />
                  <AvatarFallback className="text-xs">
                    {person.name
                      .split(" ")
                      .map((n) => n[0])
                      .join("")
                      .toUpperCase()}
                  </AvatarFallback>
                </Avatar>
                <div className="flex-1 min-w-0">
                  <p className="text-sm font-medium truncate">{person.name}</p>
                  <p className="text-xs text-muted-foreground truncate">{person.email}</p>
                </div>
                <Button
                  size="sm"
                  variant="ghost"
                  className="h-8 w-8 p-0 opacity-0 group-hover:opacity-100 transition-opacity"
                  onClick={() => {
                    // Navigate to messages page with message query
                    window.location.href = `/messages?message=${person.id}`;
                  }}
                >
                  <MessageCircle className="h-4 w-4" />
                </Button>
              </div>
            ))}
          </div>
        )}
      </CardContent>
    </Card>
  );
}
