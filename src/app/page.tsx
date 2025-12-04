"use client";

import { useRouter } from "next/navigation";
import { useEffect } from "react";
import { useSession } from "@/lib/auth-client";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { 
  Calendar, 
  CreditCard, 
  FileText, 
  MessageSquare, 
  Shield, 
  Video,
  CheckCircle,
  ArrowRight,
  Sparkles
} from "lucide-react";

export default function Home() {
  const router = useRouter();
  const { data: session, isPending } = useSession();

  useEffect(() => {
    if (!isPending && session?.user) {
      router.push("/dashboard");
    }
  }, [isPending, session, router]);

  const features = [
    {
      icon: Shield,
      title: "Credentialing Automation",
      description: "RAG-powered document extraction reduces manual application work and streamlines insurance credentialing."
    },
    {
      icon: Video,
      title: "Integrated Telehealth",
      description: "Seamless Zoom and Google Meet integration for secure virtual therapy sessions."
    },
    {
      icon: MessageSquare,
      title: "HIPAA-Compliant Messaging",
      description: "Real-time encrypted messaging with moderation, audit trails, and patient communication."
    },
    {
      icon: Calendar,
      title: "Smart Scheduling",
      description: "Appointment management with availability tracking, waitlists, and automated reminders."
    },
    {
      icon: CreditCard,
      title: "Payment & Claims",
      description: "Stripe integration for payments and Optum API for automated insurance claims submission."
    },
    {
      icon: FileText,
      title: "Lightweight EHR",
      description: "Essential electronic health records with session notes, treatment plans, and secure storage."
    }
  ];

  const benefits = [
    "Self-hosted or cloud multi-tenant deployment",
    "Open-source with extensible integrations",
    "Full HIPAA compliance with audit trails",
    "Automated credentialing and claims processing",
    "Built-in telehealth and secure messaging",
    "Low-cost alternative to Alma and Headway"
  ];

  return (
    <main className="min-h-screen bg-gradient-to-b from-background via-background to-muted/20">
      {/* Navigation */}
      <nav className="border-b bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60">
        <div className="container mx-auto px-4 h-16 flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Sparkles className="h-6 w-6 text-primary" />
            <span className="font-bold text-xl">Bloom</span>
          </div>
          <div className="flex items-center gap-3">
            <Button
              variant="ghost"
              onClick={() => router.push("/sign-in")}
            >
              Sign In
            </Button>
            <Button
              onClick={() => router.push("/sign-up")}
            >
              Get Started
            </Button>
          </div>
        </div>
      </nav>

      {/* Hero Section */}
      <section className="container mx-auto px-4 py-20 md:py-32">
        <div className="max-w-4xl mx-auto text-center space-y-6">
          <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-primary/10 text-primary text-sm font-medium mb-4 animate-fade-in">
            <Sparkles className="h-4 w-4 animate-pulse" />
            <span>Open-Source Therapy Practice Platform</span>
          </div>
          
          <h1 className="text-4xl md:text-6xl font-bold tracking-tight animate-fade-in-up">
            Your Complete Practice
            <br />
            <span className="text-primary">Management Solution</span>
          </h1>
          
          <p className="text-xl text-muted-foreground max-w-2xl mx-auto animate-fade-in-up [animation-delay:200ms]">
            Bloom helps therapists set up and operate a practice from onboarding to billing and patient care. 
            Credentialing automation, telehealth, payments, EHR, and real-time messaging—all in one platform.
          </p>
          
          <div className="flex flex-col sm:flex-row items-center justify-center gap-4 pt-4 animate-fade-in-up [animation-delay:400ms]">
            <Button
              size="lg"
              onClick={() => router.push("/sign-up")}
              className="text-lg px-8 transition-all hover:scale-105"
            >
              Start Your Practice
              <ArrowRight className="ml-2 h-5 w-5 transition-transform group-hover:translate-x-1" />
            </Button>
            <Button
              size="lg"
              variant="outline"
              onClick={() => router.push("/sign-in")}
              className="text-lg px-8 transition-all hover:scale-105"
            >
              Sign In to Dashboard
            </Button>
          </div>
        </div>
      </section>

      {/* Features Grid */}
      <section className="container mx-auto px-4 py-20">
        <div className="max-w-6xl mx-auto">
          <div className="text-center space-y-4 mb-12">
            <h2 className="text-3xl md:text-4xl font-bold">Everything You Need</h2>
            <p className="text-lg text-muted-foreground max-w-2xl mx-auto">
              A complete suite of tools designed for modern therapy practices
            </p>
          </div>
          
          <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6">
            {features.map((feature, index) => (
              <Card 
                key={index} 
                className="border-2 hover:border-primary/50 transition-all duration-300 hover:shadow-lg hover:-translate-y-1 animate-fade-in-up"
                style={{ animationDelay: `${index * 100}ms` }}
              >
                <CardHeader>
                  <div className="h-12 w-12 rounded-lg bg-primary/10 flex items-center justify-center mb-4 transition-transform hover:scale-110">
                    <feature.icon className="h-6 w-6 text-primary" />
                  </div>
                  <CardTitle>{feature.title}</CardTitle>
                  <CardDescription className="text-base">
                    {feature.description}
                  </CardDescription>
                </CardHeader>
              </Card>
            ))}
          </div>
        </div>
      </section>

      {/* Benefits Section */}
      <section className="container mx-auto px-4 py-20 bg-muted/30">
        <div className="max-w-4xl mx-auto">
          <Card className="border-2 animate-fade-in-up hover:shadow-xl transition-shadow duration-300">
            <CardHeader className="text-center">
              <CardTitle className="text-3xl">Why Choose Bloom?</CardTitle>
              <CardDescription className="text-lg">
                Inspired by Alma and Headway, built for complete control and transparency
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="grid md:grid-cols-2 gap-4">
                {benefits.map((benefit, index) => (
                  <div 
                    key={index} 
                    className="flex items-start gap-3 animate-fade-in"
                    style={{ animationDelay: `${index * 75}ms` }}
                  >
                    <CheckCircle className="h-5 w-5 text-primary flex-shrink-0 mt-0.5" />
                    <span className="text-sm">{benefit}</span>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        </div>
      </section>

      {/* CTA Section */}
      <section className="container mx-auto px-4 py-20">
        <div className="max-w-3xl mx-auto text-center space-y-6 animate-fade-in-up">
          <h2 className="text-3xl md:text-4xl font-bold">
            Ready to Transform Your Practice?
          </h2>
          <p className="text-lg text-muted-foreground">
            Join therapists who are streamlining their practice with Bloom. 
            Choose self-hosted for full control or our managed cloud for simplicity.
          </p>
          <div className="flex flex-col sm:flex-row items-center justify-center gap-4 pt-4">
            <Button
              size="lg"
              onClick={() => router.push("/sign-up")}
              className="text-lg px-8 transition-all hover:scale-105 hover:shadow-lg"
            >
              Get Started Free
            </Button>
            <Button
              size="lg"
              variant="outline"
              onClick={() => router.push("/sign-in")}
              className="text-lg px-8 transition-all hover:scale-105"
            >
              Sign In
            </Button>
          </div>
        </div>
      </section>

      {/* Footer */}
      <footer className="border-t mt-20">
        <div className="container mx-auto px-4 py-8">
          <div className="flex flex-col md:flex-row items-center justify-between gap-4">
            <div className="flex items-center gap-2">
              <Sparkles className="h-5 w-5 text-primary" />
              <span className="font-semibold">Bloom</span>
              <span className="text-sm text-muted-foreground">— Therapy Practice Platform</span>
            </div>
            <p className="text-sm text-muted-foreground">
              © 2025 Bloom. Open-source and HIPAA-compliant.
            </p>
          </div>
        </div>
      </footer>
    </main>
  );
}
