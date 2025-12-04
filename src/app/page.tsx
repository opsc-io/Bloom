"use client";

import { useRouter } from "next/navigation";
import { useEffect } from "react";
import { useSession } from "@/lib/auth-client";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import Image from "next/image";
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
    <main className="min-h-screen bg-background">
      {/* Navigation */}
      <nav className="border-b bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60 sticky top-0 z-50">
        <div className="container mx-auto px-4 h-16 flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Image src="/logo.svg" alt="Bloom Logo" width={120} height={40} className="h-8 w-auto" />
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

      {/* Hero Section with Unique Layout */}
      <section className="relative overflow-hidden">
        {/* Gradient Background Effects */}
        <div className="absolute inset-0 -z-10">
          <div className="absolute top-20 left-10 w-72 h-72 bg-primary/20 rounded-full blur-3xl animate-pulse"></div>
          <div className="absolute bottom-20 right-10 w-96 h-96 bg-primary/10 rounded-full blur-3xl animate-pulse [animation-delay:2s]"></div>
        </div>

        <div className="container mx-auto px-4 py-20 md:py-28">
          <div className="grid lg:grid-cols-2 gap-12 items-center">
            {/* Left Content */}
            <div className="space-y-8">
              <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full border border-primary/20 bg-primary/5 text-primary text-sm font-medium animate-fade-in">
                <Sparkles className="h-4 w-4 animate-pulse" />
                <span>Open-Source Therapy Practice Platform</span>
              </div>
              
              <h1 className="text-5xl md:text-7xl font-bold tracking-tight animate-fade-in-up leading-tight">
                Transform Your
                <br />
                <span className="text-primary relative inline-block">
                  Practice
                  <svg className="absolute -bottom-2 left-0 w-full" viewBox="0 0 200 12" fill="none" xmlns="http://www.w3.org/2000/svg">
                    <path d="M2 10C50 2 150 2 198 10" stroke="currentColor" strokeWidth="3" strokeLinecap="round"/>
                  </svg>
                </span>
                <br />
                Experience
              </h1>
              
              <p className="text-xl text-muted-foreground animate-fade-in-up [animation-delay:200ms] leading-relaxed">
                All-in-one platform for therapists and patients. Credentialing automation, telehealth, secure messaging, 
                EHR, and billing—designed with privacy and compliance at its core.
              </p>
              
              <div className="flex flex-wrap gap-4 animate-fade-in-up [animation-delay:400ms]">
                <Button
                  size="lg"
                  onClick={() => router.push("/sign-up")}
                  className="text-lg px-8 h-12 transition-all hover:scale-105 hover:shadow-xl group"
                >
                  Start Free Trial
                  <ArrowRight className="ml-2 h-5 w-5 transition-transform group-hover:translate-x-1" />
                </Button>
                <Button
                  size="lg"
                  variant="outline"
                  onClick={() => router.push("/sign-in")}
                  className="text-lg px-8 h-12 transition-all hover:scale-105"
                >
                  View Demo
                </Button>
              </div>
            </div>

            {/* Right Visual Element - Feature Cards Stack */}
            <div className="relative h-[500px] animate-fade-in-up [animation-delay:300ms] hidden lg:block">
              <Card className="absolute top-0 right-0 w-80 border-2 shadow-2xl rotate-3 hover:rotate-0 transition-transform duration-300 bg-background">
                <CardHeader>
                  <div className="h-12 w-12 rounded-lg bg-primary/10 flex items-center justify-center mb-2">
                    <Shield className="h-6 w-6 text-primary" />
                  </div>
                  <CardTitle>HIPAA Compliant</CardTitle>
                  <CardDescription>End-to-end encrypted messaging and secure data storage</CardDescription>
                </CardHeader>
              </Card>
              
              <Card className="absolute top-32 right-12 w-80 border-2 shadow-2xl -rotate-2 hover:rotate-0 transition-transform duration-300 bg-background">
                <CardHeader>
                  <div className="h-12 w-12 rounded-lg bg-primary/10 flex items-center justify-center mb-2">
                    <Calendar className="h-6 w-6 text-primary" />
                  </div>
                  <CardTitle>Smart Scheduling</CardTitle>
                  <CardDescription>Automated appointments with waitlist management</CardDescription>
                </CardHeader>
              </Card>
              
              <Card className="absolute top-64 right-6 w-80 border-2 shadow-2xl rotate-1 hover:rotate-0 transition-transform duration-300 bg-background">
                <CardHeader>
                  <div className="h-12 w-12 rounded-lg bg-primary/10 flex items-center justify-center mb-2">
                    <CreditCard className="h-6 w-6 text-primary" />
                  </div>
                  <CardTitle>Automated Billing</CardTitle>
                  <CardDescription>Stripe payments and Optum claims integration</CardDescription>
                </CardHeader>
              </Card>
            </div>
          </div>
        </div>
      </section>

      {/* Features Section - Bento Grid Style */}
      <section className="container mx-auto px-4 py-20 bg-muted/30">
        <div className="max-w-7xl mx-auto">
          <div className="text-center space-y-4 mb-16">
            <h2 className="text-4xl md:text-5xl font-bold">Built for Modern Practices</h2>
            <p className="text-lg text-muted-foreground max-w-2xl mx-auto">
              Everything you need to run a therapy practice, minus the complexity
            </p>
          </div>
          
          {/* Bento Grid */}
          <div className="grid md:grid-cols-3 gap-6">
            {/* Large Feature - Spans 2 columns */}
            <Card className="md:col-span-2 md:row-span-2 border-2 hover:border-primary/50 transition-all duration-300 hover:shadow-2xl overflow-hidden group relative">
              <div className="absolute inset-0 bg-gradient-to-br from-primary/5 to-transparent opacity-0 group-hover:opacity-100 transition-opacity"></div>
              <CardHeader className="relative z-10">
                <div className="h-16 w-16 rounded-xl bg-primary/10 flex items-center justify-center mb-6 group-hover:scale-110 transition-transform">
                  <Shield className="h-8 w-8 text-primary" />
                </div>
                <CardTitle className="text-2xl mb-3">Credentialing Automation</CardTitle>
                <CardDescription className="text-base leading-relaxed">
                  Revolutionary RAG-powered document extraction that reads credentialing requirements and auto-fills 
                  applications. What used to take hours now takes minutes. Built-in tracking for renewals and 
                  expirations across all insurance providers.
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="flex flex-wrap gap-2 mt-4">
                  <span className="px-3 py-1 rounded-full bg-primary/10 text-xs font-medium">Auto-extraction</span>
                  <span className="px-3 py-1 rounded-full bg-primary/10 text-xs font-medium">Multi-provider</span>
                  <span className="px-3 py-1 rounded-full bg-primary/10 text-xs font-medium">Smart tracking</span>
                </div>
              </CardContent>
            </Card>

            {/* Smaller Features */}
            {[
              { icon: Video, title: "Telehealth", desc: "Zoom & Meet integration", tags: ["Video", "Recording"] },
              { icon: MessageSquare, title: "Secure Messaging", desc: "HIPAA-compliant chat", tags: ["Encrypted", "Audit trails"] },
              { icon: Calendar, title: "Scheduling", desc: "Smart appointment management", tags: ["Waitlists", "Reminders"] },
              { icon: CreditCard, title: "Payments", desc: "Stripe + Optum claims", tags: ["Auto-billing", "Claims"] },
              { icon: FileText, title: "EHR", desc: "Lightweight records system", tags: ["Notes", "Plans"] },
            ].map((feature, index) => (
              <Card 
                key={index} 
                className="border-2 hover:border-primary/50 transition-all duration-300 hover:shadow-xl hover:-translate-y-1 group relative overflow-hidden"
                style={{ animationDelay: `${index * 100}ms` }}
              >
                <div className="absolute inset-0 bg-gradient-to-br from-primary/5 to-transparent opacity-0 group-hover:opacity-100 transition-opacity"></div>
                <CardHeader className="relative z-10">
                  <div className="h-12 w-12 rounded-lg bg-primary/10 flex items-center justify-center mb-4 group-hover:scale-110 transition-transform">
                    <feature.icon className="h-6 w-6 text-primary" />
                  </div>
                  <CardTitle className="text-lg">{feature.title}</CardTitle>
                  <CardDescription className="text-sm">{feature.desc}</CardDescription>
                  <div className="flex flex-wrap gap-1.5 mt-3">
                    {feature.tags.map((tag, i) => (
                      <span key={i} className="px-2 py-0.5 rounded-full bg-primary/10 text-xs">{tag}</span>
                    ))}
                  </div>
                </CardHeader>
              </Card>
            ))}
          </div>
        </div>
      </section>

      {/* CTA Section - Bold Design */}
      <section className="relative overflow-hidden py-20">
        {/* Background Pattern */}
        <div className="absolute inset-0 -z-10 bg-gradient-to-br from-primary/10 via-background to-primary/5">
          <div className="absolute inset-0" style={{
            backgroundImage: 'radial-gradient(circle at 2px 2px, rgb(var(--primary) / 0.1) 1px, transparent 0)',
            backgroundSize: '40px 40px'
          }}></div>
        </div>

        <div className="container mx-auto px-4">
          <div className="max-w-4xl mx-auto text-center space-y-8 animate-fade-in-up">
            <h2 className="text-4xl md:text-6xl font-bold leading-tight">
              Ready to Transform
              <br />
              Your Practice?
            </h2>
            
            <p className="text-xl text-muted-foreground max-w-2xl mx-auto">
              Join hundreds of therapists and patients who've modernized their practice with Bloom. 
              No credit card required to start.
            </p>
            
            <div className="flex flex-col sm:flex-row items-center justify-center gap-4 pt-4">
              <Button
                size="lg"
                onClick={() => router.push("/sign-up")}
                className="text-lg px-10 h-14 transition-all hover:scale-105 hover:shadow-2xl group"
              >
                Get Started Free
                <ArrowRight className="ml-2 h-5 w-5 transition-transform group-hover:translate-x-1" />
              </Button>
              <Button
                size="lg"
                variant="outline"
                onClick={() => router.push("/sign-in")}
                className="text-lg px-10 h-14 transition-all hover:scale-105 bg-background"
              >
                Sign In
              </Button>
            </div>

            <p className="text-sm text-muted-foreground pt-4">
              ✓ No credit card required  ✓ 14-day free trial  ✓ Cancel anytime
            </p>
          </div>
        </div>
      </section>

      {/* Footer */}
      <footer className="border-t mt-20">
        <div className="container mx-auto px-4 py-8">
          <div className="flex flex-col md:flex-row items-center justify-between gap-4">
            <div className="flex items-center gap-2">
              <Image src="/logo.svg" alt="Bloom Logo" width={100} height={33} className="h-6 w-auto" />
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
