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
        {/* Animated Gradient Background Effects */}
        <div className="absolute inset-0 -z-10">
          <div className="absolute top-20 left-10 w-72 h-72 bg-gradient-to-r from-purple-500/10 via-primary/10 to-blue-500/10 rounded-full opacity-60"></div>
          <div className="absolute bottom-20 right-10 w-96 h-96 bg-gradient-to-l from-pink-500/10 via-primary/10 to-orange-500/10 rounded-full opacity-60"></div>
        </div>

        <div className="container mx-auto px-4 py-20 md:py-28">
          <div className="grid lg:grid-cols-2 gap-12 items-center">
            {/* Left Content */}
            <div className="space-y-8">
              <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full border border-primary/20 bg-primary/5 text-primary text-sm font-medium animate-fade-in">
                <Sparkles className="h-4 w-4" />
                <span className="font-semibold">Open-Source Therapy Practice Platform</span>
              </div>
              
              <h1 className="text-5xl md:text-7xl font-bold tracking-tight animate-fade-in-up leading-tight">
                <span className="text-foreground">
                  Transform Your
                </span>
                <br />
                <span className="bg-gradient-to-r from-blue-600 to-cyan-600 bg-clip-text text-transparent relative inline-block">
                  Practice
                  <svg className="absolute -bottom-2 left-0 w-full" viewBox="0 0 200 12" fill="none" xmlns="http://www.w3.org/2000/svg">
                    <path d="M2 10C50 2 150 2 198 10" stroke="url(#blue-gradient)" strokeWidth="3" strokeLinecap="round"/>
                    <defs>
                      <linearGradient id="blue-gradient" x1="0%" y1="0%" x2="100%" y2="0%">
                        <stop offset="0%" stopColor="rgb(37, 99, 235)" />
                        <stop offset="100%" stopColor="rgb(8, 145, 178)" />
                      </linearGradient>
                    </defs>
                  </svg>
                </span>
                <br />
                <span className="text-foreground">
                  Experience
                </span>
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
                  <ArrowRight className="ml-2 h-5 w-5 transition-transform group-hover:translate-x-1 group-hover:scale-110" />
                </Button>
                <Button
                  size="lg"
                  variant="outline"
                  onClick={() => router.push("/sign-in")}
                  className="text-lg px-8 h-12 transition-all hover:scale-105 border-2 hover:bg-primary/5"
                >
                  View Demo
                </Button>
              </div>
            </div>

            {/* Right Visual Element - Feature Cards Stack with 3D effect */}
            <div className="relative h-[500px] animate-fade-in-up [animation-delay:300ms] hidden lg:block">
              <Card className="absolute top-0 right-0 w-80 border-2 shadow-2xl rotate-3 hover:rotate-0 transition-all duration-500 bg-gradient-to-br from-background via-background to-purple-500/5 hover:shadow-purple-500/20 hover:-translate-y-2">
                <CardHeader className="-mt-4">
                  <div className="h-12 w-12 rounded-lg bg-gradient-to-br from-purple-500/20 to-pink-500/20 flex items-center justify-center mb-2">
                    <MessageSquare className="h-6 w-6 text-purple-600" />
                  </div>
                  <CardTitle className="bg-gradient-to-r from-purple-600 to-pink-600 bg-clip-text text-transparent leading-relaxed pb-1">Secure Messaging</CardTitle>
                  <CardDescription>HIPAA-compliant encrypted messaging with audit trails</CardDescription>
                </CardHeader>
              </Card>
              
              <Card className="absolute top-32 right-12 w-80 border-2 shadow-2xl -rotate-2 hover:rotate-0 transition-all duration-500 bg-gradient-to-br from-background via-background to-emerald-500/5 hover:shadow-emerald-500/20 hover:-translate-y-2">
                <CardHeader className="-mt-4">
                  <div className="h-12 w-12 rounded-lg bg-gradient-to-br from-emerald-500/20 to-teal-500/20 flex items-center justify-center mb-2">
                    <CreditCard className="h-6 w-6 text-emerald-600" />
                  </div>
                  <CardTitle className="bg-gradient-to-r from-emerald-600 to-teal-600 bg-clip-text text-transparent leading-relaxed pb-1">Payments & Billing</CardTitle>
                  <CardDescription>Stripe payments and automated insurance claims</CardDescription>
                </CardHeader>
              </Card>
              
              <Card className="absolute top-64 right-6 w-80 border-2 shadow-2xl rotate-1 hover:rotate-0 transition-all duration-500 bg-gradient-to-br from-background via-background to-blue-500/5 hover:shadow-blue-500/20 hover:-translate-y-2">
                <CardHeader className="-mt-4">
                  <div className="h-12 w-12 rounded-lg bg-gradient-to-br from-blue-500/20 to-cyan-500/20 flex items-center justify-center mb-2">
                    <Calendar className="h-6 w-6 text-blue-600" />
                  </div>
                  <CardTitle className="bg-gradient-to-r from-blue-600 to-cyan-600 bg-clip-text text-transparent leading-relaxed pb-1">Smart Scheduling</CardTitle>
                  <CardDescription>Automated appointments with waitlist management</CardDescription>
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
            {/* Scheduling - Large Feature */}
            <Card className="md:col-span-2 md:row-span-2 border-2 hover:border-blue-500/50 transition-all duration-300 hover:shadow-2xl overflow-hidden group relative bg-gradient-to-br from-background via-blue-500/5 to-cyan-500/5">
              <div className="absolute inset-0 bg-gradient-to-br from-blue-500/5 via-cyan-500/5 to-blue-500/5 opacity-0 group-hover:opacity-100 transition-opacity duration-500"></div>
              <CardHeader className="relative z-10">
                <div className="h-16 w-16 rounded-xl bg-gradient-to-br from-blue-500/20 via-cyan-500/20 to-blue-500/20 flex items-center justify-center mb-6 group-hover:scale-110 transition-transform duration-500">
                  <Calendar className="h-8 w-8 text-blue-600" />
                </div>
                <CardTitle className="text-2xl mb-3 bg-gradient-to-r from-blue-600 via-cyan-600 to-blue-600 bg-clip-text text-transparent leading-relaxed pb-1">Smart Scheduling</CardTitle>
                <CardDescription className="text-base leading-relaxed">
                  Intelligent appointment management with automated waitlists, reminders, and availability tracking. 
                  Integrates seamlessly with your calendar and sends automatic notifications to reduce no-shows.
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="flex flex-wrap gap-2 mt-4">
                  <span className="px-3 py-1 rounded-full bg-gradient-to-r from-blue-500/10 to-cyan-500/10 text-xs font-medium border border-blue-500/20 animate-fade-in hover:scale-105 transition-transform">Waitlists</span>
                  <span className="px-3 py-1 rounded-full bg-gradient-to-r from-cyan-500/10 to-blue-500/10 text-xs font-medium border border-cyan-500/20 animate-fade-in [animation-delay:100ms] hover:scale-105 transition-transform">Auto-reminders</span>
                  <span className="px-3 py-1 rounded-full bg-gradient-to-r from-blue-500/10 to-cyan-500/10 text-xs font-medium border border-blue-500/20 animate-fade-in [animation-delay:200ms] hover:scale-105 transition-transform">Calendar sync</span>
                </div>
              </CardContent>
            </Card>

            {/* Payments - Medium Feature */}
            <Card className="border-2 hover:border-emerald-500/50 transition-all duration-300 hover:shadow-xl hover:-translate-y-1 group relative overflow-hidden bg-gradient-to-br from-background to-emerald-500/5">
              <div className="absolute inset-0 bg-gradient-to-br from-emerald-500/5 to-transparent opacity-0 group-hover:opacity-100 transition-opacity"></div>
              <CardHeader className="relative z-10">
                <div className="h-12 w-12 rounded-lg bg-gradient-to-br from-emerald-500/20 to-teal-500/20 flex items-center justify-center mb-4 group-hover:scale-110 transition-transform">
                  <CreditCard className="h-6 w-6 text-emerald-600" />
                </div>
                <CardTitle className="text-lg bg-gradient-to-r from-emerald-600 to-teal-600 bg-clip-text text-transparent leading-relaxed pb-1">Payments & Billing</CardTitle>
                <CardDescription className="text-sm">Stripe integration for seamless payments and automated insurance claims through Optum API</CardDescription>
                <div className="flex flex-wrap gap-1.5 mt-3">
                  <span className="px-2 py-0.5 rounded-full bg-emerald-500/10 text-xs border border-emerald-500/20">Auto-billing</span>
                  <span className="px-2 py-0.5 rounded-full bg-emerald-500/10 text-xs border border-emerald-500/20">Claims</span>
                </div>
              </CardHeader>
            </Card>

            {/* Secure Messaging - Medium Feature */}
            <Card className="border-2 hover:border-purple-500/50 transition-all duration-300 hover:shadow-xl hover:-translate-y-1 group relative overflow-hidden bg-gradient-to-br from-background to-purple-500/5">
              <div className="absolute inset-0 bg-gradient-to-br from-purple-500/5 to-transparent opacity-0 group-hover:opacity-100 transition-opacity"></div>
              <CardHeader className="relative z-10">
                <div className="h-12 w-12 rounded-lg bg-gradient-to-br from-purple-500/20 to-pink-500/20 flex items-center justify-center mb-4 group-hover:scale-110 transition-transform">
                  <MessageSquare className="h-6 w-6 text-purple-600" />
                </div>
                <CardTitle className="text-lg bg-gradient-to-r from-purple-600 to-pink-600 bg-clip-text text-transparent leading-relaxed pb-1">Secure Messaging</CardTitle>
                <CardDescription className="text-sm">HIPAA-compliant real-time chat with end-to-end encryption and complete audit trails</CardDescription>
                <div className="flex flex-wrap gap-1.5 mt-3">
                  <span className="px-2 py-0.5 rounded-full bg-purple-500/10 text-xs border border-purple-500/20">Encrypted</span>
                  <span className="px-2 py-0.5 rounded-full bg-purple-500/10 text-xs border border-purple-500/20">Audit trails</span>
                </div>
              </CardHeader>
            </Card>
          </div>

          {/* Coming Soon Section */}
          <div className="mt-16">
            <div className="text-center mb-8">
              <h3 className="text-2xl font-bold">More Features on the Way</h3>
              <p className="text-muted-foreground mt-2">Powerful tools currently in development</p>
            </div>

            <div className="grid md:grid-cols-3 gap-6">
              {[
                { icon: Shield, title: "Credentialing Automation", desc: "RAG-powered document extraction", tags: ["Auto-extraction", "Multi-provider"] },
                { icon: Video, title: "Telehealth", desc: "Zoom & Google Meet integration", tags: ["Video", "Recording"] },
                { icon: FileText, title: "EHR", desc: "Lightweight electronic health records", tags: ["Notes", "Plans"] },
              ].map((feature, index) => (
                <Card 
                  key={index} 
                  className="border-2 border-dashed border-muted-foreground/30 hover:border-primary/30 transition-all duration-300 group relative overflow-hidden opacity-60 hover:opacity-100"
                  style={{ animationDelay: `${index * 100}ms` }}
                >
                  <CardHeader className="relative z-10">
                    <div className="h-12 w-12 rounded-lg bg-muted flex items-center justify-center mb-4">
                      <feature.icon className="h-6 w-6 text-muted-foreground" />
                    </div>
                    <CardTitle className="text-lg text-muted-foreground">{feature.title}</CardTitle>
                    <CardDescription className="text-sm">{feature.desc}</CardDescription>
                    <div className="flex flex-wrap gap-1.5 mt-3">
                      {feature.tags.map((tag, i) => (
                        <span key={i} className="px-2 py-0.5 rounded-full bg-muted text-xs text-muted-foreground">{tag}</span>
                      ))}
                    </div>
                  </CardHeader>
                </Card>
              ))}
            </div>
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
              Join hundreds of therapists and patients who&apos;ve modernized their practice with Bloom. 
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
