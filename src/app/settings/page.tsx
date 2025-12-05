"use client";

import { AppSidebar } from "@/components/app-sidebar";
import {
  Breadcrumb,
  BreadcrumbItem,
  BreadcrumbLink,
  BreadcrumbList,
  BreadcrumbPage,
  BreadcrumbSeparator,
} from "@/components/ui/breadcrumb";
import { Separator } from "@/components/ui/separator";
import {
  SidebarInset,
  SidebarProvider,
  SidebarTrigger,
} from "@/components/ui/sidebar";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Button } from "@/components/ui/button";
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar";
import { Textarea } from "@/components/ui/textarea";

import { useRouter } from "next/navigation";
import { useSession } from "@/lib/auth-client";
import { useEffect, useState, ChangeEvent } from "react";
import { Upload } from "lucide-react";

export default function ProfilePage() {
  const router = useRouter();
  const { data: session, isPending } = useSession();
  const [isLoadingProfile, setIsLoadingProfile] = useState(true);
  const [isSavingProfile, setIsSavingProfile] = useState(false);
  const [isSavingPassword, setIsSavingPassword] = useState(false);
  const [profileError, setProfileError] = useState<string | null>(null);
  const [passwordError, setPasswordError] = useState<string | null>(null);
  const [profileSaved, setProfileSaved] = useState(false);
  const [passwordSaved, setPasswordSaved] = useState(false);
  const [image, setImage] = useState("");
  const [isUploading, setIsUploading] = useState(false);
  const [uploadError, setUploadError] = useState<string | null>(null);
  const [allowPasswordChange, setAllowPasswordChange] = useState(true);
  const [linkedAccounts, setLinkedAccounts] = useState<Array<{ providerId: string }>>([]);
  
  // Profile form state
  const [firstname, setFirstname] = useState("");
  const [lastname, setLastname] = useState("");
  const [email, setEmail] = useState("");
  const [bio, setBio] = useState("");
  
  // Password form state
  const [currentPassword, setCurrentPassword] = useState("");
  const [newPassword, setNewPassword] = useState("");
  const [confirmPassword, setConfirmPassword] = useState("");

  useEffect(() => {
    if (!isPending && !session?.user) {
      router.push("/sign-in");
    }
    
    if (session?.user) {
      setFirstname(session.user.firstname || "");
      setLastname(session.user.lastname || "");
      setEmail(session.user.email || "");
      setImage(session.user.image || "");
    }
  }, [isPending, session, router]);

  useEffect(() => {
    if (isPending || !session?.user) return;
    let cancelled = false;

    const loadProfile = async () => {
      setIsLoadingProfile(true);
      setProfileError(null);
      try {
        const res = await fetch("/api/user/settings");
        if (!res.ok) {
          throw new Error("Failed to fetch profile");
        }
        const data = await res.json();
        if (cancelled) return;
        setFirstname(data?.user?.firstname ?? "");
        setLastname(data?.user?.lastname ?? "");
        setEmail(data?.user?.email ?? "");
        setBio(data?.user?.bio ?? "");
        setImage(data?.user?.image ?? "");
        setAllowPasswordChange(Boolean(data?.allowPasswordChange));
        setLinkedAccounts(data?.accounts ?? []);
      } catch (err) {
        if (!cancelled) {
          setProfileError("Unable to load profile right now. Please try again.");
        }
      } finally {
        if (!cancelled) {
          setIsLoadingProfile(false);
        }
      }
    };

    loadProfile();
    return () => {
      cancelled = true;
    };
  }, [isPending, session]);

  const handleProfileSave = async (overrides?: Partial<{ firstname: string; lastname: string; bio: string; image: string }>) => {
    setProfileError(null);
    setProfileSaved(false);
    setIsSavingProfile(true);
    try {
      const response = await fetch("/api/user/settings", {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          firstname,
          lastname,
          bio,
          image,
          ...overrides,
        }),
      });

      if (!response.ok) {
        const body = await response.json().catch(() => ({}));
        throw new Error(body.error || "Failed to update profile");
      }

      setProfileSaved(true);
      // Refresh session
      router.refresh();
    } catch (error) {
      console.error("Error updating profile:", error);
      setProfileError(
        error instanceof Error
          ? error.message
          : "Error updating profile. Please try again."
      );
    } finally {
      setIsSavingProfile(false);
    }
  };

  const handleAvatarUpload = async (e: ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    setUploadError(null);
    setIsUploading(true);
    setProfileSaved(false);

    try {
      const formData = new FormData();
      formData.append("file", file);

      const res = await fetch("/api/upload", {
        method: "POST",
        body: formData,
      });

      if (!res.ok) {
        const body = await res.json().catch(() => ({}));
        throw new Error(body.error || "Upload failed");
      }

      const data = await res.json();
      setImage(data.url);
      // Persist the image immediately so the user doesn't need to click save again
      await handleProfileSave({ image: data.url });
    } catch (err) {
      setUploadError(
        err instanceof Error ? err.message : "Upload failed. Please try again."
      );
    } finally {
      setIsUploading(false);
      // Reset the input value so the same file can be reselected if needed
      e.target.value = "";
    }
  };

  const handlePasswordChange = async () => {
    if (newPassword !== confirmPassword) {
      alert("New passwords don't match");
      return;
    }

    setPasswordError(null);
    setPasswordSaved(false);
    setIsSavingPassword(true);
    try {
      const response = await fetch("/api/user/password", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          currentPassword,
          newPassword,
        }),
      });

      if (!response.ok) {
          const body = await response.json().catch(() => ({}));
          throw new Error(body.error || "Failed to change password");
      }

      setCurrentPassword("");
      setNewPassword("");
      setConfirmPassword("");
      setPasswordSaved(true);
    } catch (error) {
      console.error("Error changing password:", error);
      setPasswordError(
        error instanceof Error
          ? error.message
          : "Error changing password. Please try again."
      );
    } finally {
      setIsSavingPassword(false);
    }
  };

  if (isPending)
    return <p className="text-center mt-8 text-white">Loading...</p>;
  if (!session?.user)
    return <p className="text-center mt-8 text-white">Redirecting...</p>;

  const { user } = session;

  return (
    <SidebarProvider>
      <AppSidebar user={user} />
      <SidebarInset>
        <header className="flex h-16 shrink-0 items-center gap-2 border-b px-4">
          <SidebarTrigger className="-ml-1" />
          <Separator orientation="vertical" className="mr-2 h-4" />
          <Breadcrumb>
            <BreadcrumbList>
              <BreadcrumbItem className="hidden md:block">
                <BreadcrumbLink href="/dashboard">Dashboard</BreadcrumbLink>
              </BreadcrumbItem>
              <BreadcrumbSeparator className="hidden md:block" />
              <BreadcrumbItem>
                <BreadcrumbPage>Settings</BreadcrumbPage>
              </BreadcrumbItem>
            </BreadcrumbList>
          </Breadcrumb>
        </header>

        <div className="flex flex-1 flex-col gap-4 p-4 pt-6">
          <div className="mx-auto w-full max-w-6xl">
            <div className="space-y-6">
              <div>
                <h1 className="text-3xl font-bold tracking-tight">Settings</h1>
                <p className="text-muted-foreground">
                  Manage your account settings and preferences
                </p>
              </div>

              <Tabs defaultValue="profile" className="space-y-4">
                <TabsList>
                  <TabsTrigger value="profile">Profile</TabsTrigger>
                  <TabsTrigger value="account">Account</TabsTrigger>
                </TabsList>

                <TabsContent value="profile" className="space-y-4">
                  <Card>
                    <CardHeader>
                      <CardTitle>Profile Information</CardTitle>
                      <CardDescription>
                        Update your profile information and how others see you
                      </CardDescription>
                    </CardHeader>
                    <CardContent className="space-y-6">
                      {/* Avatar Upload */}
                      <div className="flex items-center gap-4">
                        <Avatar className="h-20 w-20">
                          <AvatarImage src={image || user.image || ""} />
                          <AvatarFallback className="text-lg">
                            {user.firstname?.[0]}{user.lastname?.[0]}
                          </AvatarFallback>
                        </Avatar>
                        <div>
                          <input
                            id="avatar-upload"
                            type="file"
                            accept="image/*"
                            className="hidden"
                            onChange={handleAvatarUpload}
                            disabled={isUploading}
                          />
                          <Button asChild variant="outline" size="sm" disabled={isUploading}>
                            <label htmlFor="avatar-upload" className="flex cursor-pointer items-center">
                              <Upload className="mr-2 h-4 w-4" />
                              {isUploading ? "Uploading..." : "Upload Photo"}
                            </label>
                          </Button>
                          <p className="text-xs text-muted-foreground mt-2">
                            JPG, PNG or GIF. Max size 2MB.
                          </p>
                          {uploadError ? (
                            <p className="text-xs text-destructive mt-1">{uploadError}</p>
                          ) : null}
                        </div>
                      </div>

                      {/* Name Fields */}
                      <div className="grid gap-4 sm:grid-cols-2">
                        <div className="space-y-2">
                          <Label htmlFor="firstname">First Name</Label>
                          <Input
                            id="firstname"
                            disabled={isLoadingProfile}
                            value={firstname}
                            onChange={(e) => setFirstname(e.target.value)}
                          />
                        </div>
                        <div className="space-y-2">
                          <Label htmlFor="lastname">Last Name</Label>
                          <Input
                            id="lastname"
                            disabled={isLoadingProfile}
                            value={lastname}
                            onChange={(e) => setLastname(e.target.value)}
                          />
                        </div>
                      </div>

                      {/* Bio */}
                      <div className="space-y-2">
                        <Label htmlFor="bio">Bio</Label>
                        <Textarea
                          id="bio"
                          placeholder="Tell us a little about yourself..."
                          disabled={isLoadingProfile}
                          value={bio}
                          onChange={(e) => setBio(e.target.value)}
                          rows={4}
                        />
                        <p className="text-xs text-muted-foreground">
                          Brief description for your profile.
                        </p>
                      </div>

                      {profileError ? (
                        <p className="text-sm text-destructive">{profileError}</p>
                      ) : null}
                      {profileSaved ? (
                        <p className="text-sm text-emerald-600">Profile updated.</p>
                      ) : null}

                      <div className="flex justify-end">
                        <Button onClick={() => handleProfileSave()} disabled={isSavingProfile || isLoadingProfile || isUploading}>
                          {isSavingProfile ? "Saving..." : "Save Changes"}
                        </Button>
                      </div>
                    </CardContent>
                  </Card>
                </TabsContent>

                <TabsContent value="account" className="space-y-4">
                  <Card>
                    <CardHeader>
                      <CardTitle>Email Address</CardTitle>
                      <CardDescription>
                        Your email address is used for sign in and notifications
                      </CardDescription>
                    </CardHeader>
                    <CardContent className="space-y-4">
                      <div className="space-y-2">
                        <Label htmlFor="email">Email</Label>
                        <Input
                          id="email"
                          type="email"
                          value={email}
                          disabled
                        />
                        <p className="text-xs text-muted-foreground">
                          Contact support to change your email address
                        </p>
                      </div>
                    </CardContent>
                  </Card>

                  <Card>
                    <CardHeader>
                      <CardTitle>Linked Accounts</CardTitle>
                      <CardDescription>
                        Third-party accounts connected to your profile
                      </CardDescription>
                    </CardHeader>
                    <CardContent className="space-y-3">
                      {linkedAccounts.length === 0 ? (
                        <p className="text-sm text-muted-foreground">No linked accounts.</p>
                      ) : (
                        <ul className="space-y-2 text-sm">
                          {linkedAccounts.map((acct, idx) => (
                            <li key={`${acct.providerId}-${idx}`} className="flex items-center justify-between rounded-md border p-2">
                              <span className="font-medium capitalize">{acct.providerId}</span>
                              <span className="text-xs text-muted-foreground">Connected</span>
                            </li>
                          ))}
                        </ul>
                      )}
                    </CardContent>
                  </Card>

                  <Card>
                    <CardHeader>
                      <CardTitle>Change Password</CardTitle>
                      <CardDescription>
                        {allowPasswordChange
                          ? "Ensure your account is using a strong password"
                          : "Password changes are disabled for social sign-ins"}
                      </CardDescription>
                    </CardHeader>
                    <CardContent className="space-y-4">
                      <div className="space-y-2">
                        <Label htmlFor="current-password">Current Password</Label>
                        <Input
                          id="current-password"
                          type="password"
                          autoComplete="current-password"
                          disabled={!allowPasswordChange}
                          value={currentPassword}
                          onChange={(e) => setCurrentPassword(e.target.value)}
                        />
                      </div>
                      <div className="space-y-2">
                        <Label htmlFor="new-password">New Password</Label>
                        <Input
                          id="new-password"
                          type="password"
                          autoComplete="new-password"
                          disabled={!allowPasswordChange}
                          value={newPassword}
                          onChange={(e) => setNewPassword(e.target.value)}
                        />
                      </div>
                      <div className="space-y-2">
                        <Label htmlFor="confirm-password">Confirm New Password</Label>
                        <Input
                          id="confirm-password"
                          type="password"
                          autoComplete="new-password"
                          disabled={!allowPasswordChange}
                          value={confirmPassword}
                          onChange={(e) => setConfirmPassword(e.target.value)}
                        />
                      </div>

                      {passwordError ? (
                        <p className="text-sm text-destructive">{passwordError}</p>
                      ) : null}
                      {passwordSaved ? (
                        <p className="text-sm text-emerald-600">Password updated.</p>
                      ) : null}

                      <div className="flex justify-end">
                        <Button
                          onClick={handlePasswordChange}
                          disabled={isSavingPassword || !allowPasswordChange}
                          variant={allowPasswordChange ? "default" : "secondary"}
                        >
                          {allowPasswordChange
                            ? isSavingPassword
                              ? "Updating..."
                              : "Update Password"
                            : "Not available for social login"}
                        </Button>
                      </div>
                    </CardContent>
                  </Card>

                  <Card className="border-destructive">
                    <CardHeader>
                      <CardTitle className="text-destructive">Danger Zone</CardTitle>
                      <CardDescription>
                        Irreversible actions for your account
                      </CardDescription>
                    </CardHeader>
                    <CardContent>
                      <Button variant="destructive">Delete Account</Button>
                    </CardContent>
                  </Card>
                </TabsContent>
              </Tabs>
            </div>
          </div>
        </div>
      </SidebarInset>
    </SidebarProvider>
  );
}
