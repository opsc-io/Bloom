import { Button } from "@/components/ui/button"

type RoleDialogProps = {
  open: boolean
  error?: string | null
  isSaving?: boolean
  onSelect: (role: "practitioner" | "patient") => void
}

export function DashboardRoleDialog({
  open,
  error,
  isSaving = false,
  onSelect,
}: RoleDialogProps) {
  if (!open) return null

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 p-4">
      <div className="w-full max-w-md rounded-lg bg-background p-6 shadow-lg">
        <h2 className="text-lg font-semibold">Tell us about you</h2>
        <p className="mt-2 text-sm text-muted-foreground">
          Are you using Bloom as a practitioner or a patient? We use this to tailor your dashboard.
        </p>
        {error ? (
          <p className="mt-3 text-sm text-red-500">{error}</p>
        ) : null}
        <div className="mt-5 flex flex-col gap-3 sm:flex-row">
          <Button onClick={() => onSelect("practitioner")} disabled={isSaving}>
            I am a Practitioner
          </Button>
          <Button
            variant="outline"
            onClick={() => onSelect("patient")}
            disabled={isSaving}
          >
            I am a Patient
          </Button>
        </div>
      </div>
    </div>
  )
}
