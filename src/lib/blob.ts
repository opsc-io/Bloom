/**
 * Vercel Blob Upload Utilities
 *
 * Usage in components:
 *
 * // Upload a file
 * const result = await uploadFile(file, 'avatars')
 * console.log(result.url) // https://oozfaovsgf7fxajz.public.blob.vercel-storage.com/avatars/image.png
 *
 * // Delete a file
 * await deleteFile(result.url)
 *
 * // List files
 * const files = await listFiles('avatars')
 */

export interface BlobResult {
  url: string
  pathname: string
  contentType: string
  contentDisposition: string
}

export async function uploadFile(file: File, folder?: string): Promise<BlobResult> {
  const formData = new FormData()
  formData.append('file', file)
  if (folder) {
    formData.append('folder', folder)
  }

  const response = await fetch('/api/upload', {
    method: 'POST',
    body: formData,
  })

  if (!response.ok) {
    const error = await response.json()
    throw new Error(error.error || 'Upload failed')
  }

  return response.json()
}

export async function deleteFile(url: string): Promise<void> {
  const response = await fetch('/api/upload', {
    method: 'DELETE',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ url }),
  })

  if (!response.ok) {
    const error = await response.json()
    throw new Error(error.error || 'Delete failed')
  }
}

export async function listFiles(prefix?: string): Promise<BlobResult[]> {
  const params = prefix ? `?prefix=${encodeURIComponent(prefix)}` : ''
  const response = await fetch(`/api/upload${params}`)

  if (!response.ok) {
    const error = await response.json()
    throw new Error(error.error || 'List failed')
  }

  const { blobs } = await response.json()
  return blobs
}
