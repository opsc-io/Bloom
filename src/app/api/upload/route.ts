import { put, del, list } from '@vercel/blob'
import { NextRequest, NextResponse } from 'next/server'

export async function POST(request: NextRequest) {
  try {
    const formData = await request.formData()
    const file = formData.get('file') as File

    if (!file) {
      return NextResponse.json({ error: 'No file provided' }, { status: 400 })
    }

    // Optional: get folder path from form data
    const folder = formData.get('folder') as string || ''
    const filename = folder ? `${folder}/${file.name}` : file.name

    const blob = await put(filename, file, {
      access: 'public',
    })

    return NextResponse.json(blob)
  } catch (error) {
    console.error('Upload error:', error)
    return NextResponse.json(
      { error: error instanceof Error ? error.message : 'Upload failed' },
      { status: 500 }
    )
  }
}

export async function DELETE(request: NextRequest) {
  try {
    const { url } = await request.json()

    if (!url) {
      return NextResponse.json({ error: 'No URL provided' }, { status: 400 })
    }

    await del(url)
    return NextResponse.json({ success: true })
  } catch (error) {
    console.error('Delete error:', error)
    return NextResponse.json(
      { error: error instanceof Error ? error.message : 'Delete failed' },
      { status: 500 }
    )
  }
}

export async function GET(request: NextRequest) {
  try {
    const { searchParams } = new URL(request.url)
    const prefix = searchParams.get('prefix') || undefined

    const { blobs } = await list({ prefix })
    return NextResponse.json({ blobs })
  } catch (error) {
    console.error('List error:', error)
    return NextResponse.json(
      { error: error instanceof Error ? error.message : 'List failed' },
      { status: 500 }
    )
  }
}
