"use client"
import { useState } from 'react'
import { Search, Phone, MoreVertical, Paperclip, Send, Check, Play, FileText } from 'lucide-react'
import { Input } from '@/components/ui/input'
import { Button } from '@/components/ui/button'
import { Avatar, AvatarFallback, AvatarImage } from '@/components/ui/avatar'
import { Badge } from '@/components/ui/badge'

const conversations = [
  { id: 1, name: 'X-AE-A-13b', avatar: '👤', lastMessage: 'Enter your message description here...', time: '12:25', online: true },
  { id: 2, name: 'Jerome White', avatar: '👤', lastMessage: 'Enter your message description here...', time: '12:25', online: true },
  { id: 3, name: 'Madagascar Silver', avatar: '👤', lastMessage: 'Enter your message description he...', time: '12:25', unread: 999, online: false },
  { id: 4, name: 'Pippins McGray', avatar: '👤', lastMessage: 'Enter your message description here...', time: '12:25', online: true },
  { id: 5, name: 'McKinsey Vermillion', avatar: '👤', lastMessage: 'Enter your message description here...', time: '12:25', unread: 8, online: true },
  { id: 6, name: 'Dorian F. Gray', avatar: '👤', lastMessage: 'Enter your message description here...', time: '12:25', unread: 2, online: false },
  { id: 7, name: 'Benedict Combersmacks', avatar: '👤', lastMessage: 'Enter your message description here...', time: '12:25', online: false },
  { id: 8, name: 'Kaori D. Miyazono', avatar: '👤', lastMessage: 'Enter your message description here...', time: '12:25', online: false },
  { id: 9, name: 'Saylor Twift', avatar: '👤', lastMessage: 'Enter your message description here...', time: '12:25', online: true },
  { id: 10, name: 'Miranda Blue', avatar: '👤', lastMessage: 'Enter your message description here...', time: '12:25', online: false },
]

export default function Page() {
  const [selectedChat, setSelectedChat] = useState(2)
  const [message, setMessage] = useState('')

  return (
    <div className="flex h-full max-h-full bg-gray-50 overflow-hidden">
      {/* Messages List */}
      <div className="w-80 bg-white border-r flex flex-col h-full">
        <div className="p-4 border-b flex-shrink-0">
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-xl font-semibold">Messages</h2>
            <Badge variant="secondary" className="rounded-full">29</Badge>
          </div>
          <div className="relative">
            <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-gray-400" />
            <Input placeholder="Search..." className="pl-10" />
          </div>
        </div>

        <div className="flex-1 overflow-y-auto">
          {conversations.map((conv) => (
            <div
              key={conv.id}
              onClick={() => setSelectedChat(conv.id)}
              className={`flex items-center gap-3 p-4 cursor-pointer hover:bg-gray-50 transition-colors ${
                selectedChat === conv.id ? 'bg-gray-100' : ''
              }`}
            >
              <div className="relative">
                <Avatar>
                  <AvatarFallback className="bg-gray-200">{conv.avatar}</AvatarFallback>
                </Avatar>
                {conv.online && (
                  <div className="absolute bottom-0 right-0 w-3 h-3 bg-green-500 border-2 border-white rounded-full" />
                )}
              </div>
              <div className="flex-1 min-w-0">
                <div className="flex items-center justify-between">
                  <h3 className="font-medium text-sm truncate">{conv.name}</h3>
                  <span className="text-xs text-gray-500">{conv.time}</span>
                </div>
                <div className="flex items-center justify-between">
                  <p className="text-sm text-gray-500 truncate">{conv.lastMessage}</p>
                  {conv.unread && (
                    <Badge className="ml-2 bg-indigo-600 hover:bg-indigo-700 text-xs h-5 min-w-5 flex items-center justify-center">
                      {conv.unread > 99 ? '99+' : conv.unread}
                    </Badge>
                  )}
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Chat Area */}
      <div className="flex-1 flex flex-col h-full">
        {/* Chat Header */}
        <div className="bg-white border-b p-4 flex items-center justify-between flex-shrink-0">
          <div className="flex items-center gap-3">
            <div className="relative">
              <Avatar>
                <AvatarFallback className="bg-gray-200">👤</AvatarFallback>
              </Avatar>
              <div className="absolute bottom-0 right-0 w-3 h-3 bg-green-500 border-2 border-white rounded-full" />
            </div>
            <div>
              <h3 className="font-semibold">Azunyan U. Wu</h3>
              <div className="flex items-center gap-1">
                <span className="text-xs text-green-600">● Online</span>
                <span className="text-xs text-gray-500">@azusanakano_1997</span>
              </div>
            </div>
          </div>
          <div className="flex items-center gap-2">
            <Button variant="outline" size="sm" className="gap-2">
              <Phone className="h-4 w-4" />
              Call
            </Button>
            <Button size="sm">View Profile</Button>
            <Button variant="ghost" size="icon">
              <MoreVertical className="h-4 w-4" />
            </Button>
          </div>
        </div>

        {/* Messages */}
        <div className="flex-1 overflow-y-auto p-4">
          <div className="space-y-4 max-w-4xl mx-auto">
            {/* Date separator */}
            <div className="text-center text-sm text-gray-500 mb-6">19 August</div>

            {/* Received message */}
            <div className="flex justify-start">
              <div className="bg-white rounded-2xl rounded-tl-sm p-4 max-w-md shadow-sm">
                <p className="text-sm">Hello my dear sir, I'm here do deliver the design requirement document for our next projects.</p>
                <span className="text-xs text-gray-500 mt-1 block">10:25</span>
              </div>
            </div>

            {/* File attachment */}
            <div className="flex justify-start">
              <div className="bg-white rounded-2xl rounded-tl-sm p-4 max-w-md shadow-sm">
                <div className="flex items-center gap-3 p-3 bg-gray-50 rounded-lg">
                  <FileText className="h-8 w-8 text-blue-600" />
                  <div className="flex-1">
                    <p className="text-sm font-medium">Design_project_2025.docx</p>
                    <p className="text-xs text-gray-500">2.5gb • docx</p>
                  </div>
                </div>
                <span className="text-xs text-gray-500 mt-2 block">10:26</span>
              </div>
            </div>

            {/* Sent message */}
            <div className="flex justify-end">
              <div className="bg-indigo-600 text-white rounded-2xl rounded-tr-sm p-4 max-w-md">
                <p className="text-sm">Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco labori</p>
                <div className="flex items-center justify-end gap-1 mt-1">
                  <span className="text-xs opacity-90">11:29</span>
                  <Check className="h-3 w-3" />
                </div>
              </div>
            </div>

            {/* Date separator */}
            <div className="text-center text-sm text-gray-500 my-6">Today</div>

            {/* Received text */}
            <div className="flex justify-start">
              <div className="bg-white rounded-2xl rounded-tl-sm p-4 max-w-md shadow-sm">
                <p className="text-sm">Do androids truly dream of electric sheeps?</p>
                <div className="flex items-center gap-1 mt-1">
                  <span className="text-xs text-gray-500">12:25</span>
                  <Check className="h-3 w-3 text-gray-500" />
                </div>
              </div>
            </div>

            {/* Voice message */}
            <div className="flex justify-end">
              <div className="bg-indigo-600 text-white rounded-2xl rounded-tr-sm p-4 max-w-sm flex items-center gap-3">
                <Button size="icon" variant="ghost" className="h-8 w-8 rounded-full bg-white/20 hover:bg-white/30">
                  <Play className="h-4 w-4 fill-white" />
                </Button>
                <div className="flex-1 flex items-center gap-2">
                  <div className="flex-1 h-8 flex items-center gap-0.5">
                    {Array.from({ length: 40 }).map((_, i) => (
                      <div
                        key={i}
                        className="w-0.5 bg-white/60 rounded-full"
                        style={{ height: `90%` }}
                      />
                    ))}
                  </div>
                  <span className="text-xs">02:12</span>
                </div>
                <div className="flex items-center gap-1">
                  <span className="text-xs opacity-90">01:25</span>
                  <Check className="h-3 w-3" />
                </div>
              </div>
            </div>

            {/* Video attachment */}
            <div className="flex justify-end">
              <div className="bg-white rounded-2xl rounded-tr-sm overflow-hidden max-w-sm shadow-sm">
                <div className="relative">
                  <div className="bg-gradient-to-br from-orange-200 to-orange-300 aspect-video flex items-center justify-center">
                    <Button size="icon" className="h-12 w-12 rounded-full">
                      <Play className="h-6 w-6 fill-white" />
                    </Button>
                  </div>
                </div>
                <div className="p-2 flex items-center justify-end gap-1">
                  <span className="text-xs text-gray-500">02:25</span>
                  <Check className="h-3 w-3 text-gray-500" />
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Message Input */}
        <div className="bg-white border-t p-4 flex-shrink-0">
          <div className="max-w-4xl mx-auto flex items-center gap-2">
            <Button variant="ghost" size="icon">
              <Paperclip className="h-5 w-5 text-gray-500" />
            </Button>
            <Input
              placeholder="Type your message..."
              value={message}
              onChange={(e) => setMessage(e.target.value)}
              className="flex-1"
            />
            <Button size="icon" className="bg-indigo-600 hover:bg-indigo-700">
              <Send className="h-5 w-5" />
            </Button>
          </div>
        </div>
      </div>
    </div>
  )
}