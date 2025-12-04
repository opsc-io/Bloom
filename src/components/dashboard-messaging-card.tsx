import { useState } from "react"
import { Search, Maximize2, Minimize2, MessageSquare, Send } from "lucide-react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import { Button } from "@/components/ui/button"
import { Avatar, AvatarFallback } from "@/components/ui/avatar"

type Conversation = {
  id: string
  name: string
  avatar: string
  avatarColor: string
  lastMessage: string
  time: string
  unread: number
  active: boolean
}

type Message = {
  id: string
  sender: string
  message: string
  time: string
  isMe: boolean
  avatar: string
  avatarColor: string
}

type DashboardMessagingCardProps = {
  conversations: Conversation[]
  messages: Message[]
}

export function DashboardMessagingCard({ conversations, messages }: DashboardMessagingCardProps) {
  const [isExpanded, setIsExpanded] = useState(false)

  return (
    <Card
      className={`transition-all duration-300 ${isExpanded
        ? 'fixed inset-0 z-50 rounded-none border-0 md:col-span-1'
        : 'md:col-span-1 aspect-video cursor-pointer hover:shadow-lg'
        }`}
      onClick={() => !isExpanded && setIsExpanded(true)}
    >
      {!isExpanded ? (
        <div>
          <CardHeader className="pb-3">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <MessageSquare className="h-5 w-5 text-primary" />
                <CardTitle>Messages</CardTitle>
              </div>
              <Maximize2 className="h-4 w-4 text-muted-foreground" />
            </div>
            <CardDescription>Recent conversations</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="flex items-center gap-3 flex-wrap">
              {conversations.length > 0 ? (
                conversations.map((conv) => (
                  <div key={conv.id} className="relative">
                    <Avatar className="h-12 w-12 cursor-pointer hover:scale-110 transition-transform">
                      <AvatarFallback className={`${conv.avatarColor} text-white font-semibold`}>
                        {conv.avatar}
                      </AvatarFallback>
                    </Avatar>
                    {conv.unread > 0 && (
                      <span className="absolute -top-1 -right-1 bg-primary text-primary-foreground text-xs rounded-full h-5 w-5 flex items-center justify-center font-semibold border-2 border-background">
                        {conv.unread}
                      </span>
                    )}
                  </div>
                ))
              ) : (
                <p className="text-sm text-muted-foreground py-4">No messages yet</p>
              )}
            </div>
          </CardContent>
        </div>
      ) : (
        <div className="flex h-screen">
          <div className="w-80 border-r bg-muted/30 flex flex-col">
            <div className="p-4 border-b">
              <div className="flex items-center justify-between mb-4">
                <h2 className="text-2xl font-bold">Chats</h2>
                <Button
                  size="sm"
                  variant="ghost"
                  onClick={(e) => {
                    e.stopPropagation();
                    setIsExpanded(false);
                  }}
                >
                  <Minimize2 className="h-4 w-4" />
                </Button>
              </div>
              <div className="relative">
                <Search className="absolute left-2 top-2.5 h-4 w-4 text-muted-foreground" />
                <Input placeholder="Search conversations..." className="pl-8" />
              </div>
            </div>
            <div className="flex-1 overflow-y-auto">
              {conversations.length > 0 ? (
                conversations.map((conv) => (
                  <div
                    key={conv.id}
                    className={`flex items-center gap-3 p-4 hover:bg-muted/50 cursor-pointer transition-colors ${conv.active ? 'bg-muted/50 border-l-2 border-primary' : ''
                      }`}
                  >
                    <Avatar className="h-12 w-12 shrink-0">
                      <AvatarFallback className={`${conv.avatarColor} text-white font-semibold`}>
                        {conv.avatar}
                      </AvatarFallback>
                    </Avatar>
                    <div className="flex-1 min-w-0">
                      <div className="flex items-center justify-between mb-1">
                        <p className="font-semibold text-sm truncate">{conv.name}</p>
                        <span className="text-xs text-muted-foreground">{conv.time}</span>
                      </div>
                      <div className="flex items-center justify-between">
                        <p className="text-sm text-muted-foreground truncate">{conv.lastMessage}</p>
                        {conv.unread > 0 && (
                          <span className="ml-2 bg-primary text-primary-foreground text-xs rounded-full h-5 w-5 flex items-center justify-center shrink-0">
                            {conv.unread}
                          </span>
                        )}
                      </div>
                    </div>
                  </div>
                ))
              ) : (
                <div className="flex items-center justify-center h-full">
                  <p className="text-muted-foreground">No conversations yet</p>
                </div>
              )}
            </div>
          </div>

          <div className="flex-1 flex flex-col">
            <div className="p-4 border-b bg-background">
              {conversations.find(c => c.active) ? (
                <div className="flex items-center gap-3">
                  <Avatar className="h-10 w-10">
                    <AvatarFallback className={`${conversations.find(c => c.active)?.avatarColor} text-white font-semibold`}>
                      {conversations.find(c => c.active)?.avatar}
                    </AvatarFallback>
                  </Avatar>
                  <div>
                    <p className="font-semibold">{conversations.find(c => c.active)?.name}</p>
                    <p className="text-xs text-muted-foreground">Active now</p>
                  </div>
                </div>
              ) : (
                <p className="text-muted-foreground">Select a conversation</p>
              )}
            </div>

            <div className="flex-1 overflow-y-auto p-6 space-y-4 bg-muted/20">
              {messages.length > 0 ? (
                messages.map((msg) => (
                  <div
                    key={msg.id}
                    className={`flex gap-3 ${msg.isMe ? 'justify-end' : 'justify-start'}`}
                  >
                    {!msg.isMe && (
                      <Avatar className="h-8 w-8 shrink-0">
                        <AvatarFallback className={`${msg.avatarColor} text-white text-xs font-semibold`}>
                          {msg.avatar}
                        </AvatarFallback>
                      </Avatar>
                    )}
                    <div className={`flex flex-col ${msg.isMe ? 'items-end' : 'items-start'} max-w-[60%]`}>
                      {!msg.isMe && <p className="text-xs font-semibold text-muted-foreground mb-1">{msg.sender}</p>}
                      <div className={`${msg.isMe ? 'bg-primary text-primary-foreground' : 'bg-background border'} rounded-2xl px-4 py-2.5 shadow-sm`}>
                        <p className="text-sm">{msg.message}</p>
                      </div>
                      <p className="text-xs text-muted-foreground mt-1">{msg.time}</p>
                    </div>
                    {msg.isMe && (
                      <Avatar className="h-8 w-8 shrink-0">
                        <AvatarFallback className={`${msg.avatarColor} text-white text-xs font-semibold`}>
                          {msg.avatar}
                        </AvatarFallback>
                      </Avatar>
                    )}
                  </div>
                ))
              ) : (
                <div className="flex items-center justify-center h-full">
                  <p className="text-muted-foreground">No messages yet</p>
                </div>
              )}
            </div>

            <div className="p-4 border-t bg-background">
              <div className="flex gap-2">
                <Input placeholder="Type a message..." className="flex-1" />
                <Button size="icon" className="shrink-0">
                  <Send className="h-4 w-4" />
                </Button>
              </div>
            </div>
          </div>
        </div>
      )}
    </Card>
  )
}
