import { type Metadata } from 'next'
import { notFound, redirect } from 'next/navigation'

import { auth } from '@/auth'
import { getChat, getMissingKeys } from '@/app/actions'
import { Chat } from '@/components/chat'
import { AI } from '@/lib/chat/actions'
import { Session } from '@/lib/types'
import { Message } from 'ai'

export interface ChatPageProps {
  params: {
    id: string
  }
}

type AIMessage = {
  id: string;
  content: string;
  role: 'assistant' | 'system' | 'user' | 'data';
  name?: string;
  function_call?: any;
}

// Helper function to convert message roles
function sanitizeMessages(messages: any[]): AIMessage[] {
  return messages.map(msg => ({
    id: msg.id,
    content: msg.content,
    role: (msg.role === 'function' || msg.role === 'tool') ? 'assistant' : msg.role as AIMessage['role'],
    name: msg.name,
    function_call: msg.function_call
  }));
}

export async function generateMetadata({
  params
}: ChatPageProps): Promise<Metadata> {
  const session = await auth()

  if (!session?.user) {
    return {}
  }

  const chat = await getChat(params.id, session.user.id)
  return {
    title: chat?.title.toString().slice(0, 50) ?? 'Chat'
  }
}

export default async function ChatPage({ params }: ChatPageProps) {
  const session = (await auth()) as Session
  const missingKeys = await getMissingKeys()

  if (!session?.user) {
    redirect(`/login?next=/chat/${params.id}`)
  }

  const userId = session.user.id as string
  const chat = await getChat(params.id, userId)

  if (!chat) {
    redirect('/')
  }

  if (chat?.userId !== session?.user?.id) {
    notFound()
  }

  return (
    <AI initialAIState={{ 
      chatId: chat.id, 
      messages: sanitizeMessages(chat.messages)
    }}>
      <Chat
        id={chat.id}
        session={session}
        initialMessages={sanitizeMessages(chat.messages)}
        missingKeys={missingKeys}
      />
    </AI>
  )
}
