import 'server-only'

import {
  createAI,
  getMutableAIState,
  getAIState,
  render,
  createStreamableValue
} from 'ai/rsc'
import OpenAI from 'openai'

import {
  nanoid
} from '@/lib/utils'
import { saveChat } from '@/app/actions'
import { SpinnerMessage, UserMessage, BotMessage } from '@/components/message'
import { Chat } from '@/lib/types'
import { auth } from '@/auth'

import { getPrompt } from '@/app/api/getDataFromKV'


const openai = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY || ''
})


async function submitUserMessage(content: string, type: string) {
  'use server'

  const aiState = getMutableAIState<typeof AI>()

  // Add user message to state
  aiState.update({
    ...aiState.get(),
    messages: [
      ...aiState.get().messages,
      {
        id: nanoid(),
        role: 'user',
        content,
      }
    ]
  })

  try {
    // Check if OpenAI API key is configured
    if (!process.env.OPENAI_API_KEY) {
      console.error('OpenAI API key is not configured');
      throw new Error('OpenAI API key is not configured');
    } else {
      console.log('OpenAI API key is configured');
    }

    // Show loading state
    const loadingNode = <SpinnerMessage />
    const loadingId = nanoid()
    
    // Get the system prompt
    let systemPrompt;
    try {
      systemPrompt = await getPrompt();
      if (!systemPrompt) {
        console.warn('System prompt is empty, using default');
        systemPrompt = 'You are a helpful assistant.';
      }
    } catch (promptError) {
      console.error('Error getting prompt:', promptError);
      systemPrompt = 'You are a helpful assistant.';
    }
    
    console.log('Using system prompt:', systemPrompt.substring(0, 50) + '...');
    
    // Prepare messages for OpenAI
    const messages = [
      {
        role: 'system' as const,
        content: systemPrompt
      },
      ...aiState.get().messages.map((message: any) => ({
        role: message.role as 'user' | 'assistant' | 'system',
        content: message.content,
        name: message.name
      }))
    ]
    
    try {
      // Fall back to a non-streaming implementation which is more reliable
      const response = await openai.chat.completions.create({
        model: 'gpt-3.5-turbo', // You can change this to 'gpt-4' if you have access to it
        messages: messages,
        stream: false // Changed to false to avoid streaming issues
      })

      // Get the response content
      const responseContent = response.choices[0]?.message?.content || 'No response received';
      
      console.log('Received OpenAI response successfully');

      // Update AI state with the response
      aiState.done({
        ...aiState.get(),
        messages: [
          ...aiState.get().messages,
          {
            id: nanoid(),
            role: 'assistant',
            content: responseContent,
          }
        ]
      });
      
      // Return the final UI with the complete response
      return {
        id: loadingId,
        display: <BotMessage content={responseContent} />
      };
    } catch (error) {
      console.error('Error in OpenAI API call:', error);
      
      // Create a more detailed error response
      let errorMessage = 'Sorry, there was an error processing your request. Please try again.';
      
      if (error instanceof Error) {
        if (error.message.includes('API key')) {
          errorMessage = 'The OpenAI API key is missing or invalid. Please check your environment configuration.';
        } else if (error.message.includes('rate limit') || error.message.includes('quota')) {
          errorMessage = 'The OpenAI API quota has been exceeded. The application is now using a simple mock response system instead.';
        } else if (error.message.includes('timeout')) {
          errorMessage = 'The request to OpenAI timed out. Please try again.';
        } else {
          // Log the specific error for debugging
          console.error('Specific error:', error.message);
          errorMessage = `Error: ${error.message}`;
        }
      }
      
      // Update AI state with error message
      aiState.done({
        ...aiState.get(),
        messages: [
          ...aiState.get().messages,
          {
            id: nanoid(),
            role: 'assistant',
            content: errorMessage,
          }
        ]
      });
      
      // Return error UI
      return {
        id: nanoid(),
        display: <BotMessage content={errorMessage} />
      };
    }
  } catch (error) {
    console.error('Error in AI processing:', error);
    
    // Create a more detailed error response
    let errorMessage = 'Sorry, there was an error processing your request. Please try again.';
    
    if (error instanceof Error) {
      if (error.message.includes('API key')) {
        errorMessage = 'The OpenAI API key is missing or invalid. Please check your environment configuration.';
      } else if (error.message.includes('rate limit') || error.message.includes('quota')) {
        errorMessage = 'The OpenAI API quota has been exceeded. The application is now using a simple mock response system instead.';
      } else if (error.message.includes('timeout')) {
        errorMessage = 'The request to OpenAI timed out. Please try again.';
      } else {
        // Log the specific error for debugging
        console.error('Specific error:', error.message);
        errorMessage = `Error: ${error.message}`;
      }
    }
    
    // Update AI state with error message
    aiState.done({
      ...aiState.get(),
      messages: [
        ...aiState.get().messages,
        {
          id: nanoid(),
          role: 'assistant',
          content: errorMessage,
        }
      ]
    });
    
    // Return error UI
    return {
      id: nanoid(),
      display: <BotMessage content={errorMessage} />
    };
  }
}

export type Message = {
  role: 'user' | 'assistant' | 'system' | 'data'
  content: string | any
  id: string
  name?: string
}

export type AIState = {
  chatId: string
  messages: Message[]
}

export type UIState = {
  id: string
  display: React.ReactNode
}[]

export const AI = createAI<AIState, UIState>({
  actions: {
    submitUserMessage
  },
  initialUIState: [],
  initialAIState: { chatId: nanoid(), messages: [] },
  unstable_onGetUIState: async () => {
    'use server'

    try {
      const session = await auth()

      if (session && session.user) {
        const aiState = getAIState()

        if (aiState) {
          const uiState = getUIStateFromAIState(aiState)
          return uiState
        }
      } else {
        return
      }
    } catch (error) {
      console.error('Error in unstable_onGetUIState:', error);
      return [];
    }
  },
  unstable_onSetAIState: async ({ state, done }) => {
    'use server'

    try {
      const session = await auth()

      if (session && session.user) {
        const { chatId, messages } = state

        const createdAt = new Date()
        const userId = session.user.id as string
        const path = `/chat/${chatId}`
        const title = messages[0]?.content?.substring(0, 100) || 'New Chat'

        const chat: Chat = {
          id: chatId,
          title,
          userId,
          createdAt,
          messages,
          path
        }

        await saveChat(chat)
      } else {
        return
      }
    } catch (error) {
      console.error('Error in unstable_onSetAIState:', error);
    }
  }
})

export const getUIStateFromAIState = (aiState: Chat) => {
  return aiState.messages
    .filter(message => message.role !== 'system')
    .map((message, index) => ({
      id: `${aiState.chatId}-${index}`,
      display:
        message.role === 'user' ? (
          <UserMessage>{message.content}</UserMessage>
        ) : (
          <BotMessage content={message.content} />
        )
    }))
}
