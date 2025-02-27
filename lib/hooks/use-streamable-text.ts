import { StreamableValue } from 'ai/rsc'
import { useState } from 'react'

export const useStreamableText = (
  content: string | StreamableValue<string>
) => {
  // Simply return the content as a string
  const [text] = useState(typeof content === 'string' ? content : 'Loading response...')
  return text
}
