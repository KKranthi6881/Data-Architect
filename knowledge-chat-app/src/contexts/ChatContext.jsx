import { createContext, useState, useCallback } from 'react'

export const ChatContext = createContext()

export const ChatProvider = ({ children }) => {
  const [messages, setMessages] = useState([])

  // Mock AI response function (will be replaced with actual API call later)
  const getAIResponse = useCallback((userMessage) => {
    // Simple responses for now
    const responses = [
      "I'm an AI assistant here to help you with your questions.",
      "That's an interesting question. Let me think about it.",
      "Based on your knowledge store, I can provide this information.",
      "I don't have enough information to answer that question yet.",
      "Could you provide more details about your question?",
    ]
    
    return responses[Math.floor(Math.random() * responses.length)]
  }, [])

  const sendMessage = useCallback((text) => {
    // Add user message
    const userMessage = {
      text,
      sender: 'user',
      timestamp: new Date().toISOString()
    }
    
    setMessages(prev => [...prev, userMessage])
    
    // Simulate AI response after a short delay
    setTimeout(() => {
      const aiResponse = {
        text: getAIResponse(text),
        sender: 'ai',
        timestamp: new Date().toISOString()
      }
      
      setMessages(prev => [...prev, aiResponse])
    }, 1000)
  }, [getAIResponse])

  const clearChat = useCallback(() => {
    setMessages([])
  }, [])

  return (
    <ChatContext.Provider value={{ messages, sendMessage, clearChat }}>
      {children}
    </ChatContext.Provider>
  )
} 