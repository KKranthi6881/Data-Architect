import { createContext, useState, useCallback, useEffect } from 'react'

export const KnowledgeContext = createContext()

export const KnowledgeProvider = ({ children }) => {
  const [knowledgeItems, setKnowledgeItems] = useState(() => {
    // Load from localStorage on initial render
    const savedItems = localStorage.getItem('knowledgeItems')
    return savedItems ? JSON.parse(savedItems) : [
      {
        id: '1',
        title: 'React Basics',
        content: 'React is a JavaScript library for building user interfaces. It allows you to create reusable UI components.',
        createdAt: new Date().toISOString()
      },
      {
        id: '2',
        title: 'JavaScript ES6',
        content: 'ES6 introduced many features like arrow functions, destructuring, spread operator, and more.',
        createdAt: new Date().toISOString()
      }
    ]
  })

  // Save to localStorage whenever knowledgeItems changes
  useEffect(() => {
    localStorage.setItem('knowledgeItems', JSON.stringify(knowledgeItems))
  }, [knowledgeItems])

  const addKnowledgeItem = useCallback((item) => {
    setKnowledgeItems(prev => [...prev, item])
  }, [])

  const updateKnowledgeItem = useCallback((id, updatedItem) => {
    setKnowledgeItems(prev => 
      prev.map(item => item.id === id ? { ...item, ...updatedItem } : item)
    )
  }, [])

  const deleteKnowledgeItem = useCallback((id) => {
    setKnowledgeItems(prev => prev.filter(item => item.id !== id))
  }, [])

  return (
    <KnowledgeContext.Provider value={{
      knowledgeItems,
      addKnowledgeItem,
      updateKnowledgeItem,
      deleteKnowledgeItem
    }}>
      {children}
    </KnowledgeContext.Provider>
  )
} 