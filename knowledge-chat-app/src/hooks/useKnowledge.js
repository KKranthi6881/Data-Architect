import { useContext } from 'react'
import { KnowledgeContext } from '../contexts/KnowledgeContext'

export const useKnowledge = () => {
  const context = useContext(KnowledgeContext)
  
  if (!context) {
    throw new Error('useKnowledge must be used within a KnowledgeProvider')
  }
  
  return context
} 