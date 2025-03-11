import React, { createContext, useContext, useState, useReducer } from 'react';

const ChatContext = createContext();

const chatReducer = (state, action) => {
  switch (action.type) {
    case 'SET_THREADS':
      return { ...state, threads: action.payload };
    case 'SET_MESSAGES':
      return { 
        ...state, 
        threadMessages: {
          ...state.threadMessages,
          [action.payload.threadId]: action.payload.messages 
        }
      };
    case 'SET_SELECTED_THREAD':
      return { ...state, selectedThreadId: action.payload };
    case 'CLEAR_THREAD_MESSAGES':
      return { ...state, threadMessages: {} };
    case 'SET_LOADING':
      return { ...state, isLoading: action.payload };
    default:
      return state;
  }
};

export const ChatProvider = ({ children }) => {
  const [state, dispatch] = useReducer(chatReducer, {
    threads: [],
    threadMessages: {},
    selectedThreadId: null,
    isLoading: false
  });

  const fetchThreads = async () => {
    try {
      dispatch({ type: 'SET_LOADING', payload: true });
      const response = await fetch('http://localhost:8000/api/conversations');
      if (!response.ok) throw new Error('Failed to fetch conversations');
      
      const data = await response.json();
      if (data.status === 'success' && Array.isArray(data.conversations)) {
        const validThreads = data.conversations
          .filter(t => t.thread_id && t.thread_id !== 'undefined')
          .sort((a, b) => new Date(b.timestamp) - new Date(a.timestamp));
        
        dispatch({ type: 'SET_THREADS', payload: validThreads });
      }
    } catch (error) {
      console.error('Error fetching threads:', error);
      throw error;
    } finally {
      dispatch({ type: 'SET_LOADING', payload: false });
    }
  };

  const fetchThreadMessages = async (threadId) => {
    if (!threadId || threadId === 'undefined') return;
    
    try {
      const response = await fetch(`http://localhost:8000/api/thread/${threadId}/conversations`);
      if (!response.ok) throw new Error('Failed to fetch thread messages');
      
      const data = await response.json();
      if (data.status === 'success' && Array.isArray(data.conversations)) {
        dispatch({ 
          type: 'SET_MESSAGES', 
          payload: { 
            threadId, 
            messages: data.conversations.filter(msg => msg.id && (msg.query || msg.output))
          }
        });
      }
    } catch (error) {
      console.error('Error fetching thread messages:', error);
      throw error;
    }
  };

  const selectThread = async (threadId) => {
    dispatch({ type: 'SET_SELECTED_THREAD', payload: threadId });
    if (threadId && !state.threadMessages[threadId]) {
      await fetchThreadMessages(threadId);
    }
  };

  const value = {
    ...state,
    fetchThreads,
    fetchThreadMessages,
    selectThread,
    dispatch
  };

  return <ChatContext.Provider value={value}>{children}</ChatContext.Provider>;
};

export const useChat = () => {
  const context = useContext(ChatContext);
  if (!context) {
    throw new Error('useChat must be used within a ChatProvider');
  }
  return context;
}; 