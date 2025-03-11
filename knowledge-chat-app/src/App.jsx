import React from 'react'
import { Box, Flex, ChakraProvider } from '@chakra-ui/react'
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom'
import Navbar from './components/common/Navbar'
import HomePage from './pages/HomePage'
import AboutPage from './pages/AboutPage'
import ChatPage from './pages/ChatPage'
import FileUploadPage from './pages/knowledge/FileUploadPage'
import ChatHistoryPage from './pages/ChatHistoryPage'
import AppHeader from './components/AppHeader'
import './App.css'

function App() {
  return (
    <ChakraProvider>
      <Router>
        <Box minH="100vh">
          <AppHeader />
          <Box as="main" pt={4}>
            <Routes>
              <Route path="/" element={<HomePage />} />
              <Route path="/about" element={<AboutPage />} />
              <Route path="/upload" element={<FileUploadPage />} />
              <Route path="/chat" element={<ChatPage />} />
              <Route path="/chat/:conversationId" element={<ChatPage />} />
              <Route path="/history" element={<ChatHistoryPage />} />
              <Route path="/history/:conversationId" element={<ChatHistoryPage />} />
              <Route path="*" element={<Navigate to="/" replace />} />
            </Routes>
          </Box>
        </Box>
      </Router>
    </ChakraProvider>
  )
}

export default App
