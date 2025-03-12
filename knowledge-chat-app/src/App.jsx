import React from 'react'
import { Box, ChakraProvider, extendTheme } from '@chakra-ui/react'
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom'
import MainLayout from './layouts/MainLayout'
import HomePage from './pages/HomePage'
import AboutPage from './pages/AboutPage'
import ChatPage from './pages/ChatPage'
import FileUploadPage from './pages/knowledge/FileUploadPage'
import ChatHistoryPage from './pages/ChatHistoryPage'

// Create theme
const theme = extendTheme({
  styles: {
    global: {
      body: {
        bg: 'white',
      }
    }
  }
})

function App() {
  console.log('App is rendering');
  
  return (
    <ChakraProvider theme={theme}>
      <Router>
        <Box minH="100vh">
          <Routes>
            <Route path="/" element={<HomePage />} />
            <Route path="/chat" element={
              <MainLayout>
                <ChatPage />
              </MainLayout>
            } />
            <Route path="/chat/:conversationId" element={
              <MainLayout>
                <ChatPage />
              </MainLayout>
            } />
            <Route path="/upload" element={
              <MainLayout>
                <FileUploadPage />
              </MainLayout>
            } />
            <Route path="/history" element={
              <MainLayout>
                <ChatHistoryPage />
              </MainLayout>
            } />
            <Route path="/history/:conversationId" element={
              <MainLayout>
                <ChatHistoryPage />
              </MainLayout>
            } />
            <Route path="*" element={<Navigate to="/" />} />
          </Routes>
        </Box>
      </Router>
    </ChakraProvider>
  )
}

export default App
