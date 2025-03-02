import { Box, Flex } from '@chakra-ui/react'
import { Outlet } from 'react-router-dom'
import Navbar from '../components/common/Navbar'
import Sidebar from '../components/common/Sidebar'

const MainLayout = () => {
  return (
    <Flex direction="column" minH="100vh">
      <Navbar />
      <Flex flex="1">
        <Sidebar />
        <Box as="main" flex="1" p={4} overflowY="auto">
          <Outlet />
        </Box>
      </Flex>
    </Flex>
  )
}

export default MainLayout 