import { Box, Flex, Heading, Button, useColorMode, IconButton, HStack, Avatar } from '@chakra-ui/react'
import { Link as RouterLink } from 'react-router-dom'
import { FaMoon, FaSun } from 'react-icons/fa'
import { IoChatbubble, IoCloudUpload } from 'react-icons/io5'

const Navbar = () => {
  const { colorMode, toggleColorMode } = useColorMode()

  return (
    <Box 
      as="nav" 
      bg="white" 
      color="gray.800" 
      px={4} 
      py={3} 
      shadow="sm"
      borderBottom="1px"
      borderColor="gray.100"
    >
      <Flex align="center" justify="space-between" maxW="1200px" mx="auto">
        <HStack spacing={2}>
          <Avatar 
            size="sm" 
            bg="brand.500" 
            color="white" 
            name="K" 
            fontWeight="bold"
          />
          <Heading as={RouterLink} to="/" size="md" fontWeight="600">
            Knowledge Chat
          </Heading>
        </HStack>
        
        <Flex align="center" gap={4}>
          <Button as={RouterLink} to="/" variant="ghost" leftIcon={<IoChatbubble />}>
            AI Chat
          </Button>
          
          <Button as={RouterLink} to="/upload" variant="ghost" leftIcon={<IoCloudUpload />}>
            Upload Sources
          </Button>
          
          <IconButton
            icon={colorMode === 'light' ? <FaMoon /> : <FaSun />}
            onClick={toggleColorMode}
            variant="ghost"
            aria-label="Toggle color mode"
            color="gray.500"
          />
        </Flex>
      </Flex>
    </Box>
  )
}

export default Navbar 