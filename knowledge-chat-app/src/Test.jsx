import React from 'react'
import { ChakraProvider, Box, Text, Button, VStack } from '@chakra-ui/react'

function Test() {
  return (
    <ChakraProvider>
      <Box p={5} bg="blue.100" minH="100vh">
        <VStack spacing={4} align="start">
          <Text fontSize="2xl" color="blue.800">Test Component with Chakra UI</Text>
          <Button colorScheme="blue">Click Me</Button>
        </VStack>
      </Box>
    </ChakraProvider>
  )
}

export default Test 