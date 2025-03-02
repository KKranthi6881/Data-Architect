import { Box, Heading, Text, Button, VStack } from '@chakra-ui/react'
import { Link } from 'react-router-dom'

const AboutPage = () => {
  return (
    <Box maxW="800px" mx="auto" py={8}>
      <VStack spacing={4} align="start">
        <Heading color="brand.700">About Page</Heading>
        <Text fontSize="lg">This is a knowledge chat application</Text>
        <Button as={Link} to="/" colorScheme="brand">Go to Home</Button>
      </VStack>
    </Box>
  )
}

export default AboutPage 