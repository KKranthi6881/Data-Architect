import { Box, Heading, Text } from '@chakra-ui/react'
import FileUploadComponent from '../../components/uploads/FileUploadComponent'

const GitHubKnowledgePage = () => {
  return (
    <Box>
      <Box bg="brand.50" py={6} px={4} mb={6}>
        <Heading size="lg" mb={2}>GitHub Repository</Heading>
        <Text>Connect to GitHub repositories to include code and documentation in your knowledge base</Text>
      </Box>
      <FileUploadComponent />
    </Box>
  )
}

export default GitHubKnowledgePage 