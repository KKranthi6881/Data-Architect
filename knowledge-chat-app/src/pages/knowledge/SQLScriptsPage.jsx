import { Box, Heading, Text } from '@chakra-ui/react'
import FileUploadComponent from '../../components/uploads/FileUploadComponent'

const SQLScriptsPage = () => {
  return (
    <Box>
      <Box bg="brand.50" py={6} px={4} mb={6}>
        <Heading size="lg" mb={2}>SQL Scripts</Heading>
        <Text>Upload and analyze SQL scripts to include in your knowledge base</Text>
      </Box>
      <FileUploadComponent />
    </Box>
  )
}

export default SQLScriptsPage 