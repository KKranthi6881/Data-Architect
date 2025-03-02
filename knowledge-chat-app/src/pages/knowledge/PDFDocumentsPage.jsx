import { Box, Heading, Text } from '@chakra-ui/react'
import FileUploadComponent from '../../components/uploads/FileUploadComponent'

const PDFDocumentsPage = () => {
  return (
    <Box>
      <Box bg="brand.50" py={6} px={4} mb={6}>
        <Heading size="lg" mb={2}>PDF Documents</Heading>
        <Text>Upload PDF documents to extract text and include in your knowledge base</Text>
      </Box>
      <FileUploadComponent />
    </Box>
  )
}

export default PDFDocumentsPage 