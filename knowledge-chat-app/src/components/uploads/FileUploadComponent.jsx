import { useState } from 'react'
import {
  Box,
  Button,
  FormControl,
  FormLabel,
  Input,
  VStack,
  HStack,
  Text,
  Icon,
  useToast,
  Card,
  CardBody,
  Badge,
  Progress,
  Tabs,
  TabList,
  TabPanels,
  Tab,
  TabPanel,
  InputGroup,
  InputLeftAddon
} from '@chakra-ui/react'
import { 
  IoCloudUpload, 
  IoDocument, 
  IoLogoGithub, 
  IoCodeSlash,
  IoCheckmarkCircle,
  IoWarning
} from 'react-icons/io5'

const FileUploadComponent = () => {
  const [sqlFile, setSqlFile] = useState(null)
  const [pdfFile, setPdfFile] = useState(null)
  const [repoUrl, setRepoUrl] = useState('')
  const [isUploading, setIsUploading] = useState(false)
  const [uploadProgress, setUploadProgress] = useState(0)
  const [uploadedFiles, setUploadedFiles] = useState([])
  const toast = useToast()

  const handleSqlFileChange = (e) => {
    if (e.target.files[0]) {
      setSqlFile(e.target.files[0])
    }
  }

  const handlePdfFileChange = (e) => {
    if (e.target.files[0]) {
      setPdfFile(e.target.files[0])
    }
  }

  const handleRepoUrlChange = (e) => {
    setRepoUrl(e.target.value)
  }

  const simulateUpload = (fileType, fileName) => {
    setIsUploading(true)
    setUploadProgress(0)
    
    const interval = setInterval(() => {
      setUploadProgress(prev => {
        if (prev >= 100) {
          clearInterval(interval)
          setIsUploading(false)
          
          // Add to uploaded files
          setUploadedFiles(prev => [
            ...prev, 
            { 
              id: Date.now(), 
              name: fileName, 
              type: fileType, 
              uploadedAt: new Date().toISOString() 
            }
          ])
          
          toast({
            title: 'Upload complete',
            description: `${fileName} has been uploaded successfully.`,
            status: 'success',
            duration: 5000,
            isClosable: true,
          })
          
          // Reset form
          if (fileType === 'sql') setSqlFile(null)
          if (fileType === 'pdf') setPdfFile(null)
          if (fileType === 'github') setRepoUrl('')
          
          return 0
        }
        return prev + 10
      })
    }, 300)
  }

  const handleSqlUpload = () => {
    if (!sqlFile) return
    simulateUpload('sql', sqlFile.name)
  }

  const handlePdfUpload = () => {
    if (!pdfFile) return
    simulateUpload('pdf', pdfFile.name)
  }

  const handleGithubUpload = () => {
    if (!repoUrl) return
    
    // Extract repo name from URL
    const repoName = repoUrl.split('/').pop() || 'repository'
    simulateUpload('github', repoName)
  }

  const getFileIcon = (fileType) => {
    switch (fileType) {
      case 'sql':
        return IoCodeSlash
      case 'pdf':
        return IoDocument
      case 'github':
        return IoLogoGithub
      default:
        return IoDocument
    }
  }

  const getFileColor = (fileType) => {
    switch (fileType) {
      case 'sql':
        return 'blue'
      case 'pdf':
        return 'red'
      case 'github':
        return 'purple'
      default:
        return 'gray'
    }
  }

  return (
    <Box maxW="1000px" mx="auto" py={8} px={4}>
      <Tabs colorScheme="brand" variant="enclosed">
        <TabList>
          <Tab><Icon as={IoCodeSlash} mr={2} /> SQL Files</Tab>
          <Tab><Icon as={IoDocument} mr={2} /> PDF Documents</Tab>
          <Tab><Icon as={IoLogoGithub} mr={2} /> GitHub Repository</Tab>
        </TabList>

        <TabPanels>
          <TabPanel>
            <VStack spacing={6} align="stretch">
              <Card variant="outline" p={4}>
                <CardBody>
                  <VStack spacing={4} align="stretch">
                    <FormControl>
                      <FormLabel>Upload SQL File</FormLabel>
                      <Input
                        type="file"
                        accept=".sql"
                        onChange={handleSqlFileChange}
                        p={1}
                      />
                      <Text fontSize="xs" color="gray.500" mt={1}>
                        Upload SQL scripts to analyze and include in your knowledge base
                      </Text>
                    </FormControl>
                    
                    {sqlFile && (
                      <HStack>
                        <Text fontSize="sm">Selected file: {sqlFile.name}</Text>
                        <Badge colorScheme="blue">{(sqlFile.size / 1024).toFixed(2)} KB</Badge>
                      </HStack>
                    )}
                    
                    <Button
                      leftIcon={<IoCloudUpload />}
                      colorScheme="brand"
                      onClick={handleSqlUpload}
                      isDisabled={!sqlFile || isUploading}
                      isLoading={isUploading}
                      loadingText="Uploading..."
                    >
                      Upload SQL File
                    </Button>
                    
                    {isUploading && (
                      <Progress value={uploadProgress} size="sm" colorScheme="brand" borderRadius="md" />
                    )}
                  </VStack>
                </CardBody>
              </Card>
            </VStack>
          </TabPanel>

          <TabPanel>
            <VStack spacing={6} align="stretch">
              <Card variant="outline" p={4}>
                <CardBody>
                  <VStack spacing={4} align="stretch">
                    <FormControl>
                      <FormLabel>Upload PDF Document</FormLabel>
                      <Input
                        type="file"
                        accept=".pdf"
                        onChange={handlePdfFileChange}
                        p={1}
                      />
                      <Text fontSize="xs" color="gray.500" mt={1}>
                        Upload PDF documents to extract text and include in your knowledge base
                      </Text>
                    </FormControl>
                    
                    {pdfFile && (
                      <HStack>
                        <Text fontSize="sm">Selected file: {pdfFile.name}</Text>
                        <Badge colorScheme="red">{(pdfFile.size / 1024).toFixed(2)} KB</Badge>
                      </HStack>
                    )}
                    
                    <Button
                      leftIcon={<IoCloudUpload />}
                      colorScheme="brand"
                      onClick={handlePdfUpload}
                      isDisabled={!pdfFile || isUploading}
                      isLoading={isUploading}
                      loadingText="Uploading..."
                    >
                      Upload PDF Document
                    </Button>
                    
                    {isUploading && (
                      <Progress value={uploadProgress} size="sm" colorScheme="brand" borderRadius="md" />
                    )}
                  </VStack>
                </CardBody>
              </Card>
            </VStack>
          </TabPanel>

          <TabPanel>
            <VStack spacing={6} align="stretch">
              <Card variant="outline" p={4}>
                <CardBody>
                  <VStack spacing={4} align="stretch">
                    <FormControl>
                      <FormLabel>GitHub Repository URL</FormLabel>
                      <InputGroup>
                        <InputLeftAddon children="URL" />
                        <Input
                          type="url"
                          placeholder="https://github.com/username/repository"
                          value={repoUrl}
                          onChange={handleRepoUrlChange}
                        />
                      </InputGroup>
                      <Text fontSize="xs" color="gray.500" mt={1}>
                        Connect to a GitHub repository to include code and documentation in your knowledge base
                      </Text>
                    </FormControl>
                    
                    <Button
                      leftIcon={<IoLogoGithub />}
                      colorScheme="brand"
                      onClick={handleGithubUpload}
                      isDisabled={!repoUrl || isUploading}
                      isLoading={isUploading}
                      loadingText="Connecting..."
                    >
                      Connect Repository
                    </Button>
                    
                    {isUploading && (
                      <Progress value={uploadProgress} size="sm" colorScheme="brand" borderRadius="md" />
                    )}
                  </VStack>
                </CardBody>
              </Card>
            </VStack>
          </TabPanel>
        </TabPanels>
      </Tabs>

      {/* Uploaded Files Section */}
      {uploadedFiles.length > 0 && (
        <Box mt={8}>
          <Text fontSize="lg" fontWeight="bold" mb={4}>Uploaded Knowledge Sources</Text>
          <VStack spacing={3} align="stretch">
            {uploadedFiles.map(file => (
              <Card key={file.id} variant="outline" _hover={{ shadow: 'sm' }}>
                <CardBody py={3} px={4}>
                  <HStack justify="space-between">
                    <HStack>
                      <Icon 
                        as={getFileIcon(file.type)} 
                        color={`${getFileColor(file.type)}.500`} 
                        boxSize={5} 
                      />
                      <VStack align="start" spacing={0}>
                        <Text fontWeight="medium">{file.name}</Text>
                        <Text fontSize="xs" color="gray.500">
                          Uploaded {new Date(file.uploadedAt).toLocaleString()}
                        </Text>
                      </VStack>
                    </HStack>
                    <Badge colorScheme="green">
                      <HStack spacing={1}>
                        <Icon as={IoCheckmarkCircle} />
                        <Text>Processed</Text>
                      </HStack>
                    </Badge>
                  </HStack>
                </CardBody>
              </Card>
            ))}
          </VStack>
        </Box>
      )}
    </Box>
  )
}

export default FileUploadComponent 