import { useState } from 'react'
import {
  Box,
  Heading,
  Text,
  SimpleGrid,
  Card,
  CardBody,
  CardHeader,
  Icon,
  Input,
  InputGroup,
  InputLeftElement,
  Button,
  Flex,
  Tabs,
  TabList,
  TabPanels,
  Tab,
  TabPanel,
  VStack,
  HStack,
  Badge,
  useColorModeValue,
  Tag,
  TagLabel
} from '@chakra-ui/react'
import { IoSearch, IoLogoGithub, IoCloudUpload, IoCodeSlash, IoDocument, IoFolder, IoAdd } from 'react-icons/io5'
import { Link } from 'react-router-dom'

const KnowledgeStorePage = () => {
  const [searchQuery, setSearchQuery] = useState('')
  const cardBg = useColorModeValue('white', 'gray.700')
  
  // Sample knowledge items
  const knowledgeItems = [
    {
      id: '1',
      title: 'React Hooks Guide',
      type: 'document',
      source: 'upload',
      tags: ['react', 'javascript', 'frontend'],
      createdAt: '2023-06-15T10:30:00Z'
    },
    {
      id: '2',
      title: 'SQL Query Optimization',
      type: 'script',
      source: 'sql',
      tags: ['database', 'performance', 'sql'],
      createdAt: '2023-07-22T14:45:00Z'
    },
    {
      id: '3',
      title: 'User Authentication Flow',
      type: 'pdf',
      source: 'pdf',
      tags: ['security', 'authentication', 'design'],
      createdAt: '2023-08-05T09:15:00Z'
    },
    {
      id: '4',
      title: 'API Documentation',
      type: 'repository',
      source: 'github',
      tags: ['api', 'documentation', 'backend'],
      createdAt: '2023-09-10T16:20:00Z'
    }
  ]
  
  // Filter knowledge items based on search query
  const filteredItems = knowledgeItems.filter(item => 
    item.title.toLowerCase().includes(searchQuery.toLowerCase()) ||
    item.tags.some(tag => tag.toLowerCase().includes(searchQuery.toLowerCase()))
  )
  
  // Get icon based on item type
  const getItemIcon = (type) => {
    switch(type) {
      case 'document': return IoDocument
      case 'script': return IoCodeSlash
      case 'pdf': return IoDocument
      case 'repository': return IoLogoGithub
      default: return IoDocument
    }
  }
  
  // Get color based on source
  const getSourceColor = (source) => {
    switch(source) {
      case 'github': return 'purple'
      case 'sql': return 'blue'
      case 'pdf': return 'red'
      case 'upload': return 'green'
      default: return 'gray'
    }
  }
  
  return (
    <Box maxW="1200px" mx="auto" py={8} px={4}>
      <Flex justify="space-between" align="center" mb={8}>
        <Heading 
          color="gray.800" 
          size="lg"
          bgGradient="linear(to-r, brand.500, brand.700)" 
          bgClip="text"
        >
          Knowledge Store
        </Heading>
        <InputGroup maxW="400px">
          <InputLeftElement pointerEvents="none">
            <IoSearch color="gray.400" />
          </InputLeftElement>
          <Input 
            placeholder="Search knowledge..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            bg={cardBg}
            borderRadius="full"
            shadow="sm"
          />
        </InputGroup>
      </Flex>
      
      <Tabs colorScheme="brand" mb={8} variant="soft-rounded">
        <TabList>
          <Tab>All</Tab>
          <Tab>Documents</Tab>
          <Tab>Scripts</Tab>
          <Tab>PDFs</Tab>
          <Tab>Repositories</Tab>
        </TabList>
        
        <TabPanels>
          <TabPanel px={0}>
            <Flex justify="space-between" align="center" mb={6}>
              <Text fontSize="lg" fontWeight="medium" color="gray.700">
                All Knowledge Items ({filteredItems.length})
              </Text>
              <HStack>
                <Button 
                  as={Link} 
                  to="/knowledge/upload" 
                  leftIcon={<IoCloudUpload />} 
                  colorScheme="brand" 
                  size="sm"
                  shadow="sm"
                >
                  Upload File
                </Button>
                <Button 
                  as={Link} 
                  to="/knowledge/github" 
                  leftIcon={<IoLogoGithub />} 
                  colorScheme="brand" 
                  size="sm"
                  variant="outline"
                >
                  Add Repository
                </Button>
              </HStack>
            </Flex>
            
            {filteredItems.length === 0 ? (
              <Box 
                textAlign="center" 
                py={12} 
                bg={cardBg} 
                borderRadius="xl" 
                shadow="sm"
              >
                <Icon as={IoFolder} boxSize={12} color="gray.300" mb={4} />
                <Text color="gray.500" fontSize="lg">
                  No knowledge items found. Try a different search or add new items.
                </Text>
                <Button 
                  mt={6} 
                  leftIcon={<IoAdd />} 
                  colorScheme="brand" 
                  size="md"
                  as={Link}
                  to="/knowledge/upload"
                >
                  Add Knowledge
                </Button>
              </Box>
            ) : (
              <SimpleGrid columns={{ base: 1, md: 2, lg: 3 }} spacing={6}>
                {filteredItems.map(item => (
                  <Card 
                    key={item.id} 
                    _hover={{ transform: 'translateY(-4px)', shadow: 'md' }} 
                    transition="all 0.3s"
                    borderRadius="xl"
                    overflow="hidden"
                    bg={cardBg}
                  >
                    <CardHeader pb={2}>
                      <Flex justify="space-between" align="start">
                        <HStack>
                          <Box 
                            bg={`${getSourceColor(item.source)}.50`} 
                            color={`${getSourceColor(item.source)}.500`} 
                            p={2} 
                            borderRadius="md" 
                          >
                            <Icon as={getItemIcon(item.type)} boxSize={5} />
                          </Box>
                          <Heading size="sm">
                            {item.title}
                          </Heading>
                        </HStack>
                        <Badge 
                          colorScheme={getSourceColor(item.source)}
                          borderRadius="full"
                          px={2}
                          py={1}
                          fontSize="xs"
                        >
                          {item.source}
                        </Badge>
                      </Flex>
                    </CardHeader>
                    <CardBody pt={2}>
                      <VStack align="start" spacing={3}>
                        <HStack spacing={1} flexWrap="wrap">
                          {item.tags.map(tag => (
                            <Tag 
                              key={tag} 
                              size="sm" 
                              borderRadius="full" 
                              variant="subtle"
                              colorScheme="gray"
                              mt={1}
                              mr={1}
                            >
                              <TagLabel>{tag}</TagLabel>
                            </Tag>
                          ))}
                        </HStack>
                        <Text fontSize="xs" color="gray.500">
                          Added on {new Date(item.createdAt).toLocaleDateString()}
                        </Text>
                      </VStack>
                    </CardBody>
                  </Card>
                ))}
              </SimpleGrid>
            )}
          </TabPanel>
          
          {/* Other tab panels would be similar but filtered by type */}
          <TabPanel px={0}>
            <Text>Documents content</Text>
          </TabPanel>
          <TabPanel px={0}>
            <Text>Scripts content</Text>
          </TabPanel>
          <TabPanel px={0}>
            <Text>PDFs content</Text>
          </TabPanel>
          <TabPanel px={0}>
            <Text>Repositories content</Text>
          </TabPanel>
        </TabPanels>
      </Tabs>
    </Box>
  )
}

export default KnowledgeStorePage 