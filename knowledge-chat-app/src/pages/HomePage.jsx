import { 
  Box, 
  Heading, 
  Text, 
  SimpleGrid, 
  Flex, 
  VStack, 
  Button, 
  Icon, 
  HStack, 
  Container, 
  Divider, 
  useColorModeValue, 
  Image,
  Card, 
  CardBody
} from '@chakra-ui/react'
import { 
  IoAnalytics, 
  IoCodeSlash, 
  IoServer, 
  IoLayers, 
  IoArrowForward, 
  IoBulb, 
  IoCheckmarkCircle 
} from 'react-icons/io5'
import { Link } from 'react-router-dom'

const HomePage = () => {
  const gradientText = useColorModeValue(
    'linear(to-r, purple.600, blue.500)',
    'linear(to-r, purple.400, blue.300)'
  )
  const cardBg = useColorModeValue('white', 'gray.700')
  const stepBg = useColorModeValue('blue.50', 'blue.900')
  const stepBorder = useColorModeValue('blue.100', 'blue.700')
  
  return (
    <Box>
      {/* Hero Section */}
      <Box 
        bg={useColorModeValue('gray.50', 'gray.900')} 
        pt={20} 
        pb={16} 
        px={4}
      >
        <Container maxW="1200px">
          <VStack spacing={6} align="center" textAlign="center">
            <Heading 
              as="h1" 
              size="3xl" 
              fontWeight="bold" 
              lineHeight="1.2" 
              bgGradient={gradientText}
              bgClip="text"
            >
              Data-Driven Architecture Insights
            </Heading>
            
            <Text fontSize="xl" maxW="800px" color={useColorModeValue('gray.600', 'gray.300')}>
              Transform your database schemas and code repositories into actionable architecture roadmaps for development and testing strategies.
            </Text>
            
            <HStack spacing={4} pt={6}>
              <Button 
                as={Link} 
                to="/chat" 
                size="lg" 
                colorScheme="purple" 
                rightIcon={<IoAnalytics />}
                boxShadow="md"
                _hover={{ transform: 'translateY(-2px)', boxShadow: 'lg' }}
              >
                Start Analyzing
              </Button>
              
              <Button 
                as={Link} 
                to="/history" 
                size="lg" 
                variant="outline" 
                colorScheme="blue"
                _hover={{ bg: 'blue.50' }}
              >
                View Past Analyses
              </Button>
            </HStack>
          </VStack>
        </Container>
      </Box>
      
      {/* How It Works Section */}
      <Box py={16} px={4}>
        <Container maxW="1200px">
          <VStack spacing={12}>
            <VStack spacing={3} textAlign="center">
              <Heading size="xl" color={useColorModeValue('gray.700', 'white')}>
                How It Works
              </Heading>
              <Text fontSize="lg" color={useColorModeValue('gray.600', 'gray.300')} maxW="800px">
                Our AI-powered system helps you understand your data structures and provides intelligent architecture recommendations
              </Text>
            </VStack>
            
            {/* Process Steps */}
            <Box w="100%" position="relative">
              {/* Connecting line between steps */}
              <Box 
                position="absolute" 
                top="120px" 
                left="50px" 
                right="50px" 
                height="2px" 
                bg={useColorModeValue('blue.100', 'blue.700')}
                zIndex={0}
                display={{ base: 'none', md: 'block' }}
              />
              
              <SimpleGrid columns={{ base: 1, md: 3 }} spacing={8} position="relative" zIndex={1}>
                {/* Step 1 */}
                <VStack 
                  bg={cardBg} 
                  p={6} 
                  borderRadius="xl" 
                  boxShadow="md" 
                  spacing={4}
                  position="relative"
                >
                  <Flex 
                    bg={stepBg} 
                    borderRadius="full" 
                    w="60px" 
                    h="60px" 
                    align="center" 
                    justify="center"
                    border="2px solid"
                    borderColor={stepBorder}
                  >
                    <Icon as={IoLayers} boxSize={6} color="blue.500" />
                  </Flex>
                  <Text fontWeight="bold" fontSize="xl">
                    Upload your schema
                  </Text>
                  <Text textAlign="center" color={useColorModeValue('gray.600', 'gray.300')}>
                    Upload database schemas to analyze your data architecture
                  </Text>
                  <Box 
                    position="absolute" 
                    right="-20px" 
                    top="120px" 
                    display={{ base: 'none', md: 'block' }}
                    zIndex={2}
                  >
                    <Icon as={IoArrowForward} boxSize={6} color="blue.500" />
                  </Box>
                </VStack>
                
                {/* Step 2 */}
                <VStack 
                  bg={cardBg} 
                  p={6} 
                  borderRadius="xl" 
                  boxShadow="md" 
                  spacing={4}
                  position="relative"
                >
                  <Flex 
                    bg={stepBg} 
                    borderRadius="full" 
                    w="60px" 
                    h="60px" 
                    align="center" 
                    justify="center"
                    border="2px solid"
                    borderColor={stepBorder}
                  >
                    <Icon as={IoCodeSlash} boxSize={6} color="blue.500" />
                  </Flex>
                  <Text fontWeight="bold" fontSize="xl">
                    Add Business Context
                  </Text>
                  <Text textAlign="center" color={useColorModeValue('gray.600', 'gray.300')}>
                    Link to GitHub repositories and describe your business requirements to enhance analysis
                  </Text>
                  <Box 
                    position="absolute" 
                    right="-20px" 
                    top="120px" 
                    display={{ base: 'none', md: 'block' }}
                    zIndex={2}
                  >
                    <Icon as={IoArrowForward} boxSize={6} color="blue.500" />
                  </Box>
                </VStack>
                
                {/* Step 3 */}
                <VStack 
                  bg={cardBg} 
                  p={6} 
                  borderRadius="xl" 
                  boxShadow="md" 
                  spacing={4}
                >
                  <Flex 
                    bg={stepBg} 
                    borderRadius="full" 
                    w="60px" 
                    h="60px" 
                    align="center" 
                    justify="center"
                    border="2px solid"
                    borderColor={stepBorder}
                  >
                    <Icon as={IoBulb} boxSize={6} color="blue.500" />
                  </Flex>
                  <Text fontWeight="bold" fontSize="xl">
                    Get Architecture Insights
                  </Text>
                  <Text textAlign="center" color={useColorModeValue('gray.600', 'gray.300')}>
                    Receive detailed data architecture recommendations and development roadmaps
                  </Text>
                </VStack>
              </SimpleGrid>
            </Box>
          </VStack>
        </Container>
      </Box>
      
      {/* Features Section */}
      <Box bg={useColorModeValue('gray.50', 'gray.800')} py={16} px={4}>
        <Container maxW="1200px">
          <VStack spacing={12}>
            <VStack spacing={3} textAlign="center">
              <Heading size="xl" color={useColorModeValue('gray.700', 'white')}>
                Key Features
              </Heading>
              <Text fontSize="lg" color={useColorModeValue('gray.600', 'gray.300')} maxW="800px">
                Our platform provides comprehensive tools for understanding and improving your data architecture
              </Text>
            </VStack>
            
            <SimpleGrid columns={{ base: 1, md: 2 }} spacing={8}>
              <Feature 
                icon={IoServer}
                title="Data Architecture Analysis"
                description="Get detailed analysis of your database schemas with recommendations for optimization and improvement"
                iconColor="purple"
              />
              
              <Feature 
                icon={IoCodeSlash}
                title="Code Integration"
                description="Connect your GitHub repositories to understand how your code interacts with your data structures"
                iconColor="blue"
              />
              
              <Feature 
                icon={IoAnalytics}
                title="Development Roadmaps"
                description="Generate clear development roadmaps based on your data architecture and business requirements"
                iconColor="teal"
              />
              
              <Feature 
                icon={IoCheckmarkCircle}
                title="Testing Strategy"
                description="Create comprehensive testing strategies that ensure your data models work as expected"
                iconColor="green"
              />
            </SimpleGrid>
          </VStack>
        </Container>
      </Box>
      
      {/* Call to Action */}
      <Box py={16} px={4}>
        <Container maxW="1000px">
          <Card
            direction={{ base: 'column', md: 'row' }}
            overflow='hidden'
            variant='outline'
            bg={useColorModeValue('blue.50', 'blue.900')}
            borderColor={useColorModeValue('blue.100', 'blue.700')}
          >
            <CardBody p={8}>
              <VStack align="start" spacing={4}>
                <Heading size="lg">Ready to transform your data architecture?</Heading>
                <Text>
                  Start chatting with our AI assistant to get instant insights about your database schemas and code repositories.
                </Text>
                <Button 
                  as={Link} 
                  to="/chat" 
                  colorScheme="purple" 
                  size="lg" 
                  mt={2}
                  rightIcon={<IoArrowForward />}
                >
                  Start Now
                </Button>
              </VStack>
            </CardBody>
          </Card>
        </Container>
      </Box>
    </Box>
  )
}

// Feature component for the features section
const Feature = ({ icon, title, description, iconColor }) => {
  const cardBg = useColorModeValue('white', 'gray.700')
  
  return (
    <Card bg={cardBg} boxShadow="md" borderRadius="lg" overflow="hidden" h="100%">
      <CardBody p={6}>
        <HStack spacing={4} align="flex-start">
          <Flex 
            bg={`${iconColor}.50`}
            color={`${iconColor}.500`}
            p={3}
            borderRadius="lg"
            align="center"
            justify="center"
            minW="50px"
            minH="50px"
          >
            <Icon as={icon} boxSize={6} />
          </Flex>
          <VStack align="start" spacing={2}>
            <Heading size="md">{title}</Heading>
            <Text color={useColorModeValue('gray.600', 'gray.300')}>
              {description}
            </Text>
          </VStack>
        </HStack>
      </CardBody>
    </Card>
  )
}

export default HomePage 