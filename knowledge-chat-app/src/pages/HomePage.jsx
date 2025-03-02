import { Box, Heading, Text, SimpleGrid, Card, CardBody, CardHeader, Icon, Button, Flex, VStack, useColorModeValue } from '@chakra-ui/react'
import { IoDocumentText, IoChatbubble, IoLogoGithub, IoServer } from 'react-icons/io5'
import { Link } from 'react-router-dom'

const FeatureCard = ({ icon, title, description, to, iconColor }) => {
  const cardBg = useColorModeValue('white', 'gray.700')
  const hoverBg = useColorModeValue('gray.50', 'gray.600')
  
  return (
    <Card 
      as={Link} 
      to={to} 
      bg={cardBg}
      borderRadius="xl"
      overflow="hidden"
      transition="all 0.3s"
      _hover={{ 
        transform: 'translateY(-4px)', 
        shadow: 'md',
        bg: hoverBg
      }}
      height="100%"
    >
      <CardHeader pb={0}>
        <Flex align="center" mb={2}>
          <Box 
            bg={`${iconColor}.50`} 
            color={`${iconColor}.500`} 
            p={2} 
            borderRadius="md" 
            mr={3}
          >
            <Icon as={icon} boxSize={6} />
          </Box>
          <Heading size="md">{title}</Heading>
        </Flex>
      </CardHeader>
      <CardBody>
        <Text color="gray.600">{description}</Text>
      </CardBody>
    </Card>
  )
}

const HomePage = () => {
  return (
    <Box maxW="1200px" mx="auto" py={12} px={4}>
      <VStack spacing={8} align="start" mb={12}>
        <Heading 
          size="2xl" 
          fontWeight="bold" 
          bgGradient="linear(to-r, brand.500, brand.700)" 
          bgClip="text"
        >
          Welcome to Knowledge Chat
        </Heading>
        <Text fontSize="xl" color="gray.600" maxW="800px">
          Your enterprise knowledge base and AI assistant in one place. Store your knowledge and chat with AI to get answers to your questions.
        </Text>
        <Button 
          as={Link} 
          to="/chat" 
          size="lg" 
          colorScheme="brand" 
          rightIcon={<IoChatbubble />}
          shadow="md"
        >
          Start Chatting
        </Button>
      </VStack>
      
      <SimpleGrid columns={{ base: 1, md: 2, lg: 4 }} spacing={8}>
        <FeatureCard
          icon={IoDocumentText}
          title="Knowledge Store"
          description="Store and organize your enterprise knowledge in one place. Add documents, code, and more."
          to="/knowledge"
          iconColor="brand"
        />
        
        <FeatureCard
          icon={IoChatbubble}
          title="AI Chat"
          description="Chat with AI to get answers based on your organization's knowledge base."
          to="/chat"
          iconColor="blue"
        />
        
        <FeatureCard
          icon={IoLogoGithub}
          title="GitHub Connector"
          description="Connect to your GitHub repositories to include code and documentation."
          to="/connectors/github"
          iconColor="purple"
        />
        
        <FeatureCard
          icon={IoServer}
          title="Snowflake Connector"
          description="Connect to your Snowflake database to query and analyze your data."
          to="/connectors/snowflake"
          iconColor="cyan"
        />
      </SimpleGrid>
    </Box>
  )
}

export default HomePage 