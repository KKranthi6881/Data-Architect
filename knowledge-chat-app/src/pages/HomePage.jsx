import React from 'react'
import { 
  Box, 
  Container,
  Flex,
  HStack,
  Heading, 
  Text, 
  VStack, 
  Button,
  useColorModeValue,
  Image,
  Tooltip
} from '@chakra-ui/react'
import { Link } from 'react-router-dom'
import { IoLogoGithub } from 'react-icons/io5'

const HomePage = () => {
  console.log('HomePage rendering...');
  
  const bgColor = useColorModeValue('white', 'gray.900')
  const textColor = useColorModeValue('gray.900', 'white')

  return (
    <Box bg={bgColor} minH="100vh">
      {/* Navigation */}
      <Box borderBottom="1px" borderColor="gray.100" py={4}>
        <Container maxW="container.xl">
          <Flex justify="space-between" align="center">
            <HStack spacing={2}>
              <Heading size="lg">
                Data
                <Text as="span" color="orange.500" fontWeight="bold">
                  NEURO
                </Text>
                <Text as="span" color="gray.500" fontSize="lg">
                  .AI
                </Text>
              </Heading>
              
              <HStack spacing={4} ml={8}>
                <Text color="gray.400" fontSize="sm">powered by</Text>
                <Tooltip label="Snowflake">
                  <Image
                    src="https://www.snowflake.com/wp-content/themes/snowflake/assets/img/logo.svg"
                    alt="Snowflake"
                    height="20px"
                    opacity={0.7}
                    _hover={{ opacity: 1 }}
                  />
                </Tooltip>
                <Text color="gray.400">+</Text>
                <Tooltip label="dbt">
                  <Image
                    src="https://www.getdbt.com/ui/img/logos/dbt-logo.svg"
                    alt="dbt"
                    height="20px"
                    opacity={0.7}
                    _hover={{ opacity: 1 }}
                  />
                </Tooltip>
              </HStack>
            </HStack>

            <HStack spacing={4}>
              <Button 
                as="a"
                href="https://github.com/yourusername/knowledge-chat"
                target="_blank"
                variant="ghost" 
                color="gray.500"
                leftIcon={<IoLogoGithub />}
              >
                GitHub
              </Button>
              <Button 
                as={Link} 
                to="/chat"
                colorScheme="orange"
              >
                Try Data Architect
              </Button>
            </HStack>
          </Flex>
        </Container>
      </Box>

      {/* Hero Section */}
      <Container maxW="container.xl" py={20}>
        <VStack spacing={8} align="center" textAlign="center">
          <Heading 
            size="3xl" 
            color={textColor}
            lineHeight="1.2"
          >
            Data Architecture{' '}
            <Text as="span" color="orange.500">
              Made Simple
            </Text>
          </Heading>

          <Text fontSize="xl" color="gray.600" maxW="800px">
            Open-source{' '}
            <Text as="span" color="orange.500" fontWeight="bold">
              Data Architect Agent
            </Text>
            {' '}specialized in Snowflake and dbt development
          </Text>

          <HStack spacing={4}>
            <Button
              as={Link}
              to="/chat"
              size="lg"
              colorScheme="orange"
            >
              Try Data Architect
            </Button>
            <Button
              as={Link}
              to="/history"
              size="lg"
              variant="outline"
              colorScheme="orange"
            >
              View Examples
            </Button>
          </HStack>
        </VStack>
      </Container>
    </Box>
  )
}

export default HomePage 