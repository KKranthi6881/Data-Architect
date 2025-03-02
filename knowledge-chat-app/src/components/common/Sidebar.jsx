import { Box, VStack, Text, Divider, Icon, Flex, Accordion, AccordionItem, AccordionButton, AccordionPanel, AccordionIcon } from '@chakra-ui/react'
import { NavLink as RouterLink } from 'react-router-dom'
import { FaHome, FaBook, FaRobot, FaGithub, FaDatabase, FaFileUpload, FaCode, FaSnowflake, FaFilePdf } from 'react-icons/fa'

const NavItem = ({ icon, children, to }) => {
  return (
    <Box
      as={RouterLink}
      to={to}
      display="block"
      p={2}
      borderRadius="md"
      _hover={{ bg: 'gray.100' }}
      _activeLink={{ bg: 'brand.50', color: 'brand.600', fontWeight: 'bold' }}
      width="100%"
    >
      <Flex align="center">
        <Icon as={icon} mr={3} />
        <Text>{children}</Text>
      </Flex>
    </Box>
  )
}

const Sidebar = () => {
  return (
    <Box
      as="aside"
      w="240px"
      bg="white"
      borderRight="1px"
      borderColor="gray.200"
      p={4}
      display={{ base: 'none', md: 'block' }}
      overflowY="auto"
      h="calc(100vh - 64px)"
    >
      <VStack align="stretch" spacing={1}>
        <NavItem icon={FaHome} to="/">Home</NavItem>
        
        <Accordion allowToggle defaultIndex={[0]} borderWidth={0}>
          <AccordionItem border="none">
            <AccordionButton px={2} py={1} _hover={{ bg: 'gray.100' }} borderRadius="md">
              <Flex align="center" flex="1">
                <Icon as={FaBook} mr={3} />
                <Text>Knowledge Store</Text>
              </Flex>
              <AccordionIcon />
            </AccordionButton>
            <AccordionPanel pb={2} pt={0} pl={6}>
              <VStack align="stretch" spacing={1}>
                <NavItem icon={FaGithub} to="/knowledge/github">GitHub</NavItem>
                <NavItem icon={FaFileUpload} to="/knowledge/upload">File Upload</NavItem>
                <NavItem icon={FaCode} to="/knowledge/sql">SQL Scripts</NavItem>
                <NavItem icon={FaFilePdf} to="/knowledge/pdf">PDF Documents</NavItem>
              </VStack>
            </AccordionPanel>
          </AccordionItem>
        </Accordion>
        
        <NavItem icon={FaRobot} to="/chat">AI Chat</NavItem>
        
        <Accordion allowToggle borderWidth={0}>
          <AccordionItem border="none">
            <AccordionButton px={2} py={1} _hover={{ bg: 'gray.100' }} borderRadius="md">
              <Flex align="center" flex="1">
                <Icon as={FaDatabase} mr={3} />
                <Text>Connectors</Text>
              </Flex>
              <AccordionIcon />
            </AccordionButton>
            <AccordionPanel pb={2} pt={0} pl={6}>
              <VStack align="stretch" spacing={1}>
                <NavItem icon={FaSnowflake} to="/connectors/snowflake">Snowflake</NavItem>
                <NavItem icon={FaGithub} to="/connectors/github">GitHub</NavItem>
              </VStack>
            </AccordionPanel>
          </AccordionItem>
        </Accordion>
      </VStack>
      
      <Divider my={4} />
      
      <Text fontSize="sm" color="gray.500" mb={2}>Recent Knowledge</Text>
      <VStack align="stretch" spacing={1}>
        <Text fontSize="sm" p={2} _hover={{ bg: 'gray.100' }} cursor="pointer">
          React Basics
        </Text>
        <Text fontSize="sm" p={2} _hover={{ bg: 'gray.100' }} cursor="pointer">
          JavaScript ES6
        </Text>
      </VStack>
    </Box>
  )
}

export default Sidebar 