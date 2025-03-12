import React from 'react';
import { Box } from '@chakra-ui/react';
import AppHeader from '../components/AppHeader';

const MainLayout = ({ children }) => {
  return (
    <>
      <AppHeader />
      <Box as="main" pt={4}>
        {children}
      </Box>
    </>
  );
};

export default MainLayout; 