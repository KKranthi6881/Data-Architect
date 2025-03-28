import React, { useState, useEffect, useMemo, useRef } from 'react';
import {
  Box,
  Text,
  VStack,
  HStack,
  Icon,
  Badge,
  Tag,
  TagLabel,
  Heading,
  Button,
  ButtonGroup,
  Tooltip,
  Flex,
  Code,
  Divider,
  IconButton,
  Menu,
  MenuButton,
  MenuList,
  MenuItem,
  InputGroup,
  Input,
  InputLeftElement,
  Drawer,
  DrawerBody,
  DrawerHeader,
  DrawerOverlay,
  DrawerContent,
  DrawerCloseButton,
  useDisclosure,
  List,
  ListItem,
  Grid,
  GridItem,
  Modal,
  ModalOverlay,
  ModalContent,
  ModalHeader,
  ModalFooter,
  ModalBody,
  ModalCloseButton,
  Slider,
  SliderTrack,
  SliderFilledTrack,
  SliderThumb,
} from '@chakra-ui/react';
import { IoChevronUp, IoChevronDown, IoArrowRedo, IoArrowForward, IoArrowUp, IoArrowBack, IoArrowDown, IoSearch, IoMenu, IoList, IoExpand, IoContract, IoResize } from 'react-icons/io5';
import { MdKeyboardArrowDown, MdKeyboardArrowRight } from 'react-icons/md';
import { TbArrowsRightLeft, TbArrowsHorizontal, TbArrowsVertical, TbZoomIn, TbZoomOut, TbArrowBack, TbArrowsMaximize, TbTable, TbMapPin, TbMaximize, TbMinimize } from 'react-icons/tb';
import { FiArrowLeft, FiArrowRight } from 'react-icons/fi';

export const LineageGraph = ({ data }) => {
  // Add error boundary state
  const [hasError, setHasError] = useState(false);
  const [errorMessage, setErrorMessage] = useState("");
  
  // Add state for panning mode
  const [isPanningMode, setIsPanningMode] = useState(false);
  const [isMouseDown, setIsMouseDown] = useState(false);
  const [lastMousePosition, setLastMousePosition] = useState({ x: 0, y: 0 });
  
  // Add state for slider values
  const [horizontalSliderValue, setHorizontalSliderValue] = useState(0);
  const [verticalSliderValue, setVerticalSliderValue] = useState(0);
  
  // Add console logging to debug incoming data
  useEffect(() => {
    console.log("LineageGraph component mounted");
    console.log("Received data:", data);
    
    // Check if necessary data properties exist
    if (data) {
      console.log("Models:", data.models ? `${data.models.length} models` : "No models");
      console.log("Edges:", data.edges ? `${data.edges.length} edges` : "No edges");
      console.log("Columns:", data.columns ? `${data.columns.length} columns` : "No columns");
      console.log("Column lineage:", data.column_lineage ? `${data.column_lineage.length} connections` : "No column connections");
    }
  }, [data]);
  
  // Add global styles for animations
  useEffect(() => {
    // Add keyframes for the dash animation
    const styleSheet = document.createElement("style");
    styleSheet.id = "lineage-graph-styles";
    styleSheet.textContent = `
      @keyframes dash {
        to {
          stroke-dashoffset: -40;
        }
      }
      
      @keyframes pulse {
        0% {
          opacity: 0.6;
        }
        50% {
          opacity: 1;
        }
        100% {
          opacity: 0.6;
        }
      }
      
      @keyframes glowPulse {
        0% {
          filter: drop-shadow(0 0 2px rgba(249, 115, 22, 0.3));
        }
        50% {
          filter: drop-shadow(0 0 5px rgba(249, 115, 22, 0.5));
        }
        100% {
          filter: drop-shadow(0 0 2px rgba(249, 115, 22, 0.3));
        }
      }
    `;
    
    // Only add if it doesn't exist yet
    if (!document.getElementById("lineage-graph-styles")) {
      document.head.appendChild(styleSheet);
    }
    
    // Cleanup on unmount
    return () => {
      const existingStyle = document.getElementById("lineage-graph-styles");
      if (existingStyle) {
        existingStyle.remove();
      }
    };
  }, []);
  
  // Default to sample data if none provided
  const graphData = data && (!hasError) ? data : {
    models: [
      { id: 'model1', name: 'order_data', path: 'models/staging/order_data.sql', type: 'staging', highlight: true },
      { id: 'model2', name: 'order_items', path: 'models/intermediate/order_items.sql', type: 'intermediate', highlight: false },
      { id: 'model3', name: 'financial_summary', path: 'models/marts/financial_summary.sql', type: 'mart', highlight: false },
    ],
    edges: [
      { source: 'model1', target: 'model2' },
      { source: 'model2', target: 'model3' },
    ],
    // Add sample columns to ensure column visualization works
    columns: [
      { id: 'col1', modelId: 'model1', name: 'order_id', type: 'primary_key', dataType: 'integer', highlight: false },
      { id: 'col2', modelId: 'model1', name: 'customer_id', type: 'foreign_key', dataType: 'integer', highlight: false },
      { id: 'col3', modelId: 'model2', name: 'order_id', type: 'foreign_key', dataType: 'integer', highlight: false },
      { id: 'col4', modelId: 'model2', name: 'product_id', type: 'foreign_key', dataType: 'integer', highlight: false },
      { id: 'col5', modelId: 'model3', name: 'order_revenue', type: 'regular', dataType: 'numeric', highlight: false },
    ],
    // Add sample column lineage
    column_lineage: [
      { source: 'col1', target: 'col3' },
      { source: 'col4', target: 'col5' },
    ]
  };

  // Add layout state variable
  const [layoutMode, setLayoutMode] = useState('horizontal'); // 'horizontal' or 'vertical'

  // Validate models and edges before processing
  useEffect(() => {
    if (data) {
      try {
        console.log("LineageGraph received data:", data);
        
        // Check if data has valid models and edges
        const valid = 
          data.models && Array.isArray(data.models) && data.models.length > 0 &&
          data.edges && Array.isArray(data.edges);
        
        if (!valid) {
          console.error("LineageGraph received invalid data structure:", data);
          setHasError(true);
          setErrorMessage("Invalid lineage data structure. Missing models or edges.");
        } else {
          // Check if models have required properties
          const modelsValid = data.models.every(model => 
            model.id && (typeof model.id === 'string' || typeof model.id === 'number')
          );
          
          // Check if edges have required properties and reference valid models
          if (data.edges.length > 0) {
            const modelIds = new Set(data.models.map(m => m.id));
            const edgesValid = data.edges.every(edge => 
              edge.source && edge.target && 
              modelIds.has(edge.source) && modelIds.has(edge.target)
            );
            
            if (!modelsValid || !edgesValid) {
              console.error("LineageGraph data has invalid models or edges:", 
                {modelsValid, edgesValid, models: data.models, edges: data.edges});
              setHasError(true);
              setErrorMessage("Invalid models or edges in lineage data.");
            }
          }
          
          // Handle case where there are models but no edges
          if (data.edges.length === 0 && data.models.length > 0) {
            console.warn("LineageGraph data has no edges, but has models. This is unusual for lineage data.");
          }
        }
      } catch (err) {
        console.error("Error validating lineage data:", err);
        setHasError(true);
        setErrorMessage(`Error: ${err.message}`);
      }
    }
  }, [data]);

  // Function to determine which models should be expanded initially
  const getInitialExpandedModels = () => {
    try {
      const result = {};
      
      // All models collapsed by default
      graphData.models.forEach(model => {
        result[model.id] = false;
      });
      
      console.log("Initial expanded models:", result);
      return result;
    } catch (err) {
      console.error("Error in getInitialExpandedModels:", err);
      return {};
    }
  };
  
  // State for tracking expanded models, active edge, and hover state
  const [expandedModels, setExpandedModels] = useState(getInitialExpandedModels());
  const [activeEdge, setActiveEdge] = useState(null);
  const [activeModelId, setActiveModelId] = useState(null);
  const [activeColumnLink, setActiveColumnLink] = useState(null);

  // Function to toggle model expansion
  const toggleModelExpansion = (modelId) => {
    expandAndFocusModel(modelId);
  };

  // Get color for model based on its type
  const getModelTypeColor = (type) => {
    switch (type) {
      case 'staging':
        return '#3B82F6'; // Modern blue
      case 'intermediate':
        return '#8B5CF6'; // Modern purple
      case 'mart':
        return '#10B981'; // Modern green
      case 'source':
        return '#F59E0B'; // Modern amber
      case 'upstream':
        return '#06B6D4'; // Modern cyan
      case 'table':
        return '#0EA5E9'; // Modern sky blue
      default:
        return '#64748B'; // Modern slate
    }
  };

  // First, let's update the color palette for column connections with more subtle colors
  const modernColors = {
    blue: {
      light: '#EBF8FF',
      medium: '#4299E1',
      dark: '#2B6CB0'
    },
    purple: {
      light: '#F3E8FF',
      medium: '#9F7AEA',
      dark: '#6B46C1'
    },
    green: {
      light: '#E6FFFA',
      medium: '#38B2AC',
      dark: '#285E61'
    },
    orange: {
      light: '#FFFAF0',
      medium: '#ED8936',
      dark: '#C05621'
    },
    gray: {
      light: '#F7FAFC',
      medium: '#A0AEC0',
      dark: '#4A5568'
    },
    connection: {
      default: 'rgba(160, 174, 192, 0.6)',
      active: '#ED8936',
      text: '#4A5568',
      highlight: 'rgba(237, 137, 54, 0.1)'
    }
  };
  
  // Function to get color for column type
  const getColumnTypeColor = (type, isActive = false) => {
    if (isActive) {
      return modernColors.orange.medium;
    }
    
    switch (type) {
      case 'primary_key':
        return modernColors.purple.medium;
      case 'foreign_key':
        return modernColors.blue.medium;
      case 'timestamp':
        return modernColors.green.medium;
      default:
        return modernColors.gray.medium;
    }
  };
  
  // Function to get column type icon and label
  const getColumnTypeInfo = (type) => {
    switch (type) {
      case 'primary_key':
        return { icon: '🔑', label: 'PK' };
      case 'foreign_key':
        return { icon: '🔗', label: 'FK' };
      case 'timestamp':
        return { icon: '⏱️', label: 'TS' };
      default:
        return { icon: '', label: '' };
    }
  };
  
  // Function to toggle layout mode
  const toggleLayoutMode = () => {
    setLayoutMode(prev => prev === 'horizontal' ? 'vertical' : 'horizontal');
  };
  
  // Function to get model type badge
  const getModelTypeBadge = (type) => {
    const getBadgeProps = (type) => {
      switch (type) {
        case 'staging':
          return { colorScheme: 'blue', label: 'Staging' };
        case 'intermediate':
          return { colorScheme: 'purple', label: 'Intermediate' };
        case 'mart':
          return { colorScheme: 'green', label: 'Mart' };
        case 'source':
          return { colorScheme: 'orange', label: 'Source' };
        case 'upstream':
          return { colorScheme: 'cyan', label: 'Upstream' };
        case 'table':
          return { colorScheme: 'blue', label: 'Table' };
        default:
          return { colorScheme: 'gray', label: type || 'Unknown' };
      }
    };
    
    const { colorScheme, label } = getBadgeProps(type);
    
    return (
      <Badge 
        colorScheme={colorScheme} 
        variant="subtle"
        px={2}
        py={0.5}
        borderRadius="md"
        fontWeight="medium"
        fontSize="xs"
        textTransform="lowercase"
      >
        {label}
      </Badge>
    );
  };
  
  // State for layout, scale, offset, and drag
  const [layout, setLayout] = useState({});
  const [scale, setScale] = useState(1);
  const [offset, setOffset] = useState({ x: 0, y: 0 });
  const [isDragging, setIsDragging] = useState(false);
  const [dragModelId, setDragModelId] = useState(null);
  const containerRef = useRef(null);
  
  // Calculate and memoize the column layout data - the key function that had duplicates
  const columnLayoutData = useMemo(() => {
    const data = {};
    
    // Skip if no models or columns
    if (!graphData.models || !graphData.columns || !layout) {
      return data;
    }
    
    // Group columns by model first
    const columnsByModel = {};
    graphData.columns.forEach(column => {
      if (!columnsByModel[column.modelId]) {
        columnsByModel[column.modelId] = [];
      }
      columnsByModel[column.modelId].push(column);
    });
    
    // Calculate the number of models to adjust spacing
    const modelCount = Object.keys(layout).length;
    
    // Count max columns per model to determine column spacing
    let maxColumnsPerModel = 0;
    Object.values(columnsByModel).forEach(columns => {
      maxColumnsPerModel = Math.max(maxColumnsPerModel, columns.length);
    });
    
    // Now calculate positions with better spacing
    Object.entries(columnsByModel).forEach(([modelId, columns]) => {
      const modelLayout = layout[modelId];
      if (!modelLayout) return;
      
      // Adjust spacing based on number of columns
      const columnVerticalSpacing = maxColumnsPerModel > 8 ? 25 : 35; // Less spacing for many columns
      
      // Calculate position for each column
      columns.forEach((column, index) => {
        // Calculate positions with more room between columns
        data[column.id] = {
          x: modelLayout.x + 15,
          y: modelLayout.y + 85 + (index * columnVerticalSpacing),
          width: modelLayout.width - 30,
          height: 30 // Taller column boxes
        };
      });
    });
    
    return data;
  }, [graphData.models, graphData.columns, layout]);
  
  // The renderEdges function - another key function that had duplicates
  const renderEdges = () => {
    try {
      // Create a lookup of direct relationships to avoid rendering redundant edges
      const directRelationships = new Set();
      graphData.edges.forEach(edge => {
        directRelationships.add(`${edge.source}-${edge.target}`);
      });

      // Filter out redundant edges (transitively implied relationships)
      const nonRedundantEdges = graphData.edges.filter(edge => {
        // Check if there's a path from source to target via another node
        const hasIndirectPath = graphData.edges.some(e1 => 
          e1.source === edge.source && 
          graphData.edges.some(e2 => 
            e2.source === e1.target && e2.target === edge.target
          )
        );
        
        // Keep the edge if it's not redundant or if it's part of a direct important relationship
        return !hasIndirectPath || edge.important;
      });

      return nonRedundantEdges.map((edge, idx) => {
        // Skip edge if we don't have layout info for source or target
        const sourceModel = layout[edge.source];
        const targetModel = layout[edge.target];
        
        if (!sourceModel || !targetModel) return null;
        
        // Determine if this is horizontal or vertical layout
        const isHorizontal = layoutMode === 'horizontal';
        
        // Calculate start and end points for the edge based on the layout
        let startX, startY, endX, endY;
        
        // Determine relative positions
        const sourceIsLeftOfTarget = sourceModel.x < targetModel.x;
        const sourceIsAboveTarget = sourceModel.y < targetModel.y;
        
        // Calculate center points of each model for reference
        const sourceCenter = {
          x: sourceModel.x + sourceModel.width / 2,
          y: sourceModel.y + sourceModel.height / 2
        };
        
        const targetCenter = {
          x: targetModel.x + targetModel.width / 2,
          y: targetModel.y + targetModel.height / 2
        };
        
        // Determine edge attachment points based on relative positions
        if (isHorizontal) {
          if (sourceIsLeftOfTarget) {
            // Standard left-to-right flow
            startX = sourceModel.x + sourceModel.width;
            startY = sourceCenter.y;
            endX = targetModel.x;
            endY = targetCenter.y;
          } else {
            // Reversed right-to-left flow
            startX = sourceModel.x;
            startY = sourceCenter.y;
            endX = targetModel.x + targetModel.width;
            endY = targetCenter.y;
          }
        } else {
          if (sourceIsAboveTarget) {
            // Standard top-to-bottom flow
            startX = sourceCenter.x;
            startY = sourceModel.y + sourceModel.height;
            endX = targetCenter.x;
            endY = targetModel.y;
          } else {
            // Reversed bottom-to-top flow
            startX = sourceCenter.x;
            startY = sourceModel.y;
            endX = targetCenter.x;
            endY = targetModel.y + targetModel.height;
          }
        }
        
        // Calculate control points for bezier curves
        let controlPoint1X, controlPoint1Y, controlPoint2X, controlPoint2Y;
        
        // For horizontal layout
        if (isHorizontal) {
          const distanceX = Math.abs(endX - startX);
          const distanceY = Math.abs(endY - startY);
          
          if (sourceIsLeftOfTarget) {
            // Normal left-to-right flow
            controlPoint1X = startX + distanceX * 0.33;
            controlPoint1Y = startY;
            controlPoint2X = endX - distanceX * 0.33;
            controlPoint2Y = endY;
            
            // Add vertical curve if there's significant Y difference
            if (distanceY > 20) {
              const verticalOffset = Math.min(distanceY * 0.3, 60);
              controlPoint1Y = startY + (sourceIsAboveTarget ? verticalOffset : -verticalOffset);
              controlPoint2Y = endY + (sourceIsAboveTarget ? -verticalOffset : verticalOffset);
            }
          } else {
            // Reversed right-to-left flow
            controlPoint1X = startX - distanceX * 0.33;
            controlPoint1Y = startY;
            controlPoint2X = endX + distanceX * 0.33;
            controlPoint2Y = endY;
            
            // Add vertical curve if there's significant Y difference
            if (distanceY > 20) {
              const verticalOffset = Math.min(distanceY * 0.3, 60);
              controlPoint1Y = startY + (sourceIsAboveTarget ? verticalOffset : -verticalOffset);
              controlPoint2Y = endY + (sourceIsAboveTarget ? -verticalOffset : verticalOffset);
            }
          }
        } 
        // For vertical layout
        else {
          const distanceX = Math.abs(endX - startX);
          const distanceY = Math.abs(endY - startY);
          
          if (sourceIsAboveTarget) {
            // Normal top-to-bottom flow
            controlPoint1X = startX;
            controlPoint1Y = startY + distanceY * 0.33;
            controlPoint2X = endX;
            controlPoint2Y = endY - distanceY * 0.33;
            
            // Add horizontal curve if there's significant X difference
            if (distanceX > 20) {
              const horizontalOffset = Math.min(distanceX * 0.3, 80);
              controlPoint1X = startX + (sourceIsLeftOfTarget ? horizontalOffset : -horizontalOffset);
              controlPoint2X = endX + (sourceIsLeftOfTarget ? -horizontalOffset : horizontalOffset);
            }
          } else {
            // Reverse bottom-to-top flow
            controlPoint1X = startX;
            controlPoint1Y = startY - distanceY * 0.33;
            controlPoint2X = endX;
            controlPoint2Y = endY + distanceY * 0.33;
            
            // Add horizontal curve if there's significant X difference
            if (distanceX > 20) {
              const horizontalOffset = Math.min(distanceX * 0.3, 80);
              controlPoint1X = startX + (sourceIsLeftOfTarget ? horizontalOffset : -horizontalOffset);
              controlPoint2X = endX + (sourceIsLeftOfTarget ? -horizontalOffset : horizontalOffset);
            }
          }
        }
        
        // Create the path string using bezier curve
        const path = `M ${startX} ${startY} C ${controlPoint1X} ${controlPoint1Y}, ${controlPoint2X} ${controlPoint2Y}, ${endX} ${endY}`;
        
        // Determine color based on model types
        let color, labelBg, labelColor;
        
        const sourceModelData = graphData.models.find(m => m.id === edge.source);
        const targetModelData = graphData.models.find(m => m.id === edge.target);
        
        if (sourceModelData && targetModelData) {
          // Color edge based on the model types it connects
          if (sourceModelData.type === 'source' && targetModelData.type === 'staging') {
            color = 'rgba(107, 114, 128, 0.8)'; // Gray for source-to-staging
            labelBg = 'rgba(243, 244, 246, 0.95)'; // Light gray bg
            labelColor = 'rgba(55, 65, 81, 0.9)'; // Dark gray text
          } else if (sourceModelData.type === 'staging' && targetModelData.type === 'intermediate') {
            color = 'rgba(79, 70, 229, 0.8)'; // Indigo for staging-to-intermediate
            labelBg = 'rgba(238, 242, 255, 0.95)'; // Light indigo bg
            labelColor = 'rgba(67, 56, 202, 0.9)'; // Indigo text
          } else if (targetModelData.type === 'mart') {
            color = 'rgba(16, 185, 129, 0.8)'; // Green for anything-to-mart
            labelBg = 'rgba(236, 253, 245, 0.95)'; // Light green bg
            labelColor = 'rgba(5, 150, 105, 0.9)'; // Green text
          } else {
            color = 'rgba(107, 114, 128, 0.8)'; // Default gray
            labelBg = 'rgba(243, 244, 246, 0.95)'; // Light gray bg
            labelColor = 'rgba(55, 65, 81, 0.9)'; // Dark gray text
          }
        } else {
          // Default colors
          color = 'rgba(107, 114, 128, 0.8)'; // Default gray
          labelBg = 'rgba(243, 244, 246, 0.95)'; // Light gray bg
          labelColor = 'rgba(55, 65, 81, 0.9)'; // Dark gray text
        }
        
        // Calculate label position in the middle of the path
        const labelPosition = {
          x: (startX + endX) / 2,
          y: (startY + endY) / 2
        };
        
        // Determine arrow direction based on model positions
        let arrowPath;
        if (isHorizontal) {
          arrowPath = sourceIsLeftOfTarget 
            ? "M -6,-6 L 0,0 L -6,6"  // Right arrow
            : "M 6,-6 L 0,0 L 6,6";   // Left arrow
        } else {
          arrowPath = sourceIsAboveTarget 
            ? "M -6,-6 L 0,0 L 6,-6"  // Down arrow
            : "M -6,6 L 0,0 L 6,6";   // Up arrow
        }
        
        return (
          <g key={`${edge.source}-${edge.target}`} className="edge">
            <path
              d={path}
              stroke={color}
              strokeWidth="2"
              strokeDasharray="4,0"
              fill="none"
            />
            
            {/* Arrow at the end */}
            <g transform={`translate(${endX}, ${endY})`}>
              <path 
                d={arrowPath}
                stroke={color}
                strokeWidth="2"
                strokeLinecap="round"
                strokeLinejoin="round"
                fill="none"
              />
            </g>
            
            {/* Label showing relationship on hover */}
            <g>
              <rect
                x={labelPosition.x - 40}
                y={labelPosition.y - 12}
                width="80"
                height="24"
                rx="12"
                ry="12"
                fill={labelBg}
                stroke={color}
                strokeWidth="1"
                opacity="0"
                style={{
                  transition: 'opacity 0.2s ease',
                }}
                className="edge-label"
              />
              <text
                x={labelPosition.x}
                y={labelPosition.y + 4}
                textAnchor="middle"
                fontSize="12px"
                fontWeight="500"
                fill={labelColor}
                opacity="0"
                style={{
                  transition: 'opacity 0.2s ease',
                  pointerEvents: 'none'
                }}
                className="edge-label-text"
              >
                {edge.label || 'relates to'}
              </text>
            </g>
          </g>
        );
      });
    } catch (err) {
      console.error("Error rendering edges:", err);
      return null;
    }
  };
  
  // The renderColumnConnections function - another key function that had duplicates
  const renderColumnConnections = () => {
    if (!graphData.column_lineage || graphData.column_lineage.length === 0) {
      return null;
    }

    // Group column links by model pair for better organization
    const linksByModelPair = {};
    
    graphData.column_lineage.forEach(link => {
      const sourceCol = graphData.columns.find(c => c.id === link.source);
      const targetCol = graphData.columns.find(c => c.id === link.target);
      
      if (!sourceCol || !targetCol) return;
      if (sourceCol.modelId === targetCol.modelId) return; // Skip self references
      
      const modelPairKey = `${sourceCol.modelId}-${targetCol.modelId}`;
      if (!linksByModelPair[modelPairKey]) {
        linksByModelPair[modelPairKey] = [];
      }
      linksByModelPair[modelPairKey].push({ link, sourceCol, targetCol });
    });
    
    // Render all connections with neutral styling by default
    return Object.entries(linksByModelPair).map(([modelPairKey, links]) => {
      if (links.length === 0) return null;
      
      const firstLink = links[0];
      const { sourceCol, targetCol } = firstLink;
      
      // Get layout info for the models
      const sourceModel = layout[sourceCol.modelId];
      const targetModel = layout[targetCol.modelId];
      
      if (!sourceModel || !targetModel) return null;
      
      // Determine relative positions for better connection logic
      const sourceIsLeftOfTarget = sourceModel.x < targetModel.x;
      const sourceIsAboveTarget = sourceModel.y < targetModel.y;
      
      // Get center positions for the models
      const sourceCenter = {
        x: sourceModel.x + sourceModel.width / 2,
        y: sourceModel.y + sourceModel.height / 2
      };
      
      const targetCenter = {
        x: targetModel.x + targetModel.width / 2,
        y: targetModel.y + targetModel.height / 2
      };
      
      // Calculate connection points based on model positions
      const isHorizontal = layoutMode === 'horizontal';
      
      // Determine start and end points based on layout and relative positions
      let startX, startY, endX, endY;
      
      if (isHorizontal) {
        if (sourceIsLeftOfTarget) {
          // Standard horizontal flow (left to right)
          startX = sourceModel.x + sourceModel.width;
          startY = sourceCenter.y;
          endX = targetModel.x;
          endY = targetCenter.y;
        } else {
          // Reversed horizontal flow (right to left)
          startX = sourceModel.x;
          startY = sourceCenter.y;
          endX = targetModel.x + targetModel.width;
          endY = targetCenter.y;
        }
      } else {
        if (sourceIsAboveTarget) {
          // Standard vertical flow (top to bottom)
          startX = sourceCenter.x;
          startY = sourceModel.y + sourceModel.height;
          endX = targetCenter.x;
          endY = targetModel.y;
        } else {
          // Reversed vertical flow (bottom to top)
          startX = sourceCenter.x;
          startY = sourceModel.y;
          endX = targetCenter.x;
          endY = targetModel.y + targetModel.height;
        }
      }
      
      // Calculate control points for elegant curve
      let controlPoint1X, controlPoint1Y, controlPoint2X, controlPoint2Y;
      const distanceX = Math.abs(endX - startX);
      const distanceY = Math.abs(endY - startY);
      
      if (isHorizontal) {
        controlPoint1X = startX + (sourceIsLeftOfTarget ? 1 : -1) * distanceX * 0.33;
        controlPoint1Y = startY;
        controlPoint2X = endX + (sourceIsLeftOfTarget ? -1 : 1) * distanceX * 0.33;
        controlPoint2Y = endY;
        
        // Add vertical curve if needed
        if (distanceY > 20) {
          const verticalOffset = Math.min(distanceY * 0.3, 60);
          controlPoint1Y = startY + (sourceIsAboveTarget ? verticalOffset : -verticalOffset);
          controlPoint2Y = endY + (sourceIsAboveTarget ? -verticalOffset : verticalOffset);
        }
      } else {
        controlPoint1X = startX;
        controlPoint1Y = startY + (sourceIsAboveTarget ? 1 : -1) * distanceY * 0.33;
        controlPoint2X = endX;
        controlPoint2Y = endY + (sourceIsAboveTarget ? -1 : 1) * distanceY * 0.33;
        
        // Add horizontal curve if needed
        if (distanceX > 20) {
          const horizontalOffset = Math.min(distanceX * 0.3, 80);
          controlPoint1X = startX + (sourceIsLeftOfTarget ? horizontalOffset : -horizontalOffset);
          controlPoint2X = endX + (sourceIsLeftOfTarget ? -horizontalOffset : horizontalOffset);
        }
      }
      
      // Create path
      const mainPath = `M ${startX} ${startY} C ${controlPoint1X} ${controlPoint1Y}, ${controlPoint2X} ${controlPoint2Y}, ${endX} ${endY}`;
      
      // Determine the middle point for placing the count badge
      const midX = (startX + endX) / 2;
      const midY = (startY + endY) / 2;
      
      // Improved active connection detection:
      // Check if any columns in this connection are in the active column's network
      let connectionStatus = "default";
      
      if (activeColumnLink) {
        // Check if this connection is directly related to active column
        const isDirectRelation = links.some(linkInfo => 
          linkInfo.link.source === activeColumnLink || 
          linkInfo.link.target === activeColumnLink
        );
        
        // If direct relation found, mark it as active
        if (isDirectRelation) {
          connectionStatus = "active";
        } else if (connectedColumnsInfo && connectedColumnsInfo.connected) {
          // Check if any connection in this group involves a column that's connected to our active column
          const hasIndirectRelation = links.some(linkInfo => 
            connectedColumnsInfo.connected.has(linkInfo.link.source) || 
            connectedColumnsInfo.connected.has(linkInfo.link.target)
          );
          
          if (hasIndirectRelation) {
            connectionStatus = "related";
          }
        }
      }
      
      // Apply styling based on connection status
      let connectionColor, opacity, strokeWidth, strokeDasharray;
      
      switch (connectionStatus) {
        case "active":
          connectionColor = modernColors.connection.active;
          opacity = 1;
          strokeWidth = 2;
          strokeDasharray = "none";
          break;
        case "related":
          connectionColor = "#3182CE"; // Blue
          opacity = 0.7;
          strokeWidth = 1.5;
          strokeDasharray = "4,4";
          break;
        default:
          connectionColor = modernColors.connection.default;
          opacity = activeColumnLink ? 0.2 : 0.6; // Dim non-related connections when a column is active
          strokeWidth = 1;
          strokeDasharray = "4,4";
      }
      
      // Determine arrow direction based on layout and model positions
      let arrowPath;
      if (isHorizontal) {
        arrowPath = sourceIsLeftOfTarget 
          ? "M -6,-6 L 0,0 L -6,6"  // Right arrow
          : "M 6,-6 L 0,0 L 6,6";   // Left arrow
      } else {
        arrowPath = sourceIsAboveTarget 
          ? "M -6,-6 L 0,0 L 6,-6"  // Down arrow
          : "M -6,6 L 0,0 L 6,6";   // Up arrow
      }

      return (
        <g key={modelPairKey} className="model-column-connection">
          {/* Main connection path between models */}
          <path
            d={mainPath}
            fill="none"
            stroke={connectionColor}
            strokeWidth={strokeWidth}
            opacity={opacity}
            strokeDasharray={strokeDasharray}
            style={{ 
              transition: "all 0.2s ease",
              filter: connectionStatus !== "default" ? 'drop-shadow(0 1px 2px rgba(0,0,0,0.1))' : 'none' 
            }}
          />
          
          {/* Arrow head */}
          <g transform={`translate(${endX}, ${endY})`}>
            <path
              d={arrowPath}
              fill="none"
              stroke={connectionColor}
              strokeWidth={strokeWidth}
              opacity={opacity}
              style={{ transition: "all 0.2s ease" }}
            />
          </g>
          
          {/* Count badge */}
          <g>
            <circle 
              cx={midX} 
              cy={midY} 
              r={10} 
              fill="white" 
              stroke={connectionColor} 
              strokeWidth={strokeWidth}
              filter="drop-shadow(0 1px 1px rgba(0,0,0,0.05))"
              style={{ transition: "all 0.2s ease" }}
            />
            <text
              x={midX}
              y={midY + 4}
              textAnchor="middle"
              fontSize="10px"
              fontWeight="500"
              fill={connectionStatus !== "default" ? connectionColor : modernColors.connection.text}
              style={{ transition: "all 0.2s ease" }}
            >
              {links.length}
            </text>
          </g>
        </g>
      );
    });
  };
  
  // Calculate viewport size
  const calculateViewportSize = () => {
    if (!containerRef.current) return { width: 1000, height: 800 };
    
    return {
      width: containerRef.current.clientWidth,
      height: containerRef.current.clientHeight
    };
  };
  
  // Fit view to show all models
  const fitToView = () => {
    if (Object.keys(layout).length === 0) return;
    
    const { width: containerWidth, height: containerHeight } = calculateViewportSize();
    
    // Find bounding box of all models
    let minX = Infinity;
    let minY = Infinity;
    let maxX = -Infinity;
    let maxY = -Infinity;
    
    Object.values(layout).forEach(model => {
      minX = Math.min(minX, model.x);
      minY = Math.min(minY, model.y);
      maxX = Math.max(maxX, model.x + model.width);
      maxY = Math.max(maxY, model.y + model.height);
    });
    
    // Add padding
    const padding = 50;
    minX -= padding;
    minY -= padding;
    maxX += padding;
    maxY += padding;
    
    // Calculate required scale to fit
    const scaleX = containerWidth / (maxX - minX);
    const scaleY = containerHeight / (maxY - minY);
    const newScale = Math.min(scaleX, scaleY, 1); // Limit max scale to 1
    
    // Calculate center point of models
    const centerX = (minX + maxX) / 2;
    const centerY = (minY + maxY) / 2;
    
    // Calculate offset to center the models
    const newOffsetX = (containerWidth / 2) - (centerX * newScale);
    const newOffsetY = (containerHeight / 2) - (centerY * newScale);
    
    // Update state
    setScale(newScale);
    setOffset({ x: newOffsetX, y: newOffsetY });
  };
  
  // Auto fit on mount and when layout changes
  useEffect(() => {
    // Wait for layout to be calculated
    if (Object.keys(layout).length === 0) return;
    
    // Fit view with a small delay to ensure rendering is complete
    const timer = setTimeout(() => {
      fitToView();
    }, 300);
    
    return () => clearTimeout(timer);
  }, [layout, layoutMode]);
  
  // Remove the automatic fitToView on expandedModels changes to preserve zoom levels
  // Only fit view on first mount
  useEffect(() => {
    const timer = setTimeout(() => {
      fitToView();
    }, 500);
    
    return () => clearTimeout(timer);
  }, []);
  
  // Add a function to expand and focus on a specific model
  const expandAndFocusModel = (modelId) => {
    // Toggle the expanded state
    setExpandedModels(prev => ({
      ...prev,
      [modelId]: !prev[modelId]
    }));
    
    // Give time for the layout to update before focusing
    setTimeout(() => {
      // Only focus if we're expanding the model
      if (!expandedModels[modelId]) {
        focusOnModel(modelId);
      }
    }, 300);
  };
  
  // Add a function to focus on a specific model without changing zoom level
  const focusOnModel = (modelId) => {
    const modelLayout = layout[modelId];
    if (!modelLayout) return;
    
    const { width: containerWidth, height: containerHeight } = calculateViewportSize();
    
    // Calculate center point of the model
    const modelCenterX = modelLayout.x + modelLayout.width / 2;
    const modelCenterY = modelLayout.y + modelLayout.height / 2;
    
    // Calculate new offset to center the model (keep current scale)
    const newOffsetX = (containerWidth / 2) - (modelCenterX * scale);
    const newOffsetY = (containerHeight / 2) - (modelCenterY * scale);
    
    // Update offset without changing scale
    setOffset({ x: newOffsetX, y: newOffsetY });
  };
  
  // Reset layout
  const resetLayout = () => {
    // Set zoom level back to 1
    setScale(1);
    
    // Calculate the viewport size
    const { width, height } = calculateViewportSize();
    
    // Center the layout at the origin
    setOffset({
      x: width / 2,
      y: height / 2
    });
  };
  
  // Update handleDragStart to account for panning mode
  const handleDragStart = (modelId, e) => {
    e.stopPropagation();

    // If in panning mode, don't allow model dragging
    if (isPanningMode) return;
    
    setIsDragging(true);
    setDragModelId(modelId);
    
    // Use let instead of const to allow reassignment
    let initialX = e.clientX;
    let initialY = e.clientY;
    
    const handleMouseMove = (moveEvent) => {
      const dx = (moveEvent.clientX - initialX) / scale;
      const dy = (moveEvent.clientY - initialY) / scale;
      
      setLayout(prev => ({
        ...prev,
        [modelId]: {
          ...prev[modelId],
          x: prev[modelId].x + dx,
          y: prev[modelId].y + dy
        }
      }));
      
      // Update initial positions for next movement
      initialX = moveEvent.clientX;
      initialY = moveEvent.clientY;
    };
    
    const handleMouseUp = () => {
      setIsDragging(false);
      setDragModelId(null);
      
      document.removeEventListener('mousemove', handleMouseMove);
      document.removeEventListener('mouseup', handleMouseUp);
    };
    
    document.addEventListener('mousemove', handleMouseMove);
    document.addEventListener('mouseup', handleMouseUp);
  };
  
  // Add new panning handler for the entire graph
  const handleGraphPanning = (e) => {
    if (!isPanningMode) return;
    
    e.preventDefault();
    setIsMouseDown(true);
    setLastMousePosition({
      x: e.clientX,
      y: e.clientY
    });
    
    const handleMouseMove = (moveEvent) => {
      if (!isMouseDown) return;
      
      const dx = moveEvent.clientX - lastMousePosition.x;
      const dy = moveEvent.clientY - lastMousePosition.y;
      
      setOffset(prev => ({
        x: prev.x + dx,
        y: prev.y + dy
      }));
      
      setLastMousePosition({
        x: moveEvent.clientX,
        y: moveEvent.clientY
      });
    };
    
    const handleMouseUp = () => {
      setIsMouseDown(false);
      
      document.removeEventListener('mousemove', handleMouseMove);
      document.removeEventListener('mouseup', handleMouseUp);
    };
    
    document.addEventListener('mousemove', handleMouseMove);
    document.addEventListener('mouseup', handleMouseUp);
  };
  
  // Add function to toggle panning mode
  const togglePanningMode = () => {
    setIsPanningMode(!isPanningMode);
  };
  
  // Add a function to highlight connected columns
  const getConnectedColumns = (columnId) => {
    if (!columnId || !graphData.column_lineage) {
      return { 
        connected: new Set(),
        sources: new Set(),
        targets: new Set()
      };
    }
    
    const connected = new Set();
    const sources = new Set();
    const targets = new Set();
    
    graphData.column_lineage.forEach(link => {
      if (link.source === columnId) {
        connected.add(link.target);
        targets.add(link.target);
      }
      if (link.target === columnId) {
        connected.add(link.source);
        sources.add(link.source);
      }
    });
    
    return { connected, sources, targets };
  };

  // Calculate the connected columns when activeColumnLink changes
  const connectedColumnsInfo = useMemo(() => {
    return getConnectedColumns(activeColumnLink);
  }, [activeColumnLink, graphData.column_lineage]);

  // Enhance the getColumnRelationshipInfo function to always return useful data
  const getColumnRelationshipInfo = (columnId) => {
    if (!columnId || !graphData.column_lineage) {
      return {
        relationships: [],
        hasRelationships: false,
        message: "No relationship data available"
      };
    }
    
    const relationships = [];
    const currentColumn = graphData.columns.find(c => c.id === columnId);
    const currentModel = currentColumn ? graphData.models.find(m => m.id === currentColumn.modelId) : null;
    
    // If no current column/model info, provide appropriate feedback
    if (!currentColumn || !currentModel) {
      return {
        relationships: [],
        hasRelationships: false,
        message: "Could not find column information"
      };
    }
    
    // Count incoming and outgoing relationships
    let incomingCount = 0;
    let outgoingCount = 0;
    
    graphData.column_lineage.forEach(link => {
      if (link.source === columnId || link.target === columnId) {
        const isSource = link.source === columnId;
        const otherColumnId = isSource ? link.target : link.source;
        const otherColumn = graphData.columns.find(c => c.id === otherColumnId);
        
        if (otherColumn) {
          const otherModel = graphData.models.find(m => m.id === otherColumn.modelId);
          if (otherModel) {
            // Count the relationship direction
            if (isSource) {
              outgoingCount++;
            } else {
              incomingCount++;
            }
            
            relationships.push({
              direction: isSource ? 'target' : 'source',
              columnName: otherColumn.name,
              modelName: otherModel.name,
              modelType: otherModel.type || "unknown",
              columnId: otherColumnId,
              dataType: otherColumn.dataType || 'unknown',
              description: otherColumn.description || `No description available for ${otherColumn.name}`
            });
          }
        }
      }
    });
    
    // Create a helpful message even when no relationships are found
    let message = null;
    if (relationships.length === 0) {
      // Check if the model has any relationships at all
      const modelHasAnyRelationships = graphData.column_lineage.some(link => {
        const sourceColumn = graphData.columns.find(c => c.id === link.source);
        const targetColumn = graphData.columns.find(c => c.id === link.target);
        return (sourceColumn && sourceColumn.modelId === currentModel.id) || 
               (targetColumn && targetColumn.modelId === currentModel.id);
      });
      
      if (modelHasAnyRelationships) {
        message = `No relationships found for this specific column (${currentColumn.name})`;
      } else {
        message = `No column relationships defined for this model (${currentModel.name})`;
      }
    } else {
      // Provide information about the relationship count and direction
      message = `Found ${relationships.length} relationship${relationships.length !== 1 ? 's' : ''}`;
      if (incomingCount > 0 && outgoingCount > 0) {
        message += ` (${incomingCount} incoming, ${outgoingCount} outgoing)`;
      } else if (incomingCount > 0) {
        message += ` (${incomingCount} incoming)`;
      } else if (outgoingCount > 0) {
        message += ` (${outgoingCount} outgoing)`;
      }
    }
    
    return {
      relationships,
      hasRelationships: relationships.length > 0,
      incomingCount,
      outgoingCount,
      currentColumn,
      currentModel,
      message
    };
  };
  
  // Calculate the layout of models based on the data
  useEffect(() => {
    console.log("Calculating layout for models:", graphData.models?.length);
    
    const calculateLevels = () => {
      // Skip if no models
      if (!graphData.models || graphData.models.length === 0) return {};
      
      // Create an adjacency list from the edges
      const adjList = {};
      graphData.models.forEach(model => {
        adjList[model.id] = { in: [], out: [] };
      });
      
      graphData.edges.forEach(edge => {
        if (adjList[edge.source]) adjList[edge.source].out.push(edge.target);
        if (adjList[edge.target]) adjList[edge.target].in.push(edge.source);
      });
      
      // Find nodes with no incoming edges (sources)
      const sources = graphData.models
        .filter(model => adjList[model.id].in.length === 0)
        .map(model => model.id);
      
      // If no sources, pick a random node
      if (sources.length === 0 && graphData.models.length > 0) {
        sources.push(graphData.models[0].id);
      }
      
      // BFS traversal to assign levels
      const levels = {};
      const visited = new Set();
      const queue = sources.map(id => ({ id, level: 0 }));
      
      while (queue.length > 0) {
        const { id, level } = queue.shift();
        
        if (visited.has(id)) continue;
        visited.add(id);
        
        // Set the level for this node
        if (!levels[level]) levels[level] = [];
        levels[level].push(id);
        
        // Add outgoing nodes to the queue
        adjList[id].out.forEach(targetId => {
          // Check that all dependencies are visited before adding to queue
          const canAddToQueue = adjList[targetId].in.every(depId => 
            visited.has(depId) || depId === id
          );
          
          if (canAddToQueue) {
            queue.push({ id: targetId, level: level + 1 });
          }
        });
      }
      
      // Handle any nodes not visited (disconnected)
      graphData.models.forEach(model => {
        if (!visited.has(model.id)) {
          const level = Object.keys(levels).length;
          if (!levels[level]) levels[level] = [];
          levels[level].push(model.id);
        }
      });
      
      return levels;
    };
    
    // Calculate model layout based on levels
    const calculateModelLayout = (levels) => {
      // Find the max number of nodes in any level
      let maxNodesInLevel = 0;
      Object.values(levels).forEach(nodes => {
        maxNodesInLevel = Math.max(maxNodesInLevel, nodes.length);
      });
      
      // Determine spacing based on number of models
      const totalModels = graphData.models.length;
      
      // Dynamic spacing based on model count - INCREASED spacing values
      let baseSpacing;
      if (totalModels > 12) {
        baseSpacing = 400; // Increased from 300 to 400
      } else if (totalModels > 8) {
        baseSpacing = 500; // Increased from 350 to 500
      } else if (totalModels > 4) {
        baseSpacing = 600; // Increased from 450 to 600
      } else {
        baseSpacing = 700; // Increased from 500 to 700
      }
      
      // Reduce horizontal spacing if we have many models per level, but increase minimum
      const horizontalSpacing = maxNodesInLevel > 4 ? Math.max(baseSpacing * 0.8, 350) : baseSpacing;
      
      // Calculate the layout for each model
      const newLayout = {};
      
      // Define fixed height for collapsed model boxes - make smaller
      const collapsedHeight = 60; 
      
      // Check if manual layout exists
      const hasManualPositions = Object.keys(layout).length > 0;
      
      Object.entries(levels).forEach(([level, nodeIds]) => {
        const levelNum = parseInt(level);
        
        nodeIds.forEach((nodeId, idx) => {
          const model = graphData.models.find(m => m.id === nodeId);
          if (!model) return;
          
          // If model already has position in layout and we're maintaining manual positions,
          // preserve its x/y coordinates
          if (hasManualPositions && layout[nodeId] && dragModelId !== null) {
            const existingLayout = layout[nodeId];
            
            // Count columns for this model to calculate expanded height
            const modelColumns = graphData.columns?.filter(col => col.modelId === nodeId) || [];
            const columnCount = modelColumns.length;
            
            // Calculate expanded height based on number of columns with more space
            const expandedHeight = collapsedHeight + Math.min(Math.max(columnCount * 40, 100), 400);
            
            newLayout[nodeId] = {
              x: existingLayout.x,
              y: existingLayout.y,
              width: 280,
              height: expandedModels[nodeId] ? expandedHeight : collapsedHeight
            };
          } else {
            // Calculate position based on layout mode
            let x, y;
            
            if (layoutMode === 'horizontal') {
              // In horizontal layout, levels go left to right, nodes are stacked vertically in each level
              x = levelNum * horizontalSpacing;
              // Add more vertical spacing for better separation
              y = idx * (baseSpacing + (maxNodesInLevel > 4 ? 100 : 200));
            } else {
              // In vertical layout, levels go top to bottom, nodes are arranged horizontally in each level
              // Add more horizontal spacing for better separation
              x = idx * (horizontalSpacing + (maxNodesInLevel > 4 ? 100 : 200));
              y = levelNum * baseSpacing;
            }
            
            // Count columns for this model to calculate expanded height
            const modelColumns = graphData.columns?.filter(col => col.modelId === nodeId) || [];
            const columnCount = modelColumns.length;
            
            // Calculate expanded height based on number of columns with more space
            const expandedHeight = collapsedHeight + Math.min(Math.max(columnCount * 40, 100), 400);
            
            newLayout[nodeId] = {
              x,
              y,
              width: 280,
              height: expandedModels[nodeId] ? expandedHeight : collapsedHeight
            };
          }
        });
      });
      
      // Check for overlaps and resolve them
      const resolveOverlaps = () => {
        let hasOverlap = true;
        const padding = 20; // Minimum space between boxes
        const iterations = 0;
        const maxIterations = 5;
        
        while (hasOverlap && iterations < maxIterations) {
          hasOverlap = false;
          
          // Compare each pair of models
          const modelIds = Object.keys(newLayout);
          for (let i = 0; i < modelIds.length; i++) {
            const modelA = newLayout[modelIds[i]];
            
            for (let j = i + 1; j < modelIds.length; j++) {
              const modelB = newLayout[modelIds[j]];
              
              // Check if they overlap
              const overlapX = 
                modelA.x < modelB.x + modelB.width + padding && 
                modelA.x + modelA.width + padding > modelB.x;
                
              const overlapY = 
                modelA.y < modelB.y + modelB.height + padding && 
                modelA.y + modelA.height + padding > modelB.y;
                
              if (overlapX && overlapY) {
                hasOverlap = true;
                
                // Determine the smallest movement needed to resolve overlap
                const moveRight = modelB.x + modelB.width + padding - modelA.x;
                const moveLeft = modelA.x + modelA.width + padding - modelB.x;
                const moveDown = modelB.y + modelB.height + padding - modelA.y;
                const moveUp = modelA.y + modelA.height + padding - modelB.y;
                
                // Find the smallest required move
                const minMove = Math.min(moveRight, moveLeft, moveDown, moveUp);
                
                // Apply the movement to model B
                if (minMove === moveRight) {
                  newLayout[modelIds[i]].x += moveRight;
                } else if (minMove === moveLeft) {
                  newLayout[modelIds[j]].x += moveLeft;
                } else if (minMove === moveDown) {
                  newLayout[modelIds[i]].y += moveDown;
                } else if (minMove === moveUp) {
                  newLayout[modelIds[j]].y += moveUp;
                }
              }
            }
          }
        }
      };
      
      // Run the overlap resolution if we have generated a new layout
      if (!hasManualPositions || dragModelId === null) {
        resolveOverlaps();
      }
      
      return newLayout;
    };
    
    // Calculate and set the layout
    try {
      const levels = calculateLevels();
      console.log("Calculated levels:", levels);
      const newLayout = calculateModelLayout(levels);
      console.log("New layout:", newLayout);
      setLayout(newLayout);
    } catch (err) {
      console.error("Error calculating layout:", err);
      setHasError(true);
      setErrorMessage(`Layout error: ${err.message}`);
    }
  }, [graphData.models, graphData.edges, graphData.columns, layoutMode, expandedModels, layout, dragModelId]);
  
  // Add a function to manually initialize layout with fixed positions
  const initializeManualLayout = () => {
    console.log("Manually initializing layout");
    const manualLayout = {};
    const startX = 100;
    const startY = 100;
    const spacing = 300;
    
    // Create a simple horizontal layout
    graphData.models.forEach((model, index) => {
      manualLayout[model.id] = {
        x: startX + (index * spacing),
        y: startY,
        width: 250,
        height: 70 + (expandedModels[model.id] ? 65 : 0)
      };
    });
    
    console.log("Manual layout created:", manualLayout);
    setLayout(manualLayout);
    
    // Fit to view after a short delay
    setTimeout(fitToView, 300);
  };
  
  // Add state for table navigator drawer
  const { isOpen, onOpen, onClose } = useDisclosure();
  const [searchTerm, setSearchTerm] = useState('');
  
  // Group models by type for better organization in the navigator
  const modelsByType = useMemo(() => {
    const result = {};
    
    if (!graphData.models) return result;
    
    graphData.models.forEach(model => {
      const type = model.type || 'other';
      if (!result[type]) {
        result[type] = [];
      }
      result[type].push(model);
    });
    
    return result;
  }, [graphData.models]);
  
  // Filter models based on search term
  const filteredModels = useMemo(() => {
    if (!searchTerm.trim()) return modelsByType;
    
    const result = {};
    
    Object.entries(modelsByType).forEach(([type, models]) => {
      const filtered = models.filter(model => 
        model.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
        model.path?.toLowerCase().includes(searchTerm.toLowerCase())
      );
      
      if (filtered.length > 0) {
        result[type] = filtered;
      }
    });
    
    return result;
  }, [modelsByType, searchTerm]);
  
  // Function to navigate to a specific model
  const navigateToModel = (modelId) => {
    // Expand the model if it's collapsed
    if (!expandedModels[modelId]) {
      setExpandedModels(prev => ({
        ...prev,
        [modelId]: true
      }));
    }
    
    // Focus on the model
    focusOnModel(modelId);
    
    // Close the drawer
    onClose();
  };
  
  // Add fullscreen state with modal
  const [isFullscreen, setIsFullscreen] = useState(false);
  // Save previous state to restore when exiting fullscreen
  const [previousState, setPreviousState] = useState(null);
  
  // Toggle fullscreen mode using modal
  const toggleFullscreen = () => {
    if (!isFullscreen) {
      // Save current state before going fullscreen
      setPreviousState({
        scale,
        offset: { ...offset },
        expandedModels: { ...expandedModels }
      });
      
      // When entering fullscreen, automatically fit to view after a short delay
      setIsFullscreen(true);
      setTimeout(() => {
        fitToView();
      }, 300);
    } else {
      // Restore previous state when exiting fullscreen
      setIsFullscreen(false);
      if (previousState) {
        setTimeout(() => {
          setScale(previousState.scale);
          setOffset(previousState.offset);
        }, 100);
      }
    }
  };
  
  // Function to handle horizontal scroll
  const handleHorizontalScroll = (val) => {
    // Calculate the delta from the previous value
    const delta = val - horizontalSliderValue;
    // Update the offset based on the delta
    setOffset(prev => ({ 
      x: prev.x + delta, 
      y: prev.y 
    }));
    // Store the new value
    setHorizontalSliderValue(val);
  };
  
  // Function to handle vertical scroll
  const handleVerticalScroll = (val) => {
    // Calculate the delta from the previous value
    const delta = val - verticalSliderValue;
    // Update the offset based on the delta
    setOffset(prev => ({ 
      x: prev.x, 
      y: prev.y + delta 
    }));
    // Store the new value
    setVerticalSliderValue(val);
  };
  
  // Function to reset sliders after they're released
  const resetSliders = () => {
    setHorizontalSliderValue(0);
    setVerticalSliderValue(0);
  };
  
  // Render the LineageGraph content
  const renderLineageContent = () => {
    return (
      <Box 
        ref={containerRef}
        position="relative"
        width="100%"
        height={isFullscreen ? "calc(100vh - 120px)" : "400px"}
        borderWidth={isFullscreen ? "0" : "1px"}
        borderRadius={isFullscreen ? "0" : "md"}
        borderColor="gray.200"
        overflow="hidden"
        boxShadow={isFullscreen ? "none" : "sm"}
        bg="white"
        onMouseDown={isPanningMode ? handleGraphPanning : undefined}
        style={{
          cursor: isPanningMode 
            ? (isMouseDown ? "grabbing" : "grab") 
            : "default",
          transition: "height 0.3s ease"
        }}
      >
        {/* Table navigation drawer */}
        <Drawer isOpen={isOpen} placement="left" onClose={onClose} size="xs">
          <DrawerOverlay />
          <DrawerContent>
            <DrawerCloseButton />
            <DrawerHeader borderBottomWidth="1px">
              Navigate to Table
            </DrawerHeader>

            <DrawerBody>
              <InputGroup mb={4} mt={2}>
                <InputLeftElement pointerEvents="none">
                  <Icon as={IoSearch} color="gray.400" />
                </InputLeftElement>
                <Input 
                  placeholder="Search tables..." 
                  value={searchTerm}
                  onChange={(e) => setSearchTerm(e.target.value)}
                  variant="filled"
                />
              </InputGroup>
              
              {Object.keys(filteredModels).length === 0 ? (
                <Text color="gray.500" textAlign="center" mt={8}>
                  No matching tables found
                </Text>
              ) : (
                <VStack align="stretch" spacing={2}>
                  {Object.entries(filteredModels).map(([type, models]) => (
                    <Box key={type}>
                      <Text 
                        fontWeight="bold" 
                        fontSize="sm" 
                        color="gray.600" 
                        mb={1}
                        textTransform="uppercase"
                      >
                        {type}
                      </Text>
                      <List spacing={1}>
                        {models.map(model => (
                          <ListItem 
                            key={model.id}
                            p={2}
                            borderRadius="md"
                            _hover={{ bg: "gray.100" }}
                            cursor="pointer"
                            onClick={() => navigateToModel(model.id)}
                          >
                            <HStack>
                              <Box 
                                w={2} 
                                h={2} 
                                borderRadius="full" 
                                bg={getModelTypeColor(model.type)}
                              />
                              <Text fontSize="sm">{model.name}</Text>
                            </HStack>
                          </ListItem>
                        ))}
                      </List>
                    </Box>
                  ))}
                </VStack>
              )}
            </DrawerBody>
          </DrawerContent>
        </Drawer>
        
        {/* Controls container - Top Left */}
        <Box 
          position="absolute"
          top={4}
          left={4}
          zIndex={10}
          display="flex"
          flexDirection="column"
          alignItems="flex-start"
        >
          <Button
            leftIcon={<Icon as={IoList} />}
            colorScheme="blue"
            variant="solid"
            size="sm"
            onClick={onOpen}
            boxShadow="md"
          >
            Tables
          </Button>
          
          {/* Updated Controls Container - Moved under Tables button */}
          <Box 
            mt={3}
            bg="white"
            p={2}
            borderRadius="md"
            boxShadow="md"
            display="flex"
            flexDirection="column"
            gap={2}
            width="auto"
          >
            {/* Compact Controls Row 1 - Main actions */}
            <HStack spacing={1}>
              <Tooltip label="Auto Fit" placement="bottom">
                <IconButton
                  size="sm"
                  icon={<Icon as={TbArrowsMaximize} />}
                  onClick={fitToView}
                  aria-label="Auto Fit"
                  colorScheme="blue"
                />
              </Tooltip>
              
              <Tooltip label={isFullscreen ? "Exit Fullscreen" : "Fullscreen"} placement="bottom">
                <IconButton
                  size="sm"
                  icon={<Icon as={isFullscreen ? TbMinimize : TbMaximize} />}
                  onClick={toggleFullscreen}
                  aria-label="Toggle Fullscreen"
                  colorScheme="purple"
                />
              </Tooltip>
              
              <Tooltip label="Toggle Layout Direction" placement="bottom">
                <IconButton
                  size="sm"
                  icon={<Icon as={layoutMode === 'horizontal' ? TbArrowsHorizontal : TbArrowsVertical} />}
                  onClick={toggleLayoutMode}
                  aria-label="Toggle Layout"
                  variant="outline"
                />
              </Tooltip>
              
              <Tooltip label="Reset View" placement="bottom">
                <IconButton
                  size="sm"
                  icon={<Icon as={TbArrowBack} />}
                  onClick={resetLayout}
                  aria-label="Reset View"
                  variant="outline"
                />
              </Tooltip>
            </HStack>
            
            {/* Compact Controls Row 2 - Zoom and Navigation */}
            <HStack spacing={1}>
              <Tooltip label="Zoom Out" placement="bottom">
                <IconButton
                  size="sm"
                  icon={<Icon as={TbZoomOut} />}
                  onClick={() => setScale(prev => Math.max(prev / 1.2, 0.5))}
                  aria-label="Zoom Out"
                  variant="outline"
                />
              </Tooltip>
              
              <Text fontSize="xs" fontWeight="medium" width="40px" textAlign="center">
                {Math.round(scale * 100)}%
              </Text>
              
              <Tooltip label="Zoom In" placement="bottom">
                <IconButton
                  size="sm"
                  icon={<Icon as={TbZoomIn} />}
                  onClick={() => setScale(prev => Math.min(prev * 1.2, 2))}
                  aria-label="Zoom In"
                  variant="outline"
                />
              </Tooltip>
              
              <Tooltip label={isPanningMode ? "Exit Pan Mode" : "Pan Mode"} placement="bottom">
                <IconButton
                  size="sm"
                  icon={<Icon as={TbMapPin} />}
                  onClick={togglePanningMode}
                  aria-label="Pan Mode"
                  colorScheme={isPanningMode ? "purple" : "gray"}
                  variant={isPanningMode ? "solid" : "outline"}
                />
              </Tooltip>
            </HStack>
            
            {/* Scrolling Controls - Replacing the arrow buttons */}
            <Box px={1} pt={2} width="100%">
              {/* Horizontal scroll label */}
              <Flex justify="space-between" mb={1}>
                <Text fontSize="xs" fontWeight="medium" color="gray.600">Horizontal Scroll</Text>
                <Tooltip label="Center View" placement="top">
                  <IconButton
                    size="xs"
                    icon={<Icon as={TbArrowsMaximize} />}
                    onClick={fitToView}
                    aria-label="Center View"
                    variant="ghost"
                  />
                </Tooltip>
              </Flex>

              {/* Horizontal scroll slider */}
              <Slider 
                min={-100} 
                max={100} 
                step={5}
                value={horizontalSliderValue}
                onChange={handleHorizontalScroll}
                onChangeEnd={resetSliders}
                mb={3}
                colorScheme="blue"
              >
                <SliderTrack bg="gray.100">
                  <SliderFilledTrack />
                </SliderTrack>
                <SliderThumb boxSize={4}>
                  <Box color="blue.500" as={IoResize} transform="rotate(90deg)" />
                </SliderThumb>
              </Slider>

              {/* Vertical scroll label */}
              <Text fontSize="xs" fontWeight="medium" color="gray.600" mb={1}>Vertical Scroll</Text>
              
              {/* Vertical scroll slider */}
              <Slider 
                min={-100} 
                max={100} 
                step={5}
                value={verticalSliderValue}
                onChange={handleVerticalScroll}
                onChangeEnd={resetSliders}
                colorScheme="blue"
              >
                <SliderTrack bg="gray.100">
                  <SliderFilledTrack />
                </SliderTrack>
                <SliderThumb boxSize={4}>
                  <Box color="blue.500" as={IoResize} />
                </SliderThumb>
              </Slider>
            </Box>
          </Box>
        </Box>

        {/* Error message if needed */}
        {hasError && (
          <Box 
            position="absolute" 
            top="50%" 
            left="50%" 
            transform="translate(-50%, -50%)"
            textAlign="center"
            p={4}
            bg="red.50"
            borderRadius="md"
            borderWidth="1px"
            borderColor="red.200"
            maxWidth="80%"
          >
            <Heading size="md" color="red.500" mb={2}>Error</Heading>
            <Text>{errorMessage || "There was an error rendering the lineage graph."}</Text>
          </Box>
        )}
        
        {/* Loading state */}
        {!hasError && Object.keys(layout).length === 0 && (
          <Box 
            position="absolute" 
            top="50%" 
            left="50%" 
            transform="translate(-50%, -50%)"
            textAlign="center"
            p={4}
          >
            <Heading size="md" mb={2}>Initializing Graph</Heading>
            <Text color="gray.600" mb={4}>Calculating layout for data models...</Text>
            <Text fontSize="sm" color="gray.500" mb={4}>
              Models: {graphData.models?.length || 0}, 
              Edges: {graphData.edges?.length || 0}
            </Text>
            
            <Button 
              onClick={initializeManualLayout}
              colorScheme="blue"
              size="sm"
            >
              Initialize Manually
            </Button>
          </Box>
        )}
        
        {/* Go back to the previous rendering method for consistent coordinates */}
        <Box 
          position="absolute"
          top={0}
          left={0}
          width="100%"
          height="100%"
          style={{
            transform: `translate(${offset.x}px, ${offset.y}px) scale(${scale})`,
            transformOrigin: '0 0',
            transition: isDragging ? 'none' : 'transform 0.2s ease'
          }}
        >
          {/* SVG Container for edges and column connections */}
          <svg 
            width="100%" 
            height="100%" 
            style={{ 
              position: 'absolute',
              top: 0,
              left: 0,
              pointerEvents: 'none',
              overflow: 'visible'
            }}
          >
            <g>{renderEdges()}</g>
            <g>{renderColumnConnections()}</g>
          </svg>
          
          {/* Models Container - same coordinate system as arrows now */}
          <Box position="relative" top={0} left={0} style={{ pointerEvents: 'auto' }}>
            {graphData.models.map(model => {
              const modelLayout = layout[model.id];
              
              if (!modelLayout) return null;
              
              const isExpanded = expandedModels[model.id];
              const modelTypeColor = getModelTypeColor(model.type);
              
              // Get columns for this model
              const modelColumns = graphData.columns?.filter(col => col.modelId === model.id) || [];
              
              return (
                <Box
                  key={model.id}
                  position="absolute"
                  top={modelLayout.y}
                  left={modelLayout.x}
                  width={modelLayout.width}
                  height={modelLayout.height}
                  bg="white"
                  borderWidth="1px"
                  borderRadius="md"
                  borderColor={model.highlight ? "orange.400" : "gray.200"}
                  boxShadow={model.highlight ? "0 0 0 2px rgba(237, 137, 54, 0.4)" : "md"}
                  overflow="hidden"
                  transition="all 0.3s ease" 
                  _hover={{ 
                    boxShadow: "lg",
                    borderColor: "gray.300"
                  }}
                  onMouseEnter={() => setActiveModelId(model.id)}
                  onMouseLeave={() => setActiveModelId(null)}
                  style={{
                    cursor: 'move'
                  }}
                  onMouseDown={(e) => handleDragStart(model.id, e)}
                >
                  {/* Model Header - Clickable to expand/collapse */}
                  <Box 
                    p={isExpanded ? 3 : 2}
                    borderBottomWidth={isExpanded ? "1px" : "0"}
                    borderColor="gray.100"
                    bg="white"
                    onClick={(e) => {
                      e.stopPropagation();
                      toggleModelExpansion(model.id);
                    }}
                    cursor="pointer"
                  >
                    <HStack justifyContent="space-between" mb={1}>
                      <HStack>
                        <Box 
                          w={2} 
                          h={2} 
                          borderRadius="full" 
                          bg={modelTypeColor} 
                          mr={1}
                        />
                        {getModelTypeBadge(model.type)}
                      </HStack>
                      
                      <Icon 
                        as={isExpanded ? IoChevronUp : IoChevronDown} 
                        color="gray.400"
                        _hover={{ color: "gray.600" }}
                      />
                    </HStack>
                    
                    <Text 
                      fontWeight="semibold" 
                      fontSize={isExpanded ? "md" : "sm"}
                      color="gray.800"
                      noOfLines={1}
                    >
                      {model.name}
                    </Text>
                    
                    <Text 
                      fontSize="xs" 
                      color="gray.500" 
                      mt={0.5}
                      noOfLines={1}
                      display={isExpanded ? "block" : "none"}
                    >
                      {model.path}
                    </Text>
                  </Box>
                  
                  {/* Model Columns (if expanded) */}
                  {isExpanded && modelColumns.length > 0 && (
                    <VStack 
                      spacing={1} 
                      p={2} 
                      align="stretch" 
                      maxH="350px"
                      overflowY="auto"
                      bg="gray.50"
                      css={{
                        '&::-webkit-scrollbar': {
                          width: '8px',
                        },
                        '&::-webkit-scrollbar-track': {
                          width: '10px',
                          background: 'rgba(0, 0, 0, 0.05)',
                          borderRadius: '4px',
                        },
                        '&::-webkit-scrollbar-thumb': {
                          background: 'rgba(0, 0, 0, 0.1)',
                          borderRadius: '4px',
                          '&:hover': {
                            background: 'rgba(0, 0, 0, 0.2)',
                          },
                        },
                      }}
                      position="relative"
                    >
                      {/* Scroll indicator for many columns */}
                      {modelColumns.length > 8 && (
                        <Box 
                          position="absolute" 
                          bottom="5px" 
                          right="5px" 
                          zIndex={2}
                          bg="white" 
                          borderRadius="full" 
                          boxShadow="md"
                          p={1}
                          opacity={0.8}
                        >
                          <Tooltip label={`${modelColumns.length} columns total`} placement="top">
                            <Text fontSize="xs" fontWeight="bold" color="gray.500">
                              {modelColumns.length}
                            </Text>
                          </Tooltip>
                        </Box>
                      )}
                      
                      {modelColumns.map(column => {
                        const isActive = activeColumnLink === column.id;
                        const isConnected = connectedColumnsInfo.connected && connectedColumnsInfo.connected.has(column.id);
                        const isSource = connectedColumnsInfo.sources && connectedColumnsInfo.sources.has(column.id);
                        const isTarget = connectedColumnsInfo.targets && connectedColumnsInfo.targets.has(column.id);
                        
                        let columnHighlightStatus = "default";
                        if (isActive) columnHighlightStatus = "active";
                        else if (isConnected) columnHighlightStatus = "connected";
                        else if (activeColumnLink) columnHighlightStatus = "dimmed";
                        
                        const columnColor = getColumnTypeColor(column.type, isActive);
                        const { icon, label } = getColumnTypeInfo(column.type);
                        
                        // Get relationships for tooltip - use enhanced function
                        const relationshipInfo = getColumnRelationshipInfo(column.id);

                        return (
                          <Tooltip
                            key={column.id}
                            label={
                              <Box p={2} minWidth="200px" maxWidth="300px">
                                {column.description && (
                                  <Box mb={2}>
                                    <Text fontSize="xs" fontWeight="semibold" color="gray.600">
                                      Description:
                                    </Text>
                                    <Text fontSize="xs" color="gray.700">
                                      {column.description}
                                    </Text>
                                  </Box>
                                )}
                                
                                <Box mb={2}>
                                  <Text fontSize="xs" fontWeight="semibold" color="gray.600">
                                    Type: 
                                  </Text>
                                  <HStack mt={1} spacing={2}>
                                    <Tag size="sm" colorScheme="gray" variant="subtle">
                                      <TagLabel fontSize="2xs">{column.dataType || "unknown"}</TagLabel>
                                    </Tag>
                                    <Tag size="sm" colorScheme="blue" variant="subtle">
                                      <TagLabel fontSize="2xs">{label}</TagLabel>
                                    </Tag>
                                  </HStack>
                                </Box>
                                
                                <Divider my={1} />
                                
                                <Box>
                                  <Text fontSize="xs" fontWeight="semibold" color="gray.600">
                                    Relationships:
                                  </Text>
                                  {relationshipInfo.relationships && relationshipInfo.relationships.length > 0 ? (
                                    <VStack align="start" spacing={1} mt={1}>
                                      {relationshipInfo.relationships.map((rel, idx) => (
                                        <HStack key={idx} spacing={1}>
                                          <Icon 
                                            as={rel.direction === 'source' ? FiArrowLeft : FiArrowRight} 
                                            color={rel.direction === 'source' ? "green.500" : "orange.500"}
                                            boxSize="12px"
                                          />
                                          <Text fontSize="xs">{rel.columnName}</Text>
                                          <Text fontSize="xs" color="gray.500">({rel.modelName})</Text>
                                        </HStack>
                                      ))}
                                    </VStack>
                                  ) : (
                                    <Text fontSize="xs" color="gray.500" mt={1}>
                                      No relationships found for this column
                                    </Text>
                                  )}
                                </Box>
                              </Box>
                            }
                            placement="right"
                            openDelay={300}
                            bg="white"
                            color="gray.800"
                            boxShadow="md"
                            borderRadius="md"
                            p={2}
                            hasArrow
                            gutter={12}
                          >
                            <HStack
                              p={1}
                              borderRadius="sm"
                              bg={
                                columnHighlightStatus === "active" ? "blue.50" :
                                columnHighlightStatus === "connected" ? "gray.50" :
                                "transparent"
                              }
                              cursor="pointer"
                              onClick={(e) => {
                                e.stopPropagation();
                                setActiveColumnLink(activeColumnLink === column.id ? null : column.id);
                              }}
                              _hover={{ bg: isActive ? "blue.50" : "gray.50" }}
                              opacity={
                                columnHighlightStatus === "dimmed" ? 0.5 : 1
                              }
                              transition="all 0.2s"
                              position="relative"
                              borderLeftWidth="2px"
                              borderLeftColor={
                                isActive ? "blue.500" :
                                isSource ? "green.500" :
                                isTarget ? "orange.500" :
                                "transparent"
                              }
                            >
                              <Box 
                                w={2} 
                                h={2} 
                                borderRadius="full" 
                                bg={columnColor}
                              />
                              <Text 
                                fontSize="xs" 
                                fontWeight={isActive || isConnected ? "semibold" : "medium"}
                                color="gray.800"
                                flex="1"
                                noOfLines={1}
                              >
                                {column.name}
                              </Text>
                              
                              {label && (
                                <Tag size="sm" borderRadius="full" colorScheme={column.type === 'primary_key' ? 'purple' : column.type === 'foreign_key' ? 'blue' : 'gray'} variant="subtle">
                                  <TagLabel fontSize="2xs" fontWeight="bold">{label}</TagLabel>
                                </Tag>
                              )}
                              
                              <Tag size="sm" borderRadius="full" colorScheme="gray" variant="subtle">
                                <TagLabel fontSize="2xs">{column.dataType}</TagLabel>
                              </Tag>
                              
                              {relationshipInfo.hasRelationships && (
                                <Box 
                                  position="absolute"
                                  right="-2px"
                                  top="-2px"
                                  w={2}
                                  h={2}
                                  borderRadius="full"
                                  bg={isActive || isConnected ? "orange.400" : "blue.400"}
                                  animation={isActive || isConnected ? "pulse 1.5s infinite" : "none"}
                                />
                              )}
                            </HStack>
                          </Tooltip>
                        );
                      })}
                    </VStack>
                  )}
                  
                  {/* Show a placeholder message if expanded but no columns */}
                  {isExpanded && modelColumns.length === 0 && (
                    <Box 
                      p={3}
                      bg="gray.50"
                      textAlign="center"
                    >
                      <Text fontSize="xs" color="gray.500">
                        No columns available
                      </Text>
                    </Box>
                  )}
                </Box>
              );
            })}
          </Box>
        </Box>
      </Box>
    );
  };

  // Now return either the modal or inline component based on fullscreen state
  if (isFullscreen) {
    return (
      <Modal 
        isOpen={isFullscreen} 
        onClose={toggleFullscreen} 
        size="full"
        motionPreset="slideInBottom"
      >
        <ModalOverlay />
        <ModalContent>
          <ModalHeader>
            <Flex justifyContent="space-between" alignItems="center">
              <Text>Data Lineage Visualization</Text>
              <IconButton
                icon={<Icon as={TbMinimize} />}
                onClick={toggleFullscreen}
                aria-label="Exit Fullscreen"
                colorScheme="purple"
                size="sm"
              />
            </Flex>
          </ModalHeader>
          <ModalBody p={0}>
            {renderLineageContent()}
          </ModalBody>
        </ModalContent>
      </Modal>
    );
  }
  
  // Normal inline rendering
  return renderLineageContent();
};

