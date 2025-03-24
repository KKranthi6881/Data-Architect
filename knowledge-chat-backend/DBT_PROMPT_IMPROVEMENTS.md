# DBT Data Architect Agent Prompt Improvements

## Overview

This document outlines the enhancements made to the Data Architect Agent prompts to better leverage the DeepSeek reasoning model's capabilities. By incorporating explicit thinking steps and structured analytical paths, we've improved the agent's ability to reason through complex DBT-related tasks like a professional data engineer or architect would.

## Key Improvements

### 1. Structured Thinking Steps

Each prompt now includes a `<thinking>` section that guides the model through a step-by-step analytical process before generating its response. This approach:

- Ensures thorough analysis of the problem space
- Encourages methodical exploration of relevant DBT project components
- Mirrors how experienced data professionals approach problems
- Prevents overlooking critical aspects of DBT architecture and design

### 2. Task-Specific Reasoning Paths

We've customized the reasoning process for different types of DBT tasks:

#### Model Explanation
- Step 1: Locate model files (SQL and YAML)
- Step 2: Analyze model definition and structure
- Step 3: Review schema definitions and tests
- Step 4: Map dependencies (upstream and downstream)
- Step 5: Analyze SQL transformations and business logic
- Step 6: Synthesize business context and purpose

#### Development Tasks
- Step 1: Understand requirements and business needs
- Step 2: Identify related models and sources
- Step 3: Determine appropriate model structure
- Step 4: Develop SQL logic with best practices
- Step 5: Define schema and tests
- Step 6: Ensure data quality and performance

#### Code Enhancement
- Step 1: Understand current implementation
- Step 2: Diagnose the enhancement need
- Step 3: Plan the modifications
- Step 4: Design solution approach based on enhancement type
- Step 5: Validate the approach against best practices
- Step 6: Plan implementation steps and testing

#### Lineage Analysis
- Step 1: Identify the model of interest
- Step 2: Map direct upstream dependencies
- Step 3: Analyze nested dependencies
- Step 4: Trace downstream dependencies
- Step 5: Analyze exposures and final outputs
- Step 6: Visualize the complete lineage

#### Documentation
- Step 1: Understand documentation needs and audience
- Step 2: Examine existing documentation and code
- Step 3: Identify documentation gaps
- Step 4: Plan documentation structure
- Step 5: Create clear, concise documentation content
- Step 6: Ensure completeness and accuracy

### 3. DBT-Specific Considerations

The enhanced prompts incorporate DBT-specific knowledge and best practices:

- **Project Structure Awareness**: Differentiates between staging, intermediate, and mart models
- **Materialization Strategies**: Guides thinking about table, view, incremental, or ephemeral approaches
- **Dependencies Management**: Emphasizes proper handling of ref() and source() functions
- **Testing Practices**: Encourages appropriate schema tests and custom tests
- **Documentation Standards**: Follows DBT documentation best practices

### 4. Output Format Improvements

While the thinking steps are elaborate, the final output format remains:

- Concise and focused on actionable information
- Well-structured with clear headings and sections
- Rich with practical code examples
- Path-specific with exact file locations
- Balanced between business context and technical details

## Implementation Details

The improved prompts are implemented in the following methods:

- `_get_model_explanation_instructions()`
- `_get_development_instructions()`
- `_get_code_enhancement_instructions()`
- `_get_lineage_instructions()`
- `_get_dependency_instructions()`
- `_get_documentation_instructions()`
- `_get_general_instructions()`

Each method now includes a `<thinking>` section that the DeepSeek reasoning model processes internally before generating its response.

## Benefits

1. **More Complete Analysis**: The step-by-step thinking process ensures comprehensive coverage of all relevant aspects
2. **Consistent Methodology**: Applies a standardized approach to different DBT tasks
3. **Higher Quality Outputs**: Final responses are better informed by the thorough analysis
4. **Human-Like Reasoning**: Mimics how experienced DBT professionals approach problems
5. **Reduced Errors**: Less likely to miss critical dependencies or schema considerations

## Example Transformation

Before:
```
Provide a focused, detailed response to this general dbt question...
```

After:
```
<thinking>
You are a data architect answering a general question about DBT. Follow these structured thinking steps:

STEP 1: UNDERSTAND THE QUESTION TYPE
- Is this a conceptual question about DBT?
...

STEP 6: VALIDATE YOUR ANSWER
- Ensure the answer is technically accurate
...
</thinking>

Provide a focused, detailed response to this general dbt question...
```

## Future Enhancements

Future iterations could include:

1. Adding specific thinking steps for more specialized DBT tasks (incremental models, testing strategies)
2. Incorporating more Snowflake-specific optimization considerations
3. Expanding the thinking process to include data governance and security considerations
4. Adding version-specific DBT feature considerations
5. Including project scaling and performance optimization thinking patterns 