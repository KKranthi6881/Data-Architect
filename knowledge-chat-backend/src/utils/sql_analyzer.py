import re
import logging
from typing import Dict, List, Any, Tuple, Optional

logger = logging.getLogger(__name__)

def analyze_sql_model(sql_content: str, source_column_name: str = None) -> Dict[str, Any]:
    """
    Analyze a SQL model's structure, identifying CTEs, columns, joins, and other components.
    
    Args:
        sql_content (str): The SQL content to analyze
        source_column_name (str, optional): The name of a specific column to track
        
    Returns:
        Dict[str, Any]: A detailed analysis of the SQL structure
    """
    analysis = {
        "isDBTModel": _is_dbt_model(sql_content),
        "hasCTEs": _has_ctes(sql_content),
        "ctes": [],
        "finalCTE": None,
        "finalSelect": None,
        "columns": [],
        "joins": [],
        "groupBys": [],
        "hasSourceColumn": source_column_name is not None and source_column_name in sql_content,
        "sourceColumnLocations": _find_column_references(sql_content, source_column_name) if source_column_name else []
    }
    
    # Extract all CTEs
    analysis["ctes"] = _extract_ctes(sql_content)
    
    # Find the final SELECT or final CTE
    final_cte_match = re.search(r'final\s+as\s+\(\s*\n([\s\S]*?)(?=\)\s*\n+select|\)$)', sql_content)
    final_select_match = re.search(r'select\s+[\s\S]*?from[\s\S]*?$', sql_content)
    
    if final_cte_match:
        analysis["finalCTE"] = {
            "content": final_cte_match.group(1),
            "columns": _extract_columns(final_cte_match.group(1))
        }
    
    if final_select_match:
        analysis["finalSelect"] = {
            "content": final_select_match.group(0),
            "columns": _extract_columns(final_select_match.group(0))
        }
    
    # Extract joins
    join_regex = r'(inner|left|right|full|cross)?\s*join\s+(\w+)\s+(?:as\s+)?(\w+)?\s+on\s+(.*?)(?=\s+(?:inner|left|right|full|cross)?\s*join|\s+where|\s+group\s+by|\s+order\s+by|\s*$)'
    analysis["joins"] = []
    for match in re.finditer(join_regex, sql_content, re.IGNORECASE | re.DOTALL):
        analysis["joins"].append({
            "type": match.group(1) or "inner",
            "table": match.group(2),
            "alias": match.group(3) or match.group(2),
            "condition": match.group(4)
        })
    
    # Extract GROUP BY clauses
    group_by_match = re.search(r'group\s+by\s+(.*?)(?=having|order\s+by|limit|$)', sql_content, re.IGNORECASE)
    if group_by_match:
        analysis["groupBys"] = [col.strip() for col in group_by_match.group(1).split(',')]
    
    # If source column is specified, check specifically where it appears
    if source_column_name:
        for i, cte in enumerate(analysis["ctes"]):
            if source_column_name in cte["content"]:
                cte["hasSourceColumn"] = True
                
                # Check if it's used in an aggregation in this CTE
                agg_pattern = rf'(sum|avg|count|min|max)\s*\(\s*{re.escape(source_column_name)}\s*\)'
                cte["sourceColumnInAggregation"] = bool(re.search(agg_pattern, cte["content"], re.IGNORECASE))
            else:
                cte["hasSourceColumn"] = False
                cte["sourceColumnInAggregation"] = False
    
    return analysis

def _is_dbt_model(sql_content: str) -> bool:
    """Check if the SQL content is a dbt model."""
    return '{{ ref(' in sql_content or '{{ref(' in sql_content

def _has_ctes(sql_content: str) -> bool:
    """Check if the SQL content has CTEs."""
    return 'with ' in sql_content and ' as (' in sql_content

def _extract_ctes(sql_content: str) -> List[Dict[str, Any]]:
    """Extract all CTEs from the SQL content."""
    ctes = []
    cte_regex = r'(\w+)\s+as\s+\(\s*\n([\s\S]*?)(?=\),\s*\n\w+\s+as\s+\(|\),\s*\nfinal\s+as\s+\(|\)\s*\n+select|\)$)'
    
    for match in re.finditer(cte_regex, sql_content, re.DOTALL):
        cte_name = match.group(1)
        cte_content = match.group(2)
        
        # Analyze the CTE content
        cte_type = _determine_cte_type(cte_content)
        cte_columns = _extract_columns(cte_content)
        
        ctes.append({
            "name": cte_name,
            "content": cte_content,
            "type": cte_type,
            "columns": cte_columns,
            "isAggregation": cte_type == 'aggregation'
        })
    
    return ctes

def _determine_cte_type(cte_content: str) -> str:
    """Determine the type of a CTE based on its content."""
    if re.search(r'(sum|avg|count|min|max)\s*\(', cte_content, re.IGNORECASE):
        return 'aggregation' if 'group by' in cte_content.lower() else 'calculation'
    elif 'join' in cte_content.lower():
        return 'join'
    elif 'where' in cte_content.lower():
        return 'filter'
    else:
        return 'base'

def _extract_columns(sql_segment: str) -> List[Dict[str, Any]]:
    """Extract columns from a SQL segment."""
    columns = []
    select_match = re.search(r'select\s+([\s\S]*?)(?=from)', sql_segment, re.IGNORECASE)
    
    if select_match:
        select_clause = select_match.group(1)
        # Split by commas, but handle complex expressions
        depth = 0
        current_column = ''
        
        for char in select_clause:
            if char == '(':
                depth += 1
            elif char == ')':
                depth -= 1
            
            if char == ',' and depth == 0:
                columns.append(current_column.strip())
                current_column = ''
            else:
                current_column += char
        
        if current_column.strip():
            columns.append(current_column.strip())
    
    return [_parse_column(col) for col in columns]

def _parse_column(column_definition: str) -> Dict[str, Any]:
    """Parse a column definition into a structured format."""
    as_match = re.search(r'(?:.*\s+as\s+)(\w+)$', column_definition, re.IGNORECASE)
    name = as_match.group(1) if as_match else column_definition.split('.')[-1].strip()
    
    return {
        "fullDefinition": column_definition,
        "name": name,
        "isAggregation": any(agg in column_definition.lower() for agg in ['sum(', 'avg(', 'count(', 'min(', 'max('])
    }

def _find_column_references(sql_content: str, column_name: str) -> List[Dict[str, Any]]:
    """Find all references to a specific column in the SQL content."""
    if not column_name:
        return []
    
    references = []
    lines = sql_content.split('\n')
    
    for i, line in enumerate(lines):
        if column_name in line:
            # Check the context of the reference
            context = "unknown"
            if any(agg + '(' + column_name in line.lower() for agg in ['sum', 'avg', 'count', 'min', 'max']):
                context = "aggregation"
            elif 'select' in line.lower():
                context = "select"
            elif 'where' in line.lower():
                context = "filter"
            elif 'group by' in line.lower():
                context = "groupby"
            elif 'order by' in line.lower():
                context = "orderby"
            elif 'join' in line.lower():
                context = "join"
            
            references.append({
                "lineNumber": i + 1,
                "context": context,
                "line": line.strip()
            })
    
    return references

def find_best_modification_target(analysis: Dict[str, Any], source_column_name: str) -> Dict[str, Any]:
    """
    Find the best target for adding a new aggregation based on an existing column.
    
    Args:
        analysis (Dict[str, Any]): Analysis of the SQL model
        source_column_name (str): The source column to base the aggregation on
        
    Returns:
        Dict[str, Any]: Details of the best modification target
    """
    result = {
        "targetType": None,  # 'aggregation_cte', 'new_cte', 'select_clause'
        "targetName": None,  # Name of CTE or clause
        "targetIndex": -1,   # Index in the list of CTEs
        "reason": "",        # Reason for selection
        "modificationType": None # 'add_to_existing', 'create_new', 'modify_select'
    }
    
    # If there are no CTEs, we'll need to modify the main SELECT
    if not analysis["hasCTEs"]:
        result.update({
            "targetType": "select_clause",
            "reason": "No CTEs found, will modify main SELECT statement",
            "modificationType": "modify_select"
        })
        return result
    
    # Look for the best CTE to modify
    target_cte = None
    target_cte_index = -1
    
    # First priority: aggregation CTEs that already have the source column
    for i, cte in enumerate(analysis["ctes"]):
        if cte["type"] == "aggregation" and "content" in cte and source_column_name in cte["content"]:
            target_cte = cte
            target_cte_index = i
            result.update({
                "targetType": "aggregation_cte",
                "targetName": cte["name"],
                "targetIndex": i,
                "reason": f"Found existing aggregation CTE '{cte['name']}' that already uses column '{source_column_name}'",
                "modificationType": "add_to_existing"
            })
            return result
    
    # Second priority: any aggregation CTE
    for i, cte in enumerate(analysis["ctes"]):
        if cte["type"] == "aggregation":
            target_cte = cte
            target_cte_index = i
            result.update({
                "targetType": "aggregation_cte",
                "targetName": cte["name"],
                "targetIndex": i,
                "reason": f"Using existing aggregation CTE '{cte['name']}' for new aggregation",
                "modificationType": "add_to_existing"
            })
            return result
    
    # Third priority: any CTE with the source column - we'll create a new aggregation CTE after it
    for i, cte in enumerate(analysis["ctes"]):
        if "content" in cte and source_column_name in cte["content"]:
            result.update({
                "targetType": "new_cte",
                "targetName": f"{cte['name']}_agg",
                "targetIndex": i,
                "reason": f"Will create new aggregation CTE after '{cte['name']}' which contains source column",
                "modificationType": "create_new",
                "baseCteName": cte["name"]
            })
            return result
    
    # Last resort: create a new aggregation CTE after the last CTE
    if analysis["ctes"]:
        last_cte = analysis["ctes"][-1]
        result.update({
            "targetType": "new_cte",
            "targetName": "aggregations",
            "targetIndex": len(analysis["ctes"]) - 1,
            "reason": "Will create new aggregation CTE at the end, as no suitable target was found",
            "modificationType": "create_new",
            "baseCteName": last_cte["name"]
        })
    else:
        # Fallback to modifying the main SELECT
        result.update({
            "targetType": "select_clause",
            "reason": "No suitable CTEs found, will modify main SELECT statement",
            "modificationType": "modify_select"
        })
    
    return result

def generate_enhancement_modifications(
    sql_content: str, 
    analysis: Dict[str, Any],
    source_column_name: str, 
    new_column_name: str, 
    aggregation_type: str
) -> Dict[str, Any]:
    """
    Generate the modifications needed to add a new aggregated column.
    
    Args:
        sql_content (str): Original SQL content
        analysis (Dict[str, Any]): Analysis of the SQL model
        source_column_name (str): Source column for aggregation
        new_column_name (str): Name for the new column
        aggregation_type (str): Type of aggregation (sum, avg, count, min, max)
        
    Returns:
        Dict[str, Any]: Result containing success flag, modified code, and details
    """
    # Find the best modification target
    target = find_best_modification_target(analysis, source_column_name)
    
    if target["targetType"] == "aggregation_cte":
        return _modify_existing_aggregation_cte(
            sql_content, analysis, target, source_column_name, new_column_name, aggregation_type
        )
    elif target["targetType"] == "new_cte":
        return _create_new_aggregation_cte(
            sql_content, analysis, target, source_column_name, new_column_name, aggregation_type
        )
    elif target["targetType"] == "select_clause":
        return _modify_select_clause(
            sql_content, analysis, source_column_name, new_column_name, aggregation_type
        )
    else:
        return {
            "success": False,
            "enhancedCode": sql_content,
            "reason": "Couldn't determine appropriate modification strategy"
        }

def _modify_existing_aggregation_cte(
    sql_content: str, 
    analysis: Dict[str, Any], 
    target: Dict[str, Any],
    source_column_name: str, 
    new_column_name: str, 
    aggregation_type: str
) -> Dict[str, Any]:
    """Modify an existing aggregation CTE to add a new aggregated column."""
    modified_code = sql_content
    target_cte = analysis["ctes"][target["targetIndex"]]
    
    # Find existing aggregation pattern for indentation
    agg_pattern = r'(\s+)(?:sum|avg|count|min|max)\s*\([^)]+\)\s+as\s+[\w_]+'
    indentation_match = re.search(agg_pattern, target_cte["content"], re.IGNORECASE)
    indentation = indentation_match.group(1) if indentation_match else '        '
    
    # Create the new aggregation line with proper indentation
    aggregation_line = f"{indentation}{aggregation_type}({source_column_name}) as {new_column_name},"
    
    # Find where to insert in the CTE
    cte_pattern = r'(' + re.escape(target_cte["name"]) + r'\s+as\s+\(\s*\n\s*select[\s\S]*?)(\s+from\s+)'
    cte_match = re.search(cte_pattern, sql_content, re.IGNORECASE)
    
    if not cte_match:
        return {
            "success": False,
            "enhancedCode": sql_content,
            "reason": f"Could not locate SELECT statement in target CTE '{target_cte['name']}'"
        }
    
    select_portion = cte_match.group(1)
    
    # Find the last aggregation in the SELECT
    last_agg_index = max(
        select_portion.lower().rfind('sum('),
        select_portion.lower().rfind('avg('),
        select_portion.lower().rfind('count('),
        select_portion.lower().rfind('min('),
        select_portion.lower().rfind('max(')
    )
    
    if last_agg_index == -1:
        # No existing aggregations, add after the SELECT
        select_index = select_portion.lower().rfind('select')
        if select_index != -1:
            line_end_index = select_portion.find('\n', select_index)
            if line_end_index != -1:
                insertion = (
                    select_portion[:line_end_index + 1] + 
                    f"{indentation}-- Calculate the {aggregation_type} of {source_column_name}\n" +
                    f"{aggregation_line}\n" + 
                    select_portion[line_end_index + 1:]
                )
                modified_code = modified_code.replace(select_portion, insertion)
            else:
                return {
                    "success": False,
                    "enhancedCode": sql_content,
                    "reason": "Could not find end of SELECT line in CTE"
                }
        else:
            return {
                "success": False,
                "enhancedCode": sql_content,
                "reason": "Could not find SELECT statement in CTE"
            }
    else:
        # Insert after the last aggregation line
        line_end_index = select_portion.find('\n', last_agg_index)
        if line_end_index != -1:
            insertion = (
                select_portion[:line_end_index + 1] + 
                f"{indentation}-- Calculate the {aggregation_type} of {source_column_name}\n" +
                f"{aggregation_line}\n" + 
                select_portion[line_end_index + 1:]
            )
            modified_code = modified_code.replace(select_portion, insertion)
        else:
            return {
                "success": False,
                "enhancedCode": sql_content,
                "reason": "Could not find end of last aggregation line"
            }
    
    # Now add the column to the final SELECT or final CTE
    if analysis["finalCTE"]:
        # Add to the final CTE
        final_pattern = r'(final\s+as\s+\(\s*\n\s*select[\s\S]*?' + re.escape(target_cte["name"]) + r'\.\w+,[\s\S]*?)(?=\s+from\s+)'
        final_match = re.search(final_pattern, modified_code, re.IGNORECASE)
        
        if final_match:
            indent_match = re.search(r'\n(\s+)\w', final_match.group(1))
            final_indent = indent_match.group(1) if indent_match else '        '
            
            # Add the column to the final select
            final_insertion = final_match.group(1) + f"\n{final_indent}{target_cte['name']}.{new_column_name},"
            modified_code = modified_code.replace(final_match.group(1), final_insertion)
            
            return {
                "success": True,
                "enhancedCode": modified_code,
                "details": {
                    "modifiedCte": target_cte["name"],
                    "addedColumn": new_column_name,
                    "addedToFinal": True
                }
            }
        else:
            return {
                "success": False,
                "enhancedCode": modified_code,
                "reason": "Added column to CTE but could not locate target in final SELECT"
            }
    elif analysis["finalSelect"]:
        # Add to the main SELECT statement
        select_index = modified_code.lower().rfind('select')
        from_index = modified_code.lower().find('from', select_index)
        
        if select_index != -1 and from_index != -1:
            select_clause = modified_code[select_index:from_index]
            indent_match = re.search(r'\n(\s+)\w', select_clause)
            select_indent = indent_match.group(1) if indent_match else '    '
            
            # Add column to the select clause
            select_insertion = select_clause + f"\n{select_indent}{target_cte['name']}.{new_column_name},"
            modified_code = modified_code.replace(select_clause, select_insertion)
            
            return {
                "success": True,
                "enhancedCode": modified_code,
                "details": {
                    "modifiedCte": target_cte["name"],
                    "addedColumn": new_column_name,
                    "addedToFinal": True
                }
            }
        else:
            return {
                "success": False,
                "enhancedCode": modified_code,
                "reason": "Added column to CTE but could not locate main SELECT"
            }
    else:
        return {
            "success": False,
            "enhancedCode": modified_code,
            "reason": "Added column to CTE but could not find a final SELECT to add it to"
        }

def _create_new_aggregation_cte(
    sql_content: str, 
    analysis: Dict[str, Any], 
    target: Dict[str, Any],
    source_column_name: str, 
    new_column_name: str, 
    aggregation_type: str
) -> Dict[str, Any]:
    """Create a new aggregation CTE after an existing CTE."""
    base_cte_name = target.get("baseCteName")
    new_cte_name = target["targetName"]
    
    # Find the base CTE to insert after
    cte_pattern = re.escape(base_cte_name) + r'\s+as\s+\([\s\S]*?\),\s*\n'
    cte_match = re.search(cte_pattern, sql_content, re.IGNORECASE)
    
    if not cte_match:
        return {
            "success": False,
            "enhancedCode": sql_content,
            "reason": f"Could not locate base CTE '{base_cte_name}' in SQL content"
        }
    
    # Create a new aggregation CTE
    new_cte = (
        f"{base_cte_name} as ([\s\S]*?\\),\n"
        f"{new_cte_name} as (\n"
        f"    select\n"
        f"        {base_cte_name}.*,\n"
        f"        {aggregation_type}({source_column_name}) as {new_column_name}\n"
        f"    from {base_cte_name}\n"
        f"    group by 1\n"
        f"),\n"
    )
    
    modified_code = re.sub(
        re.escape(cte_match.group(0)),
        new_cte,
        sql_content
    )
    
    # Add to final CTE or select
    if analysis["finalCTE"]:
        final_select_pattern = r'final\s+as\s+\(\s*\n\s*select\s+([\s\S]*?)(?=\s+from\s+)'
        final_select_match = re.search(final_select_pattern, modified_code, re.IGNORECASE)
        
        if final_select_match:
            indent_match = re.search(r'\n(\s+)\w', final_select_match.group(1))
            final_indent = indent_match.group(1) if indent_match else '        '
            
            final_insertion = final_select_match.group(1) + f"\n{final_indent}{new_cte_name}.{new_column_name},"
            modified_code = modified_code.replace(final_select_match.group(1), final_insertion)
            
            # Update the from clause to join the new CTE
            from_pattern = r'from\s+([\s\S]*?)(?=where|group|order|$)'
            from_match = re.search(from_pattern, modified_code, re.IGNORECASE)
            
            if from_match and new_cte_name not in from_match.group(1):
                join_indent = final_indent
                join_clause = (
                    from_match.group(1) + 
                    f"\n{join_indent}left join {new_cte_name}\n"
                    f"{join_indent}    on {base_cte_name}.{base_cte_name}_key = {new_cte_name}.{base_cte_name}_key"
                )
                modified_code = modified_code.replace(from_match.group(1), join_clause)
                
                return {
                    "success": True,
                    "enhancedCode": modified_code,
                    "details": {
                        "createdCte": new_cte_name,
                        "basedOn": base_cte_name,
                        "addedColumn": new_column_name,
                        "addedToFinal": True
                    }
                }
            else:
                return {
                    "success": False,
                    "enhancedCode": modified_code,
                    "reason": "Created new CTE but could not update FROM clause with join"
                }
        else:
            return {
                "success": False,
                "enhancedCode": modified_code,
                "reason": "Created new CTE but could not find final SELECT to add column to"
            }
    else:
        return {
            "success": False,
            "enhancedCode": modified_code,
            "reason": "Created new CTE but could not find a final SELECT or final CTE"
        }

def _modify_select_clause(
    sql_content: str, 
    analysis: Dict[str, Any],
    source_column_name: str, 
    new_column_name: str, 
    aggregation_type: str
) -> Dict[str, Any]:
    """Modify a simple SELECT statement to add a new aggregated column."""
    if not (sql_content.lower().find('select') != -1 and sql_content.lower().find('from') != -1):
        return {
            "success": False,
            "enhancedCode": sql_content,
            "reason": "Could not find basic SELECT...FROM structure in SQL"
        }
    
    select_index = sql_content.lower().find('select')
    from_index = sql_content.lower().find('from', select_index)
    
    select_clause = sql_content[select_index:from_index]
    
    # Look for indentation pattern
    indent_match = re.search(r'\n(\s+)\w', select_clause)
    select_indent = indent_match.group(1) if indent_match else '    '
    
    # Add column to the select clause
    select_insertion = select_clause + f"\n{select_indent}{aggregation_type}({source_column_name}) as {new_column_name},"
    
    # Check if we need to add a GROUP BY
    has_group_by = 'group by' in sql_content.lower()
    
    if not has_group_by:
        # Add group by if not present
        modified_code = sql_content.replace(select_clause, select_insertion)
        
        # Find a good place to add the GROUP BY
        order_by_index = modified_code.lower().find('order by')
        limit_index = modified_code.lower().find('limit')
        insert_index = order_by_index if order_by_index != -1 else (
            limit_index if limit_index != -1 else len(modified_code)
        )
        
        # Try to identify columns to group by (exclude aggregated columns)
        # This is an approximation - would need more complex parsing for accuracy
        potential_group_cols = []
        for col in select_clause.split(','):
            # Skip columns with aggregations
            if any(agg in col.lower() for agg in ['sum(', 'avg(', 'count(', 'min(', 'max(']):
                continue
            # Extract the column name
            col_name = col.strip()
            if ' as ' in col_name.lower():
                col_name = col_name.split(' as ')[-1].strip()
            potential_group_cols.append(col_name)
        
        group_by_clause = ""
        if potential_group_cols:
            group_by_clause = (
                f"\n{select_indent}group by\n"
                f"{select_indent}    {','.join(potential_group_cols)}\n"
            )
            
            # Insert the GROUP BY
            modified_code = (
                modified_code[:insert_index] + 
                group_by_clause + 
                modified_code[insert_index:]
            )
            
            return {
                "success": True,
                "enhancedCode": modified_code,
                "details": {
                    "addedColumn": new_column_name,
                    "addedGroupBy": True
                }
            }
        else:
            return {
                "success": False,
                "enhancedCode": modified_code,
                "reason": "Added column but could not identify appropriate GROUP BY columns"
            }
    else:
        # Just add the column if GROUP BY already exists
        modified_code = sql_content.replace(select_clause, select_insertion)
        return {
            "success": True,
            "enhancedCode": modified_code,
            "details": {
                "addedColumn": new_column_name,
                "addedGroupBy": False
            }
        }

def generate_suggested_approach(
    analysis: Dict[str, Any], 
    source_column_name: str, 
    new_column_name: str, 
    aggregation_type: str
) -> str:
    """Generate suggested approach for manual implementation when automatic modification fails."""
    suggestions = []
    
    if analysis["ctes"]:
        # Find potential places for modification
        aggregation_ctes = [cte for cte in analysis["ctes"] if cte["type"] == "aggregation"]
        source_ctes = [cte for cte in analysis["ctes"] if source_column_name in cte.get("content", "")]
        
        if aggregation_ctes:
            target_cte = next((cte for cte in aggregation_ctes if source_column_name in cte.get("content", "")), aggregation_ctes[0])
            suggestions.append(f"-- 1. Add to the '{target_cte['name']}' CTE: {aggregation_type}({source_column_name}) as {new_column_name}")
        elif source_ctes:
            source_cte = source_ctes[0]
            suggestions.append(f"-- 1. Create a new aggregation CTE after '{source_cte['name']}' that computes {aggregation_type}({source_column_name}) as {new_column_name}")
        
        # Add suggestion for the final SELECT or final CTE
        if analysis["finalCTE"]:
            suggestions.append(f"-- 2. Add the new column to the final CTE SELECT statement")
        else:
            suggestions.append(f"-- 2. Add the new column to the main SELECT statement")
    else:
        # Simple SQL suggestions
        suggestions.append(f"-- 1. Add {aggregation_type}({source_column_name}) as {new_column_name} to the SELECT clause")
        
        if not analysis["groupBys"]:
            suggestions.append(f"-- 2. Add an appropriate GROUP BY clause for non-aggregated columns")
    
    return '\n'.join(suggestions)

def generate_failure_feedback(
    sql_content: str, 
    source_column_name: str, 
    new_column_name: str, 
    aggregation_type: str
) -> str:
    """Generate detailed feedback when modification fails."""
    feedback = []
    
    # Check for source column existence
    if source_column_name not in sql_content:
        feedback.append(f"-- The source column '{source_column_name}' could not be found in the model.")
        feedback.append(f"-- Check for typos or ensure this column exists before aggregating it.")
    else:
        feedback.append(f"-- The source column '{source_column_name}' was found, but the structure is complex.")
    
    # Analyze model structure
    if 'with ' in sql_content and ' as (' in sql_content:
        feedback.append(f"-- This appears to be a model with CTEs. You should add the aggregation to an appropriate CTE")
        feedback.append(f"-- and then reference it in the final SELECT statement.")
    elif 'select' in sql_content and 'from' in sql_content:
        feedback.append(f"-- This appears to be a simple SELECT query. Add the aggregation to the SELECT clause")
        feedback.append(f"-- and add an appropriate GROUP BY clause if needed.")
    
    return '\n'.join(feedback) 