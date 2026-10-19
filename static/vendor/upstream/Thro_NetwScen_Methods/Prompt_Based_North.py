# 2025-03-25 13:30 Create prompt-based network slicing implementation
# 2025-03-25 14:45 Add constraint checking function for allocation validation
import json
import math
import random
from datetime import datetime
import re
from tabulate import tabulate
import pandas as pd
import csv
import requests
from typing import Dict, List, Any, Optional

# ====================== LLM Call Function ======================

def call_llm(prompt):
    """Call the LLM API with the given prompt"""
    try:
        api_key = "REDACTED_CONFIGURE_AT_RUNTIME"  # Replace with your API key
        base_url = "https://api.deepseek.com"
        model = "deepseek-chat"
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
        
        data = {
            "model": model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0
        }
        
        response = requests.post(f"{base_url}/v1/chat/completions", headers=headers, json=data)
        response.raise_for_status()
        
        result = response.json()
        return result["choices"][0]["message"]["content"]
    except Exception as e:
        print(f"Error calling LLM API: {e}")
        return "Error: Unable to get a response from the LLM API."


# ====================== CSV Data Loading Function ======================

def load_user_data_from_csv(file_path, num_users=None):
    """Load user data from ray tracing CSV file"""
    try:
        df = pd.read_csv(file_path)
        users = []
        
        # Limit to specified number of users if needed
        if num_users is not None and num_users < len(df):
            df = df.head(num_users)
        
        for _, row in df.iterrows():
            # Get user ID (RX_ID)
            user_id = str(row['RX_ID'])
            
            # Create location string from X, Y, Z coordinates
            location = f"({row['X']}, {row['Y']}, {row['Z']})"
            
            # Get request, CQI, and ground truth label
            request = row['User_Request']
            cqi = int(row['CQI'])
            
            # Get ground truth label if available
            ground_truth = row.get('Request_Label', None)
            
            user = {
                "user_id": user_id,
                "location": location,
                "request": request,
                "cqi": cqi,
                "ground_truth": ground_truth
            }
            users.append(user)
        
        return users
    except Exception as e:
        print(f"Error loading user data from CSV: {e}")
        # Return some default users as fallback
        return [
            {
                "user_id": "1",
                "location": "(-39.01, -0.50, 1.50)",
                "request": "I need to stream 8K video content",
                "cqi": 15,
                "ground_truth": "eMBB"
            },
            {
                "user_id": "2",
                "location": "(-62.97, 141.28, 1.50)",
                "request": "I want to watch 4K video",
                "cqi": 9,
                "ground_truth": "eMBB"
            }
        ]

# ====================== CSV Results Export Function ======================

def export_results_to_csv(results, slice_stats, intent_stats, file_path="prompt_based_slicing_results.csv"):
    """Export test results to a CSV file with enhanced analytics"""
    try:
        # Define CSV headers
        headers = [
            "User ID", "Allocation Status", "Slice", "Ground Truth", "Intent Correct", "CQI", "Bandwidth (MHz)", 
            "Rate (Mbps)", "Latency (ms)", "Constraint Violation", "Request"
        ]
        
        # Open file for writing
        with open(file_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            
            # Write headers
            writer.writerow(headers)
            
            # Write data rows
            for result in results:
                allocation_status = "Failed" if result.get("allocation_failed", True) else "Success"
                constraint_violation = "Yes" if result.get("constraint_violation", False) else "No"
                
                row = [
                    result["user_id"],
                    allocation_status,
                    result["slice_type"],
                    result.get("ground_truth", "Unknown"),
                    "Yes" if result.get("intent_correct", None) is True else 
                    "No" if result.get("intent_correct", None) is False else "N/A",
                    result["cqi"],
                    result.get("bandwidth", "N/A"),
                    result.get("rate", "N/A"),
                    result.get("latency", "N/A"),
                    constraint_violation,
                    result["request"]
                ]
                writer.writerow(row)
            
            # Add summary statistics
            writer.writerow([])
            writer.writerow(["SUMMARY STATISTICS"])
            writer.writerow(["Average eMBB Utilization (%)", slice_stats["avg_embb_util"]])
            writer.writerow(["Average URLLC Utilization (%)", slice_stats["avg_urllc_util"]])
            writer.writerow(["Constraint Violations", slice_stats["constraint_violations"]])
            
            # Add intent understanding rate
            writer.writerow([])
            writer.writerow(["INTENT UNDERSTANDING EVALUATION"])
            writer.writerow(["Total Evaluated Requests", intent_stats["total"]])
            writer.writerow(["Correctly Identified Intents", intent_stats["correct"]])
            writer.writerow(["Intent Understanding Rate (%)", intent_stats["rate"]])
        
        print(f"\nResults exported to {file_path}")
        return True
    except Exception as e:
        print(f"Error exporting results to CSV: {e}")
        return False

# ====================== Global Network State ======================

# Persistent state of network slices
GLOBAL_NETWORK_STATE = {
    "embb_slice": {
        "users": [],
        "total_capacity": 90,
        "resource_usage": 0,
        "utilization_rate": "0%"
    },
    "urllc_slice": {
        "users": [],
        "total_capacity": 30,
        "resource_usage": 0,
        "utilization_rate": "0%"
    },
    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "total_users": 0
}

# Store initial network state for reset operations
INITIAL_NETWORK_STATE = GLOBAL_NETWORK_STATE.copy()

def get_current_network_state():
    """Get a copy of the current network state"""
    return GLOBAL_NETWORK_STATE.copy()

def update_network_state(new_state):
    """Update global network state"""
    global GLOBAL_NETWORK_STATE
    GLOBAL_NETWORK_STATE = new_state
    return True

def reset_network_state():
    """Reset network state to initial values for fresh testing"""
    global GLOBAL_NETWORK_STATE
    
    # Reset to initial state
    GLOBAL_NETWORK_STATE = INITIAL_NETWORK_STATE.copy()
    
    # Reset timestamp
    GLOBAL_NETWORK_STATE["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    return True

def calculate_utilization_rate(usage, capacity):
    """Calculate utilization rate percentage"""
    return f"{(usage / capacity * 100):.2f}%"

# ====================== Constraint Checking Function ======================

def check_allocation_constraints(slice_type, bandwidth, rate, latency):
    """Check if the allocation meets the slice constraints
    
    Parameters:
    - slice_type: "eMBB" or "URLLC"
    - bandwidth: Allocated bandwidth in MHz
    - rate: Calculated data rate in Mbps
    - latency: Allocated latency in ms
    
    Returns:
    - (valid, reasons): Tuple of boolean indicating if constraints are met and list of reasons if not
    """
    valid = True
    reasons = []
    
    if slice_type == "eMBB":
        # Check bandwidth constraints
        if bandwidth < 6 or bandwidth > 20:
            valid = False
            reasons.append(f"eMBB bandwidth must be 6-20 MHz (allocated: {bandwidth} MHz)")
            
        # Check rate constraints
        if rate < 100 or rate > 400:
            valid = False
            reasons.append(f"eMBB data rate must be 100-400 Mbps (calculated: {rate:.2f} Mbps)")
            
        # Check latency constraints
        if latency < 10 or latency > 100:
            valid = False
            reasons.append(f"eMBB latency must be 10-100 ms (allocated: {latency} ms)")
            
    elif slice_type == "URLLC":
        # Check bandwidth constraints
        if bandwidth < 1 or bandwidth > 5:
            valid = False
            reasons.append(f"URLLC bandwidth must be 1-5 MHz (allocated: {bandwidth} MHz)")
            
        # Check rate constraints
        if rate < 1 or rate > 100:
            valid = False
            reasons.append(f"URLLC data rate must be 1-100 Mbps (calculated: {rate:.2f} Mbps)")
            
        # Check latency constraints
        if latency < 1 or latency > 10:
            valid = False
            reasons.append(f"URLLC latency must be 1-10 ms (allocated: {latency} ms)")
    
    return valid, reasons

# ====================== Network State Reporting Functions ======================

def generate_concise_report(current_state, new_user_id=None):
    """Generate a concise network status report"""
    # Get slice information
    embb_slice = current_state["embb_slice"]
    urllc_slice = current_state["urllc_slice"]
    
    # Basic network statistics
    stats = [
        ["Slice", "Users", "Resource Usage", "Utilization"],
        ["eMBB", len(embb_slice["users"]), f"{embb_slice['resource_usage']}/{embb_slice['total_capacity']} MHz", embb_slice["utilization_rate"]],
        ["URLLC", len(urllc_slice["users"]), f"{urllc_slice['resource_usage']}/{urllc_slice['total_capacity']} MHz", urllc_slice["utilization_rate"]]
    ]
    
    report = [
        f"Network Status @ {current_state['timestamp']}",
        f"Total Users: {current_state['total_users']}",
        "",
        tabulate(stats, headers="firstrow", tablefmt="simple")
    ]
    
    # If a new user was added, show their details
    if new_user_id:
        new_user = None
        new_user_slice = None
        
        # Find the new user
        for slice_key in ["embb_slice", "urllc_slice"]:
            for user in current_state[slice_key]["users"]:
                if user["user_id"] == new_user_id:
                    new_user = user
                    new_user_slice = "eMBB" if slice_key == "embb_slice" else "URLLC"
                    break
            if new_user:
                break
        
        if new_user:
            report.append("\nNew User Allocation:")
            report.append(f"User {new_user_id} → {new_user_slice} Slice")
            report.append(f"CQI: {new_user['cqi']}, Bandwidth: {new_user['bandwidth']} MHz, Rate: {new_user['rate']:.2f} Mbps, Latency: {new_user['latency']} ms")
    
    return "\n".join(report)

def generate_user_allocation_table(current_state, new_user_id=None):
    """Generate a table showing all user allocations"""
    # Combine all users into one list for the table
    all_users = []
    
    # Add eMBB users
    for user in current_state["embb_slice"]["users"]:
        status = "NEW" if user["user_id"] == new_user_id else ""
        
        all_users.append({
            "user_id": user["user_id"],
            "slice": "eMBB",
            "cqi": user["cqi"],
            "bandwidth": user["bandwidth"],
            "rate": user["rate"],
            "latency": user["latency"],
            "status": status
        })
    
    # Add URLLC users
    for user in current_state["urllc_slice"]["users"]:
        status = "NEW" if user["user_id"] == new_user_id else ""
        
        all_users.append({
            "user_id": user["user_id"],
            "slice": "URLLC",
            "cqi": user["cqi"],
            "bandwidth": user["bandwidth"],
            "rate": user["rate"],
            "latency": user["latency"],
            "status": status
        })
    
    # Sort by slice type and then by user_id
    all_users.sort(key=lambda x: (x["slice"], x["user_id"]))
    
    # Format data for table
    rows = []
    for user in all_users:
        row = [
            user["user_id"],
            user["slice"],
            user["cqi"],
            user["bandwidth"],
            f"{user['rate']:.2f}",
            user["latency"],
            user["status"]
        ]
        rows.append(row)
    
    # Create table
    headers = ["User ID", "Slice", "CQI", "BW (MHz)", "Rate (Mbps)", "Latency (ms)", "Status"]
    table = tabulate(rows, headers=headers, tablefmt="grid")
    
    return f"\nCurrent User Allocations:\n{table}"


# ====================== Network Slicing Prompt Function ======================

# System prompt for the LLM
SYSTEM_PROMPT = """You are a 5G network slicing expert responsible for allocating users to appropriate network slices and managing resources. You need to perform the following sequential tasks:

1. INTENT UNDERSTANDING: Analyze the user's request to understand their application needs.

2. SLICE RECOMMENDATION: Recommend the most appropriate network slice:
   - eMBB (Enhanced Mobile Broadband): For high bandwidth applications like video streaming
     * Bandwidth range: 6-20 MHz
     * Data rate range: 100-400 Mbps
     * Latency range: 10-100ms
   - URLLC (Ultra-Reliable Low-Latency Communications): For low latency applications like remote control
     * Bandwidth range: 1-5 MHz
     * Data rate range: 1-100 Mbps
     * Latency range: 1-10ms

3. RATE ALLOCATION: Allocate bandwidth and calculate data rate based on the user's CQI (Channel Quality Indicator) and typical requirements:
   - CQI ranges from 1-15 (higher is better signal quality)
   - A rough estimate for data rate for a given bandwidth and CQI is:
     * Low CQI (1-5): bandwidth × 5 Mbps
     * Medium CQI (6-10): bandwidth × 10 Mbps
     * High CQI (11-15): bandwidth × 15 Mbps

4. RATE ADJUSTMENT: Adjust bandwidth if needed to meet slice requirements:
   - For eMBB: Ensure rate is between 100-400 Mbps
   - For URLLC: Ensure rate is between 1-100 Mbps
   - Adjust bandwidth up or down to meet these requirements

5. WORKLOAD BALANCE: If one slice is significantly more utilized than the other (>20% difference) and the user could be accommodated in either slice, prefer the less utilized slice.

6. CAPACITY CHECK: Verify if the slice has enough capacity for the new user:
   - eMBB total capacity: 90 MHz
   - URLLC total capacity: 30 MHz
   - If not enough capacity, report failure

IMPORTANT: You must strictly adhere to the resource constraints:
- eMBB: Bandwidth 6-20 MHz, Rate 100-400 Mbps, Latency 10-100ms
- URLLC: Bandwidth 1-5 MHz, Rate 1-100 Mbps, Latency 1-10ms
The system will verify these constraints after your allocation, and any violation will cause the allocation to fail.

Format your response as JSON with the following fields:
{
  "intent_analysis": "Explanation of user's needs and application type",
  "recommended_slice": "eMBB or URLLC",
  "slice_reason": "Explanation for slice choice",
  "bandwidth_allocation": integer value in MHz,
  "data_rate": float value in Mbps,
  "latency": integer value in ms,
  "workload_balanced": boolean,
  "can_accommodate": boolean,
  "final_allocation": {
    "user_id": "string",
    "slice_type": "eMBB or URLLC",
    "bandwidth": integer in MHz,
    "rate": float in Mbps,
    "latency": integer in ms
  }
}
"""

def process_user_with_prompt(user_id, location, request, cqi, network_state):
    """Process a user request using the prompt-based approach"""
    # Construct the prompt with user information and network state
    embb_slice = network_state["embb_slice"]
    urllc_slice = network_state["urllc_slice"]
    
    embb_usage = embb_slice["resource_usage"]
    urllc_usage = urllc_slice["resource_usage"]
    embb_capacity = embb_slice["total_capacity"]
    urllc_capacity = urllc_slice["total_capacity"]
    embb_utilization = float(embb_slice["utilization_rate"].replace("%", ""))
    urllc_utilization = float(urllc_slice["utilization_rate"].replace("%", ""))
    
    prompt = f"""
Please allocate network resources for the following user:

USER INFORMATION:
- User ID: {user_id}
- Location: {location}
- Request: "{request}"
- CQI (Channel Quality Indicator): {cqi}

CURRENT NETWORK STATE:
- eMBB Slice:
  * Users: {len(embb_slice["users"])}
  * Resource Usage: {embb_usage}/{embb_capacity} MHz
  * Utilization Rate: {embb_utilization:.2f}%

- URLLC Slice:
  * Users: {len(urllc_slice["users"])}
  * Resource Usage: {urllc_usage}/{urllc_capacity} MHz
  * Utilization Rate: {urllc_utilization:.2f}%

Based on this information:
1. Analyze the user's intent
2. Recommend appropriate network slice
3. Allocate bandwidth and calculate data rate
4. Adjust rate if needed
5. Consider workload balance between slices
6. Verify capacity availability

IMPORTANT: Remember to strictly adhere to the following constraints:
- eMBB: Bandwidth 6-20 MHz, Rate 100-400 Mbps, Latency 10-100ms
- URLLC: Bandwidth 1-5 MHz, Rate 1-100 Mbps, Latency 1-10ms

Provide your response in the JSON format specified in your instructions.
"""

    # Call the LLM and get the response
    response = call_llm(prompt)
    
    # Parse the JSON response
    try:
        # Extract JSON from response (in case there's additional text)
        json_match = re.search(r'({[\s\S]*})', response)
        if json_match:
            json_str = json_match.group(1)
            result = json.loads(json_str)
        else:
            raise ValueError("No JSON found in response")
            
        print(f"\nIntent Analysis: {result['intent_analysis']}")
        print(f"Recommended Slice: {result['recommended_slice']} - {result['slice_reason']}")
        print(f"Bandwidth Allocation: {result['bandwidth_allocation']} MHz")
        print(f"Data Rate: {result['data_rate']} Mbps")
        print(f"Latency: {result['latency']} ms")
        print(f"Workload Balanced: {'Yes' if result.get('workload_balanced', False) else 'No'}")
        
        # Check if the allocation meets constraints
        slice_type = result["recommended_slice"]
        bandwidth = result["bandwidth_allocation"]
        rate = result["data_rate"]
        latency = result["latency"]
        
        valid, constraint_reasons = check_allocation_constraints(slice_type, bandwidth, rate, latency)
        
        if not valid:
            print(f"\nCONSTRAINT VIOLATION DETECTED:")
            for reason in constraint_reasons:
                print(f"- {reason}")
                
            # Return allocation failure due to constraint violation
            return {
                "user_id": user_id,
                "request": request,
                "cqi": cqi,
                "slice_type": slice_type,
                "bandwidth": bandwidth, 
                "rate": rate,
                "latency": latency,
                "allocation_failed": True,
                "intent_analysis": result["intent_analysis"],
                "slice_reason": result["slice_reason"],
                "failure_reason": "Constraint violation: " + "; ".join(constraint_reasons),
                "constraint_violation": True
            }
        
        # If the LLM indicates we can accommodate the user AND constraints are met
        if result.get("can_accommodate", False):
            # Check capacity availability
            slice_key = "embb_slice" if slice_type == "eMBB" else "urllc_slice"
            available_capacity = network_state[slice_key]["total_capacity"] - network_state[slice_key]["resource_usage"]
            
            if available_capacity < bandwidth:
                print(f"\nCAPACITY CHECK FAILED:")
                print(f"- Required: {bandwidth} MHz, Available: {available_capacity} MHz in {slice_type} slice")
                
                # Return allocation failure due to capacity constraints
                return {
                    "user_id": user_id,
                    "request": request,
                    "cqi": cqi,
                    "slice_type": slice_type,
                    "bandwidth": bandwidth,
                    "rate": rate,
                    "latency": latency,
                    "allocation_failed": True,
                    "intent_analysis": result["intent_analysis"],
                    "slice_reason": result["slice_reason"],
                    "failure_reason": f"Insufficient capacity in {slice_type} slice. Required: {bandwidth} MHz, Available: {available_capacity} MHz"
                }
            
            # All checks passed, proceed with allocation
            # Create new user entry
            new_user = {
                "user_id": user_id,
                "rate": rate,
                "latency": latency,
                "cqi": cqi,
                "bandwidth": bandwidth
            }
            
            # Add user and update resource usage
            network_state[slice_key]["users"].append(new_user)
            network_state[slice_key]["resource_usage"] += bandwidth
            
            # Update utilization rate
            network_state[slice_key]["utilization_rate"] = calculate_utilization_rate(
                network_state[slice_key]["resource_usage"],
                network_state[slice_key]["total_capacity"]
            )
            
            # Update total user count
            network_state["total_users"] = len(network_state["embb_slice"]["users"]) + len(network_state["urllc_slice"]["users"])
            
            # Update timestamp
            network_state["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            # Update global state
            update_network_state(network_state)
            
            print(f"\nALLOCATION SUCCESSFUL: All constraints satisfied")
            
            # Return the detailed results
            return {
                "user_id": user_id,
                "request": request,
                "cqi": cqi,
                "slice_type": slice_type,
                "bandwidth": bandwidth,
                "rate": rate,
                "latency": latency,
                "allocation_failed": False,
                "intent_analysis": result["intent_analysis"],
                "slice_reason": result["slice_reason"],
                "workload_balanced": result.get("workload_balanced", False)
            }
        else:
            # LLM determined it cannot accommodate the user
            return {
                "user_id": user_id,
                "request": request,
                "cqi": cqi,
                "slice_type": slice_type,
                "bandwidth": bandwidth,
                "rate": rate,
                "latency": latency,
                "allocation_failed": True,
                "intent_analysis": result["intent_analysis"],
                "slice_reason": result["slice_reason"],
                "failure_reason": "LLM determined it cannot accommodate user based on current network state"
            }
            
    except Exception as e:
        print(f"Error parsing LLM response: {e}")
        print(f"Raw response: {response}")
        
        # Return error result
        return {
            "user_id": user_id,
            "request": request,
            "cqi": cqi,
            "slice_type": "Failed",
            "allocation_failed": True,
            "error": str(e)
        }

# ====================== Main Process Function ======================

def process_user_request(user_id, location, request, cqi, ground_truth=None):
    """Main function for processing user requests with prompt-based approach"""
    # Get current network state
    network_state = get_current_network_state()
    
    # Record initial state for comparison
    initial_embb_usage = network_state["embb_slice"]["resource_usage"]
    initial_urllc_usage = network_state["urllc_slice"]["resource_usage"]
    initial_embb_util = network_state["embb_slice"]["utilization_rate"]
    initial_urllc_util = network_state["urllc_slice"]["utilization_rate"]
    
    # Process user with the prompt-based approach
    result = process_user_with_prompt(user_id, location, request, cqi, network_state)
    
    # Get updated network state
    updated_state = get_current_network_state()
    
    # Print concise report and user allocation table
    if not result.get("allocation_failed", True):
        print("\n" + "-"*40)
        print(f"ALLOCATION RESULT FOR USER {user_id}")
        print("-"*40)
        concise_report = generate_concise_report(updated_state, user_id)
        print(concise_report)
        
        # Print complete user allocation table
        user_table = generate_user_allocation_table(updated_state, user_id)
        print(user_table)
    else:
        print("\n" + "-"*40)
        print(f"ALLOCATION FAILED FOR USER {user_id}")
        print("-"*40)
        print(f"Request: {request}")
        print(f"Slice type: {result.get('slice_type', 'Unknown')}")
        print(f"Reason: {result.get('failure_reason', 'Unknown error')}")
        
        if result.get("constraint_violation", False):
            print(f"Violation Type: Constraint Violation")
            print(f"Attempted values: Bandwidth {result.get('bandwidth', 'N/A')} MHz, Rate {result.get('rate', 'N/A')} Mbps, Latency {result.get('latency', 'N/A')} ms")
    
    # Add comparison data
    final_embb_usage = updated_state["embb_slice"]["resource_usage"]
    final_urllc_usage = updated_state["urllc_slice"]["resource_usage"]
    final_embb_util = updated_state["embb_slice"]["utilization_rate"]
    final_urllc_util = updated_state["urllc_slice"]["utilization_rate"]
    
    result["embb_usage_before"] = initial_embb_usage
    result["embb_usage_after"] = final_embb_usage
    result["embb_util_before"] = initial_embb_util
    result["embb_util_after"] = final_embb_util
    result["urllc_usage_before"] = initial_urllc_usage
    result["urllc_usage_after"] = final_urllc_usage
    result["urllc_util_before"] = initial_urllc_util
    result["urllc_util_after"] = final_urllc_util
    
    # Check if intent understanding matches ground truth
    if ground_truth is not None and not result.get("allocation_failed", True):
        result["intent_correct"] = (result["slice_type"] == ground_truth)
    else:
        result["intent_correct"] = None
    
    result["ground_truth"] = ground_truth
    
    return result

# ====================== Main Function ======================

def main(num_users=4, export_file="prompt_based_slicing_results.csv"):
    """Main program with CSV-based user testing"""
    print("Starting prompt-based network slice management system...\n")
    
    # Path to ray tracing results CSV
    ray_tracing_csv = r"C:\Users\Jingwen TONG\Desktop\我的文档\02_项目_202301\16-WirelessAgent-ChinaCom\Simulations\WirelessAgent_LangGraph\Knowledge_Base\ray_tracing_results_north.csv"
    
    # Load users from CSV (limit to specified number)
    users = load_user_data_from_csv(ray_tracing_csv, num_users)
    
    print(f"Testing {len(users)} users from ray tracing results CSV")
    
    # Initialize results tracker
    detailed_results = []
    
    # Track slice utilization and constraint violations
    embb_utils = []
    urllc_utils = []
    constraint_violations = 0
    
    # Process each user
    for i, user in enumerate(users):
        print(f"\n{'-'*40}")
        print(f"PROCESSING USER {user['user_id']} ({i+1}/{len(users)})")
        print(f"Request: \"{user['request']}\"")
        print(f"CQI: {user['cqi']}")
        if user.get('ground_truth'):
            print(f"Ground Truth Slice: {user['ground_truth']}")
        print(f"{'-'*40}")
        
        # Reset network state if testing each user independently
        # Comment this line if you want to test cumulative allocations
        reset_network_state() 
        
        # Process the user
        result = process_user_request(
            user_id=user['user_id'],
            location=user['location'],
            request=user['request'],
            cqi=user['cqi'],
            ground_truth=user.get('ground_truth')
        )
        
        # Check for constraint violations
        if result.get("constraint_violation", False):
            constraint_violations += 1
        
        # Store detailed result for CSV export
        detailed_results.append(result)
        
        # Track slice utilization
        if not result.get("allocation_failed", True):
            # Convert percentage string to float
            try:
                if "embb_util_after" in result:
                    embb_util_str = result["embb_util_after"]
                    if isinstance(embb_util_str, str):
                        embb_util = float(embb_util_str.replace("%", ""))
                    else:
                        embb_util = float(embb_util_str)
                    embb_utils.append(embb_util)
                
                if "urllc_util_after" in result:
                    urllc_util_str = result["urllc_util_after"]
                    if isinstance(urllc_util_str, str):
                        urllc_util = float(urllc_util_str.replace("%", ""))
                    else:
                        urllc_util = float(urllc_util_str)
                    urllc_utils.append(urllc_util)
            except (ValueError, AttributeError):
                pass
    
    # Get final network state
    final_state = get_current_network_state()
    
    # Calculate average slice utilization
    avg_embb_util = sum(embb_utils) / len(embb_utils) if embb_utils else 0
    avg_urllc_util = sum(urllc_utils) / len(urllc_utils) if urllc_utils else 0
    
    # Calculate intent understanding rate
    correct_intents = 0
    total_evaluated = 0
    
    for result in detailed_results:
        if result.get("intent_correct") is not None:
            total_evaluated += 1
            if result["intent_correct"]:
                correct_intents += 1
    
    intent_rate = 0 if total_evaluated == 0 else (correct_intents / total_evaluated) * 100
    
    # Print summary of all results
    print("\n" + "="*60)
    print("SUMMARY OF USER ALLOCATIONS")
    print("="*60)
    
    # Generate summary table
    summary_rows = []
    for res in detailed_results:
        status = "Success" if not res.get("allocation_failed", True) else "Failed"
        intent_match = ""
        if res.get("intent_correct") is not None:
            intent_match = "Yes" if res["intent_correct"] else "No"
        
        constraint_viol = "Yes" if res.get("constraint_violation", False) else "No"
        
        summary_rows.append([
            res["user_id"],
            status,
            res.get("slice_type", "Failed"),
            res.get("ground_truth", "N/A"),
            intent_match,
            res["cqi"],
            res.get("bandwidth", "N/A"),
            f"{res.get('rate', 'N/A')}" if res.get("rate") is not None else "N/A",
            res.get("latency", "N/A"),
            constraint_viol
        ])
    
    headers = ["User ID", "Status", "Slice", "Ground Truth", "Intent Match", "CQI", "BW (MHz)", "Rate (Mbps)", "Latency (ms)", "Constraint Viol."]
    summary_table = tabulate(summary_rows, headers=headers, tablefmt="grid")
    print(summary_table)
    
    # Print statistics
    success_count = sum(1 for res in detailed_results if not res.get("allocation_failed", True))
    total_count = len(detailed_results)
    embb_count = sum(1 for res in detailed_results if res.get("slice_type") == "eMBB")
    urllc_count = sum(1 for res in detailed_results if res.get("slice_type") == "URLLC")
    
    print("\nStatistics:")
    print(f"Total users processed: {total_count}")
    print(f"Success rate: {success_count}/{total_count} ({success_count/total_count*100:.1f}%)")
    print(f"eMBB allocations: {embb_count} ({embb_count/total_count*100:.1f}%)")
    print(f"URLLC allocations: {urllc_count} ({urllc_count/total_count*100:.1f}%)")
    print(f"Constraint violations: {constraint_violations}/{total_count} ({constraint_violations/total_count*100:.1f}%)")
    
    # Print intent understanding statistics
    print("\nIntent Understanding Evaluation:")
    print(f"Correctly identified intents: {correct_intents}/{total_evaluated}")
    print(f"Intent understanding rate: {intent_rate:.1f}%")
    
    # Print slice utilization statistics
    print("\nSlice Utilization Statistics:")
    print(f"Average eMBB utilization: {avg_embb_util:.2f}%")
    print(f"Average URLLC utilization: {avg_urllc_util:.2f}%")
    
    # Prepare data for CSV export
    slice_stats = {
        "avg_embb_util": f"{avg_embb_util:.2f}",
        "avg_urllc_util": f"{avg_urllc_util:.2f}",
        "constraint_violations": constraint_violations
    }
    
    intent_stats = {
        "total": total_evaluated,
        "correct": correct_intents,
        "rate": f"{intent_rate:.2f}"
    }
    
    # Export results to CSV
    export_results_to_csv(detailed_results, slice_stats, intent_stats, export_file)

if __name__ == "__main__":
    # Test with 4 users by default and export results to CSV
    main(num_users=30, export_file="prompt_based_slicing_results_north.csv")