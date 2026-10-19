# 2025-03-16 14:38 add a workload balance function
# 2025-03-16 15:22 update the output format to show all users' allocations
# 2025-03-17 09:45 implement CSV-based user testing with for loop
# 2025-03-17 11:30 add results export to CSV
# 2025-03-17 14:15 implement LLM-powered bandwidth allocation
# 2025-03-17 16:40 enhance analytics with resource stats and intent evaluation
# 2025-03-22 18:20 add allocation status to CSV and clean up output
# 2025-03-23 09:15 fix encoding issues for Windows command prompt
# 2025-03-24 15:45 update to prioritize LLM-based intent understanding
# 2025-03-27 10:25 add total transmission rate tracking and average resource utilization
# 2025-03-27 14:30 remove the knowledge base

# Import necessary libraries
from typing import Dict, List, Any, Optional, TypedDict
import json
import time
import numpy as np
import math
import random
from datetime import datetime
import os
import re
from tabulate import tabulate  # For formatted table output
import pandas as pd  # For reading CSV files and exporting results
import csv  # For writing CSV files

# LangGraph and LangChain related
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.tools import BaseTool, tool
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

# Set up LLM
llm = ChatOpenAI(
    api_key="REDACTED_CONFIGURE_AT_RUNTIME",
    base_url="https://api.deepseek.com",
    model="deepseek-chat",
    temperature=0
)

# ====================== CSV Data Loading Function ======================

def load_user_data_from_csv(file_path, num_users=None):
    """Load user data from ray tracing CSV file
    
    Expected CSV format:
    RX_ID,X,Y,Z,SNR_dB,RX_Power_dBm,CQI,LOS,User_Request,Request_Label
    
    Parameters:
    - file_path: Path to CSV file
    - num_users: Optional limit on number of users to load
    
    Returns:
    List of user dictionaries with keys: user_id, location, request, cqi, ground_truth
    """
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
            },
            {
                "user_id": "3",
                "location": "(175.82, 31.38, 1.50)",
                "request": "I need to sync my calendar and contacts",
                "cqi": 8,
                "ground_truth": "URLLC"
            },
            {
                "user_id": "7",
                "location": "(-43.35, -65.84, 1.50)",
                "request": "I need to participate in a video conference meeting",
                "cqi": 14,
                "ground_truth": "eMBB"
            }
        ]

# ====================== CSV Results Export Function ======================

def export_results_to_csv(results, slice_stats, intent_stats, file_path="fileName.csv"):
    """Export test results to a CSV file with enhanced analytics
    
    Parameters:
    - results: List of result dictionaries
    - slice_stats: Dictionary containing slice utilization statistics
    - intent_stats: Dictionary with intent understanding evaluation
    - file_path: Path to save the CSV file
    """
    try:
        # Define CSV headers
        headers = [
            "User ID", "Allocation Status", "Slice", "Ground Truth", "Intent Correct", "CQI", 
            "Bandwidth (MHz)", "Rate (Mbps)", "Latency (ms)", "Adjustments Made", 
            "eMBB Total Rate Before (Mbps)", "eMBB Total Rate After (Mbps)",
            "URLLC Total Rate Before (Mbps)", "URLLC Total Rate After (Mbps)",
            "Avg Resource Util Before (%)", "Avg Resource Util After (%)",
            "Request"
        ]
        
        # Open file for writing
        with open(file_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            
            # Write headers
            writer.writerow(headers)
            
            # Write data rows
            for result in results:
                allocation_status = "Failed" if result.get("allocation_failed", True) else "Success"
                
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
                    "Yes" if result.get("adjustments_made", False) else "No",
                    result.get("embb_total_rate_before", "N/A"),
                    result.get("embb_total_rate_after", "N/A"),
                    result.get("urllc_total_rate_before", "N/A"),
                    result.get("urllc_total_rate_after", "N/A"),
                    result.get("avg_resource_util_before", "N/A"),
                    result.get("avg_resource_util_after", "N/A"),
                    result["request"]
                ]
                writer.writerow(row)
            
            # Add empty row as separator
            writer.writerow([])
            
            # Add summary statistics
            writer.writerow(["SUMMARY STATISTICS"])
            writer.writerow(["Average Resource Utilization (%)", slice_stats["avg_resource_util"]])
            writer.writerow(["Final Resource Utilization (%)", slice_stats["final_resource_util"]])
            writer.writerow(["Final eMBB Total Rate (Mbps)", slice_stats["final_embb_total_rate"]])
            writer.writerow(["Final URLLC Total Rate (Mbps)", slice_stats["final_urllc_total_rate"]])
            
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

# ====================== State Definition ======================

class NetworkState(TypedDict):
    """Simplified network state"""
    user_id: str
    location: str
    request: str
    cqi: int
    history: List[Dict[str, Any]]
    memory: Dict[str, Any]
    step_count: int
    current_step: str
    final_result: Optional[str]

# ====================== CQI-Related Functions ======================

def generate_random_cqi():
    """Generate a random CQI value between 1 and 15"""
    return random.randint(1, 15)

def calculate_rate_from_cqi(bandwidth, cqi):
    """Calculate data rate based on CQI using Shannon's formula
    
    bandwidth: in MHz
    cqi: Channel Quality Indicator (1-15)
    returns: data rate in Mbps
    """
    # Convert CQI to SNR in linear scale (10^(CQI/10))
    snr = 10 ** (cqi / 10)
    
    # Calculate rate using Shannon's formula: bandwidth * log2(1 + SNR)
    # Using log10 as specified in the requirements, multiplied by Shannon coefficient
    rate = bandwidth * math.log10(1 + snr) * 10  # Scaled to match expected rate ranges
    
    # Round to 2 decimal places
    return round(rate, 2)

# ====================== Global Network State ======================

# Persistent state of network slices (global variable, should use database in actual applications)
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

# Store network state before each allocation for comparison
PREVIOUS_NETWORK_STATE = None

def get_current_network_state():
    """Get a copy of the current network state"""
    # Should read from database in actual applications
    return GLOBAL_NETWORK_STATE.copy()

def update_network_state(new_state):
    """Update global network state"""
    global GLOBAL_NETWORK_STATE, PREVIOUS_NETWORK_STATE
    # Store current state before updating
    PREVIOUS_NETWORK_STATE = GLOBAL_NETWORK_STATE.copy()
    # Update state
    GLOBAL_NETWORK_STATE = new_state
    # Should write state to database in actual applications
    return True

def reset_network_state():
    """Reset network state to initial values for fresh testing"""
    global GLOBAL_NETWORK_STATE, PREVIOUS_NETWORK_STATE
    
    # Reset to initial state
    GLOBAL_NETWORK_STATE = INITIAL_NETWORK_STATE.copy()
    
    # Reset timestamp
    GLOBAL_NETWORK_STATE["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    PREVIOUS_NETWORK_STATE = None
    
    return True

def calculate_utilization_rate(usage, capacity):
    """Calculate utilization rate percentage"""
    return f"{(usage / capacity * 100):.2f}%"

def calculate_total_transmission_rates():
    """Calculate total transmission rate for each slice"""
    current_state = get_current_network_state()
    
    # Calculate total rates
    embb_total_rate = sum(user["rate"] for user in current_state["embb_slice"]["users"])
    urllc_total_rate = sum(user["rate"] for user in current_state["urllc_slice"]["users"])
    
    return embb_total_rate, urllc_total_rate

def calculate_average_resource_utilization():
    """Calculate weighted average resource utilization across all slices"""
    current_state = get_current_network_state()
    
    # Get resource usage and capacity
    embb_usage = current_state["embb_slice"]["resource_usage"]
    embb_capacity = current_state["embb_slice"]["total_capacity"]
    urllc_usage = current_state["urllc_slice"]["resource_usage"]
    urllc_capacity = current_state["urllc_slice"]["total_capacity"]
    
    # Calculate total usage and capacity
    total_usage = embb_usage + urllc_usage
    total_capacity = embb_capacity + urllc_capacity
    
    # Calculate average utilization
    avg_util = (total_usage / total_capacity) * 100 if total_capacity > 0 else 0
    return round(avg_util, 2)

# ====================== Network State Reporting Functions ======================

def generate_concise_report(current_state, new_user_id=None, adjustment_result=None):
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
    
    # Calculate total transmission rates
    embb_total_rate, urllc_total_rate = calculate_total_transmission_rates()
    
    # Calculate average resource utilization
    avg_resource_util = calculate_average_resource_utilization()
    
    report = [
        f"Network Status @ {current_state['timestamp']}",
        f"Total Users: {current_state['total_users']}",
        f"Average Resource Utilization: {avg_resource_util}%",
        f"eMBB Total Rate: {embb_total_rate:.2f} Mbps, URLLC Total Rate: {urllc_total_rate:.2f} Mbps",
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
    
    # If dynamic adjustments were made, summarize them
    if adjustment_result and adjustment_result.get("adjustments_made", False):
        report.append("\nDynamic Resource Adjustments:")
        report.append(f"Users adjusted: {len(adjustment_result['user_adjustments'])}, Bandwidth freed: {adjustment_result['freed_bandwidth']} MHz")
        
        # Show brief summary of adjusted users
        if len(adjustment_result['user_adjustments']) > 0:
            adj_summary = []
            for adj in adjustment_result["user_adjustments"]:
                adj_summary.append(f"User {adj['user_id']}: {adj['old_bandwidth']}→{adj['new_bandwidth']} MHz, Rate: {adj['old_rate']:.2f}→{adj['new_rate']:.2f} Mbps")
            report.append("  " + ", ".join(adj_summary))
    
    return "\n".join(report)

def generate_user_allocation_table(current_state, new_user_id=None, adjusted_user_ids=None):
    """Generate a table showing all user allocations"""
    if adjusted_user_ids is None:
        adjusted_user_ids = []
    
    # Combine all users into one list for the table
    all_users = []
    
    # Add eMBB users
    for user in current_state["embb_slice"]["users"]:
        status = ""
        if user["user_id"] == new_user_id:
            status = "NEW"
        elif user["user_id"] in adjusted_user_ids:
            status = "ADJUSTED"
        
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
        status = ""
        if user["user_id"] == new_user_id:
            status = "NEW"
        elif user["user_id"] in adjusted_user_ids:
            status = "ADJUSTED"
        
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

# ====================== Capacity Management Functions ======================

def check_slice_capacity(slice_type, required_bandwidth):
    """Check if the slice has enough capacity for a new allocation
    
    Parameters:
    - slice_type: "eMBB" or "URLLC"
    - required_bandwidth: Bandwidth needed for the new user in MHz
    
    Returns:
    - (has_capacity, available_bandwidth, required_adjustment)
    """
    # Get current network state
    current_state = get_current_network_state()
    
    # Determine slice key
    slice_key = "embb_slice" if slice_type == "eMBB" else "urllc_slice"
    
    # Get slice information
    total_capacity = current_state[slice_key]["total_capacity"]
    current_usage = current_state[slice_key]["resource_usage"]
    available_bandwidth = total_capacity - current_usage
    
    # Check if there's enough capacity
    if available_bandwidth >= required_bandwidth:
        return True, available_bandwidth, 0
    else:
        # Not enough capacity, return how much adjustment is needed
        return False, available_bandwidth, required_bandwidth - available_bandwidth

def find_adjustable_users(slice_type, adjustment_needed):
    """Identify users whose bandwidth can be reduced
    
    Parameters:
    - slice_type: "eMBB" or "URLLC"
    - adjustment_needed: Amount of bandwidth that needs to be freed
    
    Returns:
    - (adjustments_possible, user_adjustments, total_freed_bandwidth)
    - user_adjustments is a list of tuples: (user_id, old_bandwidth, new_bandwidth, old_rate, new_rate)
    """
    # Get current network state
    current_state = get_current_network_state()
    
    # Determine slice key and minimum rate
    slice_key = "embb_slice" if slice_type == "eMBB" else "urllc_slice"
    min_rate = 100 if slice_type == "eMBB" else 1
    
    # Get all users in this slice
    users = current_state[slice_key]["users"]
    
    # Sort users by rate (highest first) to optimize adjustments
    sorted_users = sorted(users, key=lambda u: u["rate"], reverse=True)
    
    # Track adjustments
    user_adjustments = []
    total_freed_bandwidth = 0
    
    # Try to adjust users until we free enough bandwidth
    for user in sorted_users:
        user_id = user["user_id"]
        current_bandwidth = user["bandwidth"]
        cqi = user["cqi"]
        current_rate = user["rate"]
        
        # Calculate minimum bandwidth needed to meet minimum rate
        # Inverting Shannon's formula: min_bandwidth = min_rate / (log10(1 + 10^(CQI/10)) * 10)
        snr = 10 ** (cqi / 10)
        shannon_factor = math.log10(1 + snr) * 10
        min_bandwidth_needed = math.ceil(min_rate / shannon_factor)
        
        # Ensure minimum bandwidth still meets slice constraints
        if slice_type == "eMBB":
            min_bandwidth_needed = max(min_bandwidth_needed, 6)  # eMBB minimum is 6 MHz
        else:
            min_bandwidth_needed = max(min_bandwidth_needed, 1)  # URLLC minimum is 1 MHz
        
        # Calculate how much we can reduce for this user
        max_reduction = current_bandwidth - min_bandwidth_needed
        
        # If we can reduce this user's bandwidth
        if max_reduction > 0:
            # Calculate new bandwidth and resulting rate
            new_bandwidth = current_bandwidth - max_reduction
            new_rate = calculate_rate_from_cqi(new_bandwidth, cqi)
            
            # Add to adjustments list
            user_adjustments.append((user_id, current_bandwidth, new_bandwidth, current_rate, new_rate))
            
            # Update total freed bandwidth
            total_freed_bandwidth += max_reduction
            
            # Check if we've freed enough
            if total_freed_bandwidth >= adjustment_needed:
                # Optimization: If we've freed more than needed, we can give some back
                excess = total_freed_bandwidth - adjustment_needed
                if excess > 0 and len(user_adjustments) > 0:
                    # Give excess back to the last adjusted user
                    user_id, old_bw, new_bw, old_rate, new_rate = user_adjustments[-1]
                    adjusted_new_bw = new_bw + excess
                    adjusted_new_rate = calculate_rate_from_cqi(adjusted_new_bw, cqi)
                    
                    # Update the adjustment entry
                    user_adjustments[-1] = (user_id, old_bw, adjusted_new_bw, old_rate, adjusted_new_rate)
                    total_freed_bandwidth = adjustment_needed  # We've freed exactly what was needed
                
                return True, user_adjustments, total_freed_bandwidth
    
    # If we get here, we couldn't free enough bandwidth
    return False, user_adjustments, total_freed_bandwidth

def apply_bandwidth_adjustments(slice_type, user_adjustments):
    """Apply bandwidth adjustments to existing users
    
    Parameters:
    - slice_type: "eMBB" or "URLLC"
    - user_adjustments: List of (user_id, old_bandwidth, new_bandwidth, old_rate, new_rate)
    
    Returns:
    - Updated network state
    """
    # Get current network state
    current_state = get_current_network_state()
    
    # Determine slice key
    slice_key = "embb_slice" if slice_type == "eMBB" else "urllc_slice"
    
    # Dictionary to map user_id to new values for faster lookup
    adjustment_map = {user_id: (new_bw, new_rate) for user_id, _, new_bw, _, new_rate in user_adjustments}
    
    # Calculate total bandwidth reduction
    total_reduction = sum(old_bw - new_bw for _, old_bw, new_bw, _, _ in user_adjustments)
    
    # Update each user's bandwidth and rate
    for i, user in enumerate(current_state[slice_key]["users"]):
        if user["user_id"] in adjustment_map:
            new_bw, new_rate = adjustment_map[user["user_id"]]
            current_state[slice_key]["users"][i]["bandwidth"] = new_bw
            current_state[slice_key]["users"][i]["rate"] = new_rate
    
    # Update resource usage and utilization rate
    current_state[slice_key]["resource_usage"] -= total_reduction
    current_state[slice_key]["utilization_rate"] = calculate_utilization_rate(
        current_state[slice_key]["resource_usage"],
        current_state[slice_key]["total_capacity"]
    )
    
    # Update timestamp
    current_state["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Update global state
    update_network_state(current_state)
    
    return current_state

# ====================== Workload Balancing Functions ======================

def get_slice_utilization_rates():
    """Get current utilization rates for both slices"""
    current_state = get_current_network_state()
    
    embb_total = current_state["embb_slice"]["total_capacity"]
    embb_used = current_state["embb_slice"]["resource_usage"]
    embb_rate = embb_used / embb_total
    
    urllc_total = current_state["urllc_slice"]["total_capacity"]
    urllc_used = current_state["urllc_slice"]["resource_usage"]
    urllc_rate = urllc_used / urllc_total
    
    return embb_rate, urllc_rate

def check_workload_balance(target_slice_type, cqi, required_bandwidth):
    """Check if workload needs to be balanced between slices
    
    Parameters:
    - target_slice_type: Originally targeted slice type ("eMBB" or "URLLC")
    - cqi: User's Channel Quality Indicator
    - required_bandwidth: Required bandwidth for the user
    
    Returns:
    - (should_rebalance, recommended_slice, reason)
    """
    # Get current utilization rates
    embb_rate, urllc_rate = get_slice_utilization_rates()
    
    # Calculate rate for both slice types
    embb_potential_rate = None
    urllc_potential_rate = None
    
    # Only calculate if the respective slice could potentially support this user
    if target_slice_type == "eMBB" or (cqi >= 6 and required_bandwidth <= 5):
        # See if URLLC could support this user
        if 1 <= required_bandwidth <= 5:
            urllc_potential_rate = calculate_rate_from_cqi(required_bandwidth, cqi)
    
    if target_slice_type == "URLLC" or (required_bandwidth >= 6):
        # See if eMBB could support this user
        if required_bandwidth >= 6:
            embb_potential_rate = calculate_rate_from_cqi(required_bandwidth, cqi)
    
    # Check if we should rebalance based on utilization difference
    utilization_diff = abs(embb_rate - urllc_rate)
    
    # Only consider rebalancing if the difference is significant (>20%)
    if utilization_diff > 0.2:
        # Determine which slice is less utilized
        less_utilized_slice = "eMBB" if embb_rate < urllc_rate else "URLLC"
        
        # Only rebalance if:
        # 1. The target slice is not already the less utilized one
        # 2. The user can be supported on the less utilized slice
        if target_slice_type != less_utilized_slice:
            if less_utilized_slice == "eMBB" and embb_potential_rate is not None:
                # Check if eMBB meets rate requirements
                if 100 <= embb_potential_rate <= 400:
                    return True, "eMBB", f"Rebalancing workload: eMBB utilization ({embb_rate:.2%}) is significantly lower than URLLC ({urllc_rate:.2%})"
            
            elif less_utilized_slice == "URLLC" and urllc_potential_rate is not None:
                # Check if URLLC meets rate requirements
                if 1 <= urllc_potential_rate <= 100:
                    return True, "URLLC", f"Rebalancing workload: URLLC utilization ({urllc_rate:.2%}) is significantly lower than eMBB ({embb_rate:.2%})"
    
    # No rebalancing needed or possible
    return False, target_slice_type, "No workload balancing needed"

# ====================== Bandwidth Allocation Helper Functions ======================

def apply_heuristic_bandwidth(slice_type, request, min_bandwidth, max_bandwidth):
    """Apply heuristic rules to determine bandwidth based on request type"""
    request_lower = request.lower()
    
    if slice_type == "eMBB":
        # For eMBB, use more bandwidth for video/streaming, less for other applications
        if any(keyword in request_lower for keyword in ["video", "stream", "watch", "movie", "4k", "8k"]):
            return min(max_bandwidth, 15)  # High bandwidth for video
        elif any(keyword in request_lower for keyword in ["download", "file", "upload"]):
            return min(max_bandwidth, 12)  # Medium-high for downloads
        elif any(keyword in request_lower for keyword in ["conference", "meeting", "call"]):
            return min(max_bandwidth, 10)  # Medium for video conferencing
        else:
            return min(max_bandwidth, 8)  # Medium for other eMBB applications
    else:  # URLLC
        # For URLLC, use more bandwidth for critical applications
        if any(keyword in request_lower for keyword in ["surgery", "medical", "emergency"]):
            return min(max_bandwidth, 5)  # Max for critical applications
        elif any(keyword in request_lower for keyword in ["control", "automation", "robot"]):
            return min(max_bandwidth, 3)  # Medium for control applications
        else:
            return min(max_bandwidth, 2)  # Minimum for other URLLC applications

def apply_heuristic_latency(slice_type, request, min_latency, max_latency):
    """Apply heuristic rules to determine latency based on request type"""
    request_lower = request.lower()
    
    if slice_type == "eMBB":
        if any(keyword in request_lower for keyword in ["video", "stream", "watch", "movie", "4k", "8k"]):
            return min(max_latency, 50)  # Medium latency for video is fine
        elif any(keyword in request_lower for keyword in ["download", "file", "upload"]):
            return min(max_latency, 80)  # Higher latency for downloads is acceptable
        elif any(keyword in request_lower for keyword in ["conference", "meeting", "call"]):
            return min(max_latency, 30)  # Lower latency for interactive conferencing
        else:
            return min(max_latency, 40)  # Lower latency for interactive eMBB
    else:  # URLLC
        if any(keyword in request_lower for keyword in ["surgery", "medical", "emergency"]):
            return max(min_latency, 1)  # Lowest possible latency for critical applications
        elif any(keyword in request_lower for keyword in ["control", "automation", "robot"]):
            return max(min_latency, 3)  # Very low latency for control
        else:
            return max(min_latency, 5)  # Low latency for other URLLC applications

# ====================== Tool Definitions ======================

@tool
def network_monitor(slice_type: Optional[str] = None) -> Dict[str, Any]:
    """Get network slice status"""
    # Get current network state
    current_state = get_current_network_state()
    
    if slice_type == "eMBB":
        return {"embb_slice": current_state["embb_slice"]}
    elif slice_type == "URLLC":
        return {"urllc_slice": current_state["urllc_slice"]}
    else:
        return current_state

@tool
def check_and_adjust_capacity(slice_type: str, required_bandwidth: int) -> Dict[str, Any]:
    """Check if slice has capacity for new user and perform dynamic adjustment if needed
    
    Parameters:
    - slice_type: "eMBB" or "URLLC"
    - required_bandwidth: Bandwidth needed for the new user in MHz
    
    Returns:
    - Dictionary with capacity status and adjustment results
    """
    # Check if we have enough capacity
    has_capacity, available_bandwidth, adjustment_needed = check_slice_capacity(
        slice_type, required_bandwidth
    )
    
    # If we have enough capacity, no adjustment needed
    if has_capacity:
        return {
            "status": "success",
            "has_capacity": True,
            "message": f"Sufficient capacity available: {available_bandwidth} MHz",
            "adjustments_made": False,
            "user_adjustments": []
        }
    
    # Not enough capacity, try to adjust existing users
    adjustments_possible, user_adjustments, freed_bandwidth = find_adjustable_users(
        slice_type, adjustment_needed
    )
    
    # If adjustments are not possible
    if not adjustments_possible:
        return {
            "status": "failure",
            "has_capacity": False,
            "message": f"Insufficient capacity. Need {required_bandwidth} MHz but only {available_bandwidth} MHz available. Could only free {freed_bandwidth} MHz through adjustments.",
            "adjustments_made": False,
            "user_adjustments": []
        }
    
    # Apply the adjustments
    updated_state = apply_bandwidth_adjustments(slice_type, user_adjustments)
    
    # Prepare user-friendly adjustment summary
    adjustment_summary = []
    for user_id, old_bw, new_bw, old_rate, new_rate in user_adjustments:
        adjustment_summary.append({
            "user_id": user_id,
            "old_bandwidth": old_bw,
            "new_bandwidth": new_bw,
            "bandwidth_reduction": old_bw - new_bw,
            "old_rate": old_rate,
            "new_rate": new_rate
        })
    
    # Return success result
    return {
        "status": "success",
        "has_capacity": True,
        "message": f"Dynamically adjusted {len(user_adjustments)} user(s) to free {freed_bandwidth} MHz",
        "adjustments_made": True,
        "user_adjustments": adjustment_summary,
        "freed_bandwidth": freed_bandwidth
    }

@tool
def workload_balance_tool(target_slice_type: str, cqi: int, required_bandwidth: int) -> Dict[str, Any]:
    """Check if workload should be balanced between slices
    
    Parameters:
    - target_slice_type: Originally targeted slice type ("eMBB" or "URLLC")
    - cqi: User's Channel Quality Indicator
    - required_bandwidth: Required bandwidth for the user
    
    Returns:
    - Dictionary with balancing recommendation
    """
    should_rebalance, recommended_slice, reason = check_workload_balance(
        target_slice_type, cqi, required_bandwidth
    )
    
    # Get utilization rates for reporting
    embb_rate, urllc_rate = get_slice_utilization_rates()
    
    return {
        "should_rebalance": should_rebalance,
        "original_slice": target_slice_type,
        "recommended_slice": recommended_slice,
        "reason": reason,
        "current_utilization": {
            "embb": f"{embb_rate:.2%}",
            "urllc": f"{urllc_rate:.2%}"
        }
    }

@tool
def beamforming_tool(user_id: str, slice_type: str, cqi: int, request: str) -> Dict[str, Any]:
    """Execute beamforming algorithm using CQI and request analysis
    
    Parameters:
    - user_id: User identifier
    - slice_type: Either "eMBB" or "URLLC"
    - cqi: Channel Quality Indicator (1-15)
    - request: User's request text
    
    Returns:
    - Dictionary containing allocated resources
    """
    # Get current network state to check available resources
    current_state = get_current_network_state()
    slice_key = "embb_slice" if slice_type == "eMBB" else "urllc_slice"
    available_bandwidth = current_state[slice_key]["total_capacity"] - current_state[slice_key]["resource_usage"]
    
    # Define constraints based on slice type
    if slice_type == "eMBB":
        min_bandwidth = 6
        max_bandwidth = min(20, available_bandwidth)  # Cap at available or 20, whichever is smaller
        min_latency = 10
        max_latency = 100
    else:  # URLLC
        min_bandwidth = 1
        max_bandwidth = min(5, available_bandwidth)  # Cap at available or 5, whichever is smaller
        min_latency = 1
        max_latency = 10
    
    # Ensure max_bandwidth is at least min_bandwidth (even if we're exceeding available)
    # This will get corrected during capacity check later
    max_bandwidth = max(max_bandwidth, min_bandwidth)
    
    # Create a prompt for the LLM to analyze the request and recommend a bandwidth
    bandwidth_prompt = f"""
Based on the following user request and network conditions, recommend an appropriate bandwidth allocation:

User Request: "{request}"
Slice Type: {slice_type}
CQI (Channel Quality Indicator): {cqi} (higher means better signal quality)
Available Bandwidth: {available_bandwidth} MHz

Requirements:
- For eMBB slice: Bandwidth must be between {min_bandwidth}-{max_bandwidth} MHz (integer values only)
- For URLLC slice: Bandwidth must be between {min_bandwidth}-{max_bandwidth} MHz (integer values only)

Consider:
1. The type of application implied by the request (video, download, control, etc.)
2. The resource requirements for that application
3. The available bandwidth in the network
4. The signal quality (CQI)

Please respond with a single integer number representing your recommended bandwidth in MHz.
"""
    
    try:
        # Call LLM to analyze and recommend bandwidth
        messages = [SystemMessage(content="You are a network resource allocation expert."), 
                   HumanMessage(content=bandwidth_prompt)]
        response = llm.invoke(messages)
        
        # Extract the bandwidth recommendation from the response
        # Look for an integer in the response
        bandwidth_match = re.search(r'\b(\d+)\b', response.content)
        if bandwidth_match:
            recommended_bandwidth = int(bandwidth_match.group(1))
            
            # Validate the recommendation is within allowed range
            if min_bandwidth <= recommended_bandwidth <= max_bandwidth:
                allocated_bandwidth = recommended_bandwidth
            else:
                # If outside range, cap it
                allocated_bandwidth = max(min_bandwidth, min(recommended_bandwidth, max_bandwidth))
        else:
            # Fallback if no number found
            allocated_bandwidth = apply_heuristic_bandwidth(slice_type, request, min_bandwidth, max_bandwidth)
    except Exception as e:
        print(f"LLM bandwidth recommendation failed: {e}")
        # Fallback to heuristic
        allocated_bandwidth = apply_heuristic_bandwidth(slice_type, request, min_bandwidth, max_bandwidth)
    
    # Determine latency based on application type
    allocated_latency = apply_heuristic_latency(slice_type, request, min_latency, max_latency)
    
    # Calculate rate based on CQI and bandwidth using Shannon's formula
    allocated_rate = calculate_rate_from_cqi(allocated_bandwidth, cqi)
    
    # Validate if rate meets the requirements
    rate_valid = False
    adjustment_made = False
    adjustment_reason = ""
    
    if slice_type == "eMBB":
        if allocated_rate < 100:
            # Rate too low for eMBB, adjust bandwidth
            old_bandwidth = allocated_bandwidth
            allocated_bandwidth = max(allocated_bandwidth + 2, 20)
            allocated_rate = calculate_rate_from_cqi(allocated_bandwidth, cqi)
            adjustment_made = True
            adjustment_reason = f"Initial rate ({allocated_rate:.2f} Mbps) below eMBB minimum (100 Mbps). Increased bandwidth from {old_bandwidth} to {allocated_bandwidth} MHz."
        elif allocated_rate > 400:
            # Rate too high for eMBB, adjust bandwidth
            old_bandwidth = allocated_bandwidth
            allocated_bandwidth = max(6, allocated_bandwidth - 2)
            allocated_rate = calculate_rate_from_cqi(allocated_bandwidth, cqi)
            adjustment_made = True
            adjustment_reason = f"Initial rate above eMBB maximum (400 Mbps). Decreased bandwidth from {old_bandwidth} to {allocated_bandwidth} MHz."
        rate_valid = 100 <= allocated_rate <= 400
    else:  # URLLC
        if allocated_rate > 100:
            # Rate too high for URLLC, adjust bandwidth
            old_bandwidth = allocated_bandwidth
            allocated_bandwidth = max(1, allocated_bandwidth - 1)
            allocated_rate = calculate_rate_from_cqi(allocated_bandwidth, cqi)
            adjustment_made = True
            adjustment_reason = f"Initial rate above URLLC maximum (100 Mbps). Decreased bandwidth from {old_bandwidth} to {allocated_bandwidth} MHz."
        rate_valid = 1 <= allocated_rate <= 100
    
    # Final validity check
    if slice_type == "eMBB":
        rate_valid = 100 <= allocated_rate <= 400
    else:  # URLLC
        rate_valid = 1 <= allocated_rate <= 100
    
    # Check if we have capacity for this bandwidth allocation
    has_capacity, available_bandwidth, adjustment_needed = check_slice_capacity(
        slice_type, allocated_bandwidth
    )
    
    return {
        "status": "success" if rate_valid else "warning",
        "user_id": user_id,
        "slice_type": slice_type,
        "cqi": cqi,
        "allocated_bandwidth": allocated_bandwidth,
        "allocated_rate": allocated_rate,
        "allocated_latency": allocated_latency,
        "rate_valid": rate_valid,
        "adjustment_made": adjustment_made,
        "adjustment_reason": adjustment_reason,
        "rate_requirements": "100-400 Mbps" if slice_type == "eMBB" else "1-100 Mbps",
        "has_capacity": has_capacity,
        "available_bandwidth": available_bandwidth,
        "adjustment_needed": 0 if has_capacity else adjustment_needed
    }

@tool
def slice_allocation(user_id: str, slice_type: str, rate: float, latency: float, cqi: int, bandwidth: int) -> Dict[str, Any]:
    """Allocate user to slice and update network state
    
    Parameters:
    - user_id: User identifier
    - slice_type: Either "eMBB" or "URLLC"
    - rate: Data rate in Mbps
    - latency: Latency in ms
    - cqi: Channel Quality Indicator (1-15)
    - bandwidth: Allocated bandwidth in MHz
    """
    # Get current network state
    current_state = get_current_network_state()
    
    # Create new user with CQI and bandwidth information
    new_user = {
        "user_id": user_id,
        "rate": rate,
        "latency": latency,
        "cqi": cqi,
        "bandwidth": bandwidth
    }
    
    # Update corresponding slice
    slice_key = "embb_slice" if slice_type == "eMBB" else "urllc_slice"
    
    # Add user
    current_state[slice_key]["users"].append(new_user)
    
    # Update resource usage
    current_state[slice_key]["resource_usage"] += bandwidth
    
    # Update utilization rate
    current_state[slice_key]["utilization_rate"] = calculate_utilization_rate(
        current_state[slice_key]["resource_usage"],
        current_state[slice_key]["total_capacity"]
    )
    
    # Update total user count
    current_state["total_users"] = len(current_state["embb_slice"]["users"]) + len(current_state["urllc_slice"]["users"])
    
    # Update timestamp
    current_state["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Update global state
    update_network_state(current_state)
    
    # Return allocation result
    return {
        "status": "success",
        "message": f"User {user_id} has been allocated to {slice_type} slice",
        "allocation": {
            "user_id": user_id,
            "slice_type": slice_type,
            "rate": rate,
            "latency": latency,
            "cqi": cqi,
            "bandwidth": bandwidth
        },
        "updated_slice": {
            "users_count": len(current_state[slice_key]["users"]),
            "resource_usage": current_state[slice_key]["resource_usage"],
            "utilization_rate": current_state[slice_key]["utilization_rate"]
        }
    }

# Tool list
tools = [
    network_monitor, 
    check_and_adjust_capacity, 
    workload_balance_tool, 
    beamforming_tool, 
    slice_allocation
]

# System prompt
SYSTEM_PROMPT = """You are a 5G network slicing management expert, responsible for allocating network resources for users.

The current network has two types of slices:
- eMBB (Enhanced Mobile Broadband): Suitable for high bandwidth applications such as video streaming
  - Bandwidth: 6-20 MHz (integer values only)
  - Data rate: 100-400 Mbps
  - Latency: 10-100ms
  - Total capacity: 90 MHz
  
- URLLC (Ultra-Reliable Low Latency): Suitable for low latency applications such as remote control
  - Bandwidth: 1-5 MHz (integer values only)
  - Data rate: 1-100 Mbps
  - Latency: 1-10ms
  - Total capacity: 30 MHz

Please consider the user's Channel Quality Indicator (CQI) which ranges from 1-15, indicating the signal quality. Higher CQI values allow for higher data rates with the same bandwidth.

When a slice reaches capacity, you can dynamically adjust bandwidth for existing users to accommodate new users, prioritizing users with highest rates for adjustment while ensuring all users still meet minimum requirements.

The data rate is calculated using Shannon's formula: rate = bandwidth * log10(1 + 10^(CQI/10)) * 10

When deciding on the slice type, also consider network load balancing - if one slice is significantly more utilized than another (>20% difference) and the user can be accommodated in either slice, prefer the less utilized slice.

Please clearly state in your answer: "I recommend using [eMBB/URLLC] slice, because...", and provide detailed reasons."""

# ====================== Workflow Nodes ======================

def initialize(state: NetworkState) -> NetworkState:
    """Initialize state - Step 1: Receive user request and CQI"""
    state["history"] = []
    state["memory"] = {}
    state["step_count"] = 0
    state["current_step"] = "understand_intent"
    
    # Add system message
    state["history"].append({"role": "system", "content": SYSTEM_PROMPT})
    
    # Add user request
    user_request = f"""
New User ID: {state["user_id"]}
Location: {state["location"]}
Request: "{state["request"]}"
Channel Quality Indicator (CQI): {state["cqi"]}

Please analyze this request to understand the user's intent and network requirements.
"""
    state["history"].append({"role": "user", "content": user_request})
    
    # Get current network state
    network_state = network_monitor.invoke({})
    state["memory"]["network_state"] = network_state
    
    return state

def understand_intent(state: NetworkState) -> NetworkState:
    """Step 1: Understand user intent by analyzing request"""
    state["step_count"] += 1
    
    # Enhanced intent analysis prompt with comprehensive guidance, removed CQI information
    intent_prompt = f"""
Based on the user request: "{state["request"]}"

Your task is to accurately classify this request as requiring either eMBB or URLLC network slice based ONLY on the application requirements.

CLASSIFICATION GUIDELINES:
1. eMBB (Enhanced Mobile Broadband) - Choose when the application primarily needs:
   - High bandwidth/data rate
   - Can tolerate moderate latency (10-100ms)
   - Example applications: video streaming, large file downloads/uploads, AR/VR, high-resolution video calls

2. URLLC (Ultra-Reliable Low-Latency Communications) - Choose when the application primarily needs:
   - Very low latency (<10ms)
   - High reliability
   - Example applications: remote control, autonomous driving, industrial automation, real-time monitoring, IoT sensors

KEY INDICATORS FOR SLICES:
- Words indicating eMBB: stream, download, upload, video, HD, 4K, 8K, movie, watch, gaming, browse, surfing
- Words indicating URLLC: control, real-time, monitor, automation, sensors, immediate, mission-critical, safety, emergency

DETAILED ANALYSIS REQUIRED:
1. Application type: Identify specific application and its primary purpose
2. Network requirements: Bandwidth needs, latency sensitivity, reliability requirements
3. Slice recommendation: Clear statement about which slice type is most appropriate

EXAMPLES OF GOOD INTENT ANALYSIS:

Example 1 - eMBB:
User request: "I need to stream 4K video"
Analysis:
1. Application type: Video streaming at 4K resolution - an entertainment application requiring sustained high data rates
2. Network requirements: High bandwidth (15+ MHz), moderate latency tolerance (30-50ms), continuous data flow
3. Slice recommendation: eMBB is clearly most appropriate due to high bandwidth requirements for sustained 4K video streaming

Example 2 - URLLC:
User request: "I need to control a remote surgical robot"
Analysis:
1. Application type: Remote medical operation requiring precision control with zero delay
2. Network requirements: Ultra-low latency (<5ms), highest reliability, moderate bandwidth
3. Slice recommendation: URLLC is essential due to life-critical nature requiring minimal delay and maximum reliability

Example 3 - eMBB:
User request: "I need to download large software updates"
Analysis:
1. Application type: Large file download, a data-intensive batch transfer operation
2. Network requirements: High bandwidth (10+ MHz), high tolerance for latency (50-100ms)
3. Slice recommendation: eMBB is appropriate since bandwidth is the primary requirement, not latency

Example 4 - URLLC:
User request: "I need to monitor my heart rate in real-time"
Analysis:
1. Application type: Real-time health monitoring requiring immediate data transmission
2. Network requirements: Low latency (5-10ms), high reliability, low bandwidth
3. Slice recommendation: URLLC is appropriate due to the real-time nature and potential health implications

Example 5 - Borderline case:
User request: "I need to join a video conference meeting"
Analysis:
1. Application type: Interactive video conferencing with two-way communication
2. Network requirements: Moderate bandwidth (5-10 MHz), relatively low latency (20-50ms)
3. Slice recommendation: eMBB is more appropriate overall since bandwidth for video quality typically outweighs the latency requirements for this use case

Please provide a similarly detailed analysis with a clear slice recommendation for the given request.
"""
    state["history"].append({"role": "user", "content": intent_prompt})
    
    # Create message list
    messages = []
    for msg in state["history"]:
        if msg["role"] == "system":
            messages.append(SystemMessage(content=msg["content"]))
        elif msg["role"] == "user":
            messages.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            messages.append(AIMessage(content=msg["content"]))
    
    # Call LLM to analyze intent
    response = llm.invoke(messages)
    analysis = response.content
    
    # Record response
    state["history"].append({"role": "assistant", "content": analysis})
    
    # Determine next step
    state["current_step"] = "allocate_slice_type"
    
    return state

def allocate_slice_type(state: NetworkState) -> NetworkState:
    """Step 2: Allocate appropriate slice type based on intent"""
    state["step_count"] += 1
    
    # Enhanced prompt for slice allocation, decoupled from CQI
    slice_prompt = """
Based on your intent analysis, I need your EXPLICIT recommendation for the appropriate network slice.

YOUR TASK:
Make a definitive choice between eMBB and URLLC for this request, with clear reasoning.

NETWORK SLICE CHARACTERISTICS:
- eMBB (Enhanced Mobile Broadband):
  * For high bandwidth applications
  * Data rate: 100-400 Mbps
  * Bandwidth: 6-20 MHz
  * Latency: 10-100ms
  * Best for: video streaming, downloads, gaming, web browsing

- URLLC (Ultra-Reliable Low-Latency):
  * For time-sensitive applications
  * Data rate: 1-100 Mbps 
  * Bandwidth: 1-5 MHz
  * Latency: 1-10ms
  * Best for: remote control, IoT, automation, real-time monitoring

DETAILED RECOMMENDATION REQUIRED:
Your response MUST begin with the exact phrase: "I recommend using [eMBB/URLLC] slice, because..."

Follow this with 2-3 sentences explaining:
1. Why this slice type is appropriate for the specific application
2. The key requirements that led to this decision (bandwidth vs. latency needs)

EXAMPLES OF ACCEPTABLE RESPONSES:

Example 1:
I recommend using eMBB slice, because video streaming requires sustained high bandwidth to maintain video quality, especially for 4K content. The bandwidth requirement of 15+ MHz exceeds URLLC capabilities, and video streaming can tolerate moderate latency without affecting user experience.

Example 2:
I recommend using URLLC slice, because remote surgical operation demands ultra-low latency for precise control and immediate feedback. The critical nature of this application makes latency the primary concern (requiring <5ms), which only URLLC can provide reliably.

Please provide your recommendation in this exact format for the user's request.
"""
    state["history"].append({"role": "user", "content": slice_prompt})
    
    # Create message list
    messages = []
    for msg in state["history"]:
        if msg["role"] == "system":
            messages.append(SystemMessage(content=msg["content"]))
        elif msg["role"] == "user":
            messages.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            messages.append(AIMessage(content=msg["content"]))
    
    # Call LLM to determine slice type
    response = llm.invoke(messages)
    decision = response.content
    
    # Record response
    state["history"].append({"role": "assistant", "content": decision})
    
    # Extract recommended slice type from LLM response
    slice_match = re.search(r"I recommend using (eMBB|URLLC) slice", decision)
    
    if slice_match:
        llm_recommended_slice = slice_match.group(1)
        # Store the LLM recommendation as the final decision
        state["memory"]["final_slice"] = llm_recommended_slice
    else:
        # If no explicit recommendation is found, determine based on keywords in the response
        if "bandwidth" in decision.lower() and not "latency" in decision.lower():
            state["memory"]["final_slice"] = "eMBB"
        elif "latency" in decision.lower() and not "bandwidth" in decision.lower():
            state["memory"]["final_slice"] = "URLLC"
        elif "video" in decision.lower() or "download" in decision.lower():
            state["memory"]["final_slice"] = "eMBB"
        elif "control" in decision.lower() or "real-time" in decision.lower():
            state["memory"]["final_slice"] = "URLLC"
        else:
            # Default to eMBB if we can't determine
            state["memory"]["final_slice"] = "eMBB"
        print(f"WARNING: LLM didn't provide explicit slice recommendation. Using extracted recommendation: {state['memory']['final_slice']}")
    
    # Ensure we have a final slice determination
    final_slice = state["memory"]["final_slice"]
    
    # Check if workload balancing is needed
    # Estimate bandwidth based on slice type
    estimated_bandwidth = 10 if final_slice == "eMBB" else 3
    
    # Check workload balance
    balance_result = workload_balance_tool.invoke({
        "target_slice_type": final_slice,
        "cqi": state["cqi"],
        "required_bandwidth": estimated_bandwidth
    })
    
    # Record the balance result
    state["memory"]["balance_result"] = balance_result
    
    # Apply workload balancing recommendation if needed
    if balance_result["should_rebalance"]:
        state["memory"]["final_slice"] = balance_result["recommended_slice"]
        state["memory"]["balance_applied"] = True
        
        # Add explanation to history
        balance_prompt = f"""
Workload balancing recommendation:
- Original slice type: {balance_result["original_slice"]}
- Recommended slice: {balance_result["recommended_slice"]}
- Reason: {balance_result["reason"]}
- Current utilization: eMBB {balance_result["current_utilization"]["embb"]}, URLLC {balance_result["current_utilization"]["urllc"]}

Based on workload balancing considerations, we will use {balance_result["recommended_slice"]} slice instead of {balance_result["original_slice"]}.
"""
        state["history"].append({"role": "user", "content": balance_prompt})
        
        # Get LLM response to workload balancing
        messages = []
        for msg in state["history"]:
            if msg["role"] == "system":
                messages.append(SystemMessage(content=msg["content"]))
            elif msg["role"] == "user":
                messages.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                messages.append(AIMessage(content=msg["content"]))
        
        response = llm.invoke(messages)
        balance_decision = response.content
        
        # Record response
        state["history"].append({"role": "assistant", "content": balance_decision})
    else:
        state["memory"]["balance_applied"] = False
    
    # Determine next step
    state["current_step"] = "allocate_resources"
    
    return state

def allocate_resources(state: NetworkState) -> NetworkState:
    """Step 3: Allocate bandwidth and calculate transmission rate"""
    state["step_count"] += 1
    
    # Get final slice type
    final_slice = state["memory"]["final_slice"]
    cqi = state["cqi"]
    
    # Execute beamforming to allocate resources with user's request
    beamforming_result = beamforming_tool.invoke({
        "user_id": state["user_id"], 
        "slice_type": final_slice, 
        "cqi": cqi,
        "request": state["request"]
    })
    
    # Record results
    state["memory"]["beamforming_result"] = beamforming_result
    
    # Now that we have actual bandwidth, check workload balance again if it wasn't applied before
    if not state["memory"].get("balance_applied", False):
        # Check workload balance with actual bandwidth
        balance_result = workload_balance_tool.invoke({
            "target_slice_type": final_slice,
            "cqi": cqi,
            "required_bandwidth": beamforming_result["allocated_bandwidth"]
        })
        
        # Update the balance result
        state["memory"]["balance_result"] = balance_result
        
        # Apply workload balancing recommendation if needed
        if balance_result["should_rebalance"]:
            # Store original allocation before switching
            state["memory"]["original_slice"] = final_slice
            state["memory"]["original_beamforming"] = beamforming_result
            
            # Update to new slice type
            final_slice = balance_result["recommended_slice"]
            state["memory"]["final_slice"] = final_slice
            state["memory"]["balance_applied"] = True
            
            # Execute beamforming again with new slice type and user's request
            beamforming_result = beamforming_tool.invoke({
                "user_id": state["user_id"], 
                "slice_type": final_slice, 
                "cqi": cqi,
                "request": state["request"]
            })
            
            # Update results
            state["memory"]["beamforming_result"] = beamforming_result
            
            # Add explanation to history
            balance_prompt = f"""
Workload balancing recommendation (after beamforming):
- Original slice type: {balance_result["original_slice"]}
- Recommended slice: {balance_result["recommended_slice"]}
- Reason: {balance_result["reason"]}
- Current utilization: eMBB {balance_result["current_utilization"]["embb"]}, URLLC {balance_result["current_utilization"]["urllc"]}

Based on workload balancing considerations and actual bandwidth requirements, we will use {balance_result["recommended_slice"]} slice instead of {balance_result["original_slice"]}.
"""
            state["history"].append({"role": "user", "content": balance_prompt})
            
            # Get LLM response to workload balancing
            messages = []
            for msg in state["history"]:
                if msg["role"] == "system":
                    messages.append(SystemMessage(content=msg["content"]))
                elif msg["role"] == "user":
                    messages.append(HumanMessage(content=msg["content"]))
                elif msg["role"] == "assistant":
                    messages.append(AIMessage(content=msg["content"]))
            
            response = llm.invoke(messages)
            balance_decision = response.content
            
            # Record response
            state["history"].append({"role": "assistant", "content": balance_decision})
    
    # Check if we have capacity or need adjustment
    has_capacity = beamforming_result["has_capacity"]
    required_bandwidth = beamforming_result["allocated_bandwidth"]
    
    if not has_capacity:
        # Try to dynamically adjust capacity
        adjustment_result = check_and_adjust_capacity.invoke({
            "slice_type": final_slice,
            "required_bandwidth": required_bandwidth
        })
        
        # Record adjustment results
        state["memory"]["adjustment_result"] = adjustment_result
        
        # Check if adjustment was successful
        if adjustment_result["has_capacity"]:
            adjustment_message = f"""
Dynamic resource adjustment was successful:
- Freed bandwidth: {adjustment_result['freed_bandwidth']} MHz
- Users adjusted: {len(adjustment_result['user_adjustments'])}
"""
        else:
            adjustment_message = f"""
Dynamic resource adjustment was NOT successful:
- Insufficient capacity in {final_slice} slice
- Available bandwidth: {beamforming_result['available_bandwidth']} MHz
- Required bandwidth: {required_bandwidth} MHz
- Adjustment needed: {beamforming_result['adjustment_needed']} MHz

This means we cannot accommodate this user at this time.
"""
        
        # Add adjustment details
        if adjustment_result["adjustments_made"]:
            adjustment_details = "\nAdjustment details:\n"
            for adj in adjustment_result["user_adjustments"]:
                adjustment_details += f"- User {adj['user_id']}: {adj['old_bandwidth']} MHz → {adj['new_bandwidth']} MHz, Rate: {adj['old_rate']:.2f} Mbps → {adj['new_rate']:.2f} Mbps\n"
            adjustment_message += adjustment_details
    else:
        adjustment_message = "Sufficient slice capacity available. No adjustments needed."
        state["memory"]["adjustment_result"] = {
            "has_capacity": True,
            "adjustments_made": False,
            "user_adjustments": []
        }
    
    # Create resource allocation summary
    resource_prompt = f"""
Resource allocation calculation results:

- User ID: {state["user_id"]}
- Slice Type: {final_slice}
- CQI: {cqi}
- Allocated Bandwidth: {beamforming_result["allocated_bandwidth"]} MHz
- Calculated Data Rate: {beamforming_result["allocated_rate"]:.2f} Mbps
- Allocated Latency: {beamforming_result["allocated_latency"]} ms

Rate requirements check:
- Required rate range: {beamforming_result["rate_requirements"]}
- Rate meets requirements: {'Yes' if beamforming_result["rate_valid"] else 'No'}
{beamforming_result["adjustment_reason"] if beamforming_result["adjustment_made"] else ""}

Capacity check:
{adjustment_message}

Please review these resource allocations and confirm if they are appropriate for the user's needs.
"""
    state["history"].append({"role": "user", "content": resource_prompt})
    
    # Create message list
    messages = []
    for msg in state["history"]:
        if msg["role"] == "system":
            messages.append(SystemMessage(content=msg["content"]))
        elif msg["role"] == "user":
            messages.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            messages.append(AIMessage(content=msg["content"]))
    
    # Call LLM to review resource allocation
    response = llm.invoke(messages)
    review = response.content
    
    # Record response
    state["history"].append({"role": "assistant", "content": review})
    
    # Check if we can proceed with allocation
    can_allocate = state["memory"]["adjustment_result"]["has_capacity"]
    
    if can_allocate:
        # Determine next step
        state["current_step"] = "evaluate_network"
    else:
        # Skip allocation and go straight to evaluation (which will report failure)
        state["memory"]["allocation_failed"] = True
        state["current_step"] = "evaluate_network"
    
    return state

def evaluate_network(state: NetworkState) -> NetworkState:
    """Step 4: Evaluate network state and check rate requirements"""
    state["step_count"] += 1
    
    # Check if allocation failed due to capacity constraints
    allocation_failed = state["memory"].get("allocation_failed", False)
    
    if allocation_failed:
        # Create failure message
        network_prompt = f"""
ALLOCATION FAILED: Unable to accommodate user {state["user_id"]}

Reason: Insufficient capacity in {state["memory"]["final_slice"]} slice even after dynamic adjustment attempts.

Please provide a final summary explaining why this user couldn't be accommodated and what options might be available:
1. Try again later when resources are freed
2. Consider using a different application with lower resource requirements
3. Consider moving to a different location with better signal quality
"""
        state["history"].append({"role": "user", "content": network_prompt})
        
        # Create message list
        messages = []
        for msg in state["history"]:
            if msg["role"] == "system":
                messages.append(SystemMessage(content=msg["content"]))
            elif msg["role"] == "user":
                messages.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                messages.append(AIMessage(content=msg["content"]))
        
        # Call LLM to generate failure evaluation
        response = llm.invoke(messages)
        evaluation = response.content
        
        # Record final result
        state["final_result"] = evaluation
        
        # Print concise failure report
        print("\n" + "-"*40)
        print(f"ALLOCATION FAILED FOR USER {state['user_id']}")
        print("-"*40)
        print(f"Request: {state['request']}")
        print(f"Slice type: {state['memory']['final_slice']}")
        print(f"Reason: Insufficient capacity even after attempted adjustments")
        
        return state
    
    # Get necessary information for successful allocation
    beamforming_result = state["memory"]["beamforming_result"]
    final_slice = state["memory"]["final_slice"]
    adjustment_result = state["memory"]["adjustment_result"]
    
    # Execute resource allocation
    allocation_result = slice_allocation.invoke({
        "user_id": state["user_id"],
        "slice_type": final_slice,
        "rate": beamforming_result["allocated_rate"],
        "latency": beamforming_result["allocated_latency"],
        "cqi": state["cqi"],
        "bandwidth": beamforming_result["allocated_bandwidth"]
    })
    
    # Record results
    state["memory"]["allocation_result"] = allocation_result
    
    # Get updated network state
    updated_network_state = network_monitor.invoke({})
    state["memory"]["updated_network_state"] = updated_network_state
    
    # Get list of adjusted user IDs
    adjusted_user_ids = []
    if adjustment_result["adjustments_made"]:
        adjusted_user_ids = [adj["user_id"] for adj in adjustment_result["user_adjustments"]]
    
    # Print concise report and user allocation table
    print("\n" + "-"*40)
    print(f"ALLOCATION RESULT FOR USER {state['user_id']}")
    print("-"*40)
    concise_report = generate_concise_report(
        get_current_network_state(), 
        state["user_id"], 
        adjustment_result
    )
    print(concise_report)
    
    # Print complete user allocation table
    user_table = generate_user_allocation_table(
        get_current_network_state(),
        state["user_id"], 
        adjusted_user_ids
    )
    print(user_table)
    
    # Create dynamic adjustment summary if adjustments were made
    adjustment_summary = ""
    if adjustment_result["adjustments_made"]:
        adjustment_summary = "\n## Dynamic Resource Adjustment\n"
        adjustment_summary += f"- {len(adjustment_result['user_adjustments'])} users had their bandwidth reduced to accommodate this new user\n"
        adjustment_summary += f"- Total bandwidth freed: {adjustment_result['freed_bandwidth']} MHz\n\n"
        adjustment_summary += "User adjustments:\n"
        
        for adj in adjustment_result["user_adjustments"]:
            adjustment_summary += f"- User {adj['user_id']}: {adj['old_bandwidth']} MHz → {adj['new_bandwidth']} MHz ({adj['bandwidth_reduction']} MHz reduction)\n"
            adjustment_summary += f"  Rate: {adj['old_rate']:.2f} Mbps → {adj['new_rate']:.2f} Mbps (still meets requirements)\n"
    
    # Add workload balance info if applied
    balance_summary = ""
    if state["memory"].get("balance_applied", False):
        balance_result = state["memory"]["balance_result"]
        balance_summary = "\n## Workload Balancing Applied\n"
        balance_summary += f"- Original slice recommendation: {balance_result['original_slice']}\n"
        balance_summary += f"- Final slice allocation: {balance_result['recommended_slice']}\n"
        balance_summary += f"- Reason: {balance_result['reason']}\n"
        balance_summary += f"- Utilization before allocation: eMBB {balance_result['current_utilization']['embb']}, URLLC {balance_result['current_utilization']['urllc']}\n"
    
    # Calculate total transmission rates
    embb_total_rate, urllc_total_rate = calculate_total_transmission_rates()
    avg_resource_util = calculate_average_resource_utilization()
    
    # Create network evaluation prompt
    network_prompt = f"""
Network state after allocating resources to user {state["user_id"]}:

eMBB slice:
- Users: {len(updated_network_state['embb_slice']['users'])}
- Resource usage: {updated_network_state['embb_slice']['resource_usage']} MHz
- Utilization rate: {updated_network_state['embb_slice']['utilization_rate']}
- Total Transmission Rate: {embb_total_rate:.2f} Mbps

URLLC slice:
- Users: {len(updated_network_state['urllc_slice']['users'])}
- Resource usage: {updated_network_state['urllc_slice']['resource_usage']} MHz
- Utilization rate: {updated_network_state['urllc_slice']['utilization_rate']}
- Total Transmission Rate: {urllc_total_rate:.2f} Mbps

Overall Resource Utilization: {avg_resource_util:.2f}%

User allocation details:
- Allocated to: {final_slice} slice
- Bandwidth: {beamforming_result['allocated_bandwidth']} MHz
- Data rate: {beamforming_result['allocated_rate']:.2f} Mbps (Required: {beamforming_result['rate_requirements']})
- Latency: {beamforming_result['allocated_latency']} ms
{adjustment_summary}
{balance_summary}
Please provide a final summary of the resource allocation, evaluating whether the network adequately supports this user's requirements.
Include details about any dynamic adjustments or workload balancing made to accommodate this user.
"""
    state["history"].append({"role": "user", "content": network_prompt})
    
    # Create message list
    messages = []
    for msg in state["history"]:
        if msg["role"] == "system":
            messages.append(SystemMessage(content=msg["content"]))
        elif msg["role"] == "user":
            messages.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            messages.append(AIMessage(content=msg["content"]))
    
    # Call LLM to generate network evaluation
    response = llm.invoke(messages)
    evaluation = response.content
    
    # Record final result
    state["final_result"] = evaluation
    
    return state

def route_next(state: NetworkState) -> str:
    """Route to next node"""
    # Check step count to prevent infinite loop
    if state["step_count"] >= 10:
        return END
    
    # Follow the 4-step workflow
    if state["current_step"] == "understand_intent":
        return "understand_intent"
    elif state["current_step"] == "allocate_slice_type":
        return "allocate_slice_type"
    elif state["current_step"] == "allocate_resources":
        return "allocate_resources"
    elif state["current_step"] == "evaluate_network":
        return "evaluate_network"
    else:
        return END

# ====================== Build Workflow ======================

def create_network_graph():
    """Create network slice management workflow graph"""
    graph = StateGraph(NetworkState)
    
    # Add nodes
    graph.add_node("initialize", initialize)
    graph.add_node("understand_intent", understand_intent)
    graph.add_node("allocate_slice_type", allocate_slice_type)
    graph.add_node("allocate_resources", allocate_resources)
    graph.add_node("evaluate_network", evaluate_network)
    
    # Set entry point
    graph.set_entry_point("initialize")
    
    # Add edges
    graph.add_conditional_edges(
        "initialize",
        route_next,
        {
            "understand_intent": "understand_intent"
        }
    )
    
    graph.add_conditional_edges(
        "understand_intent",
        route_next,
        {
            "understand_intent": "understand_intent",
            "allocate_slice_type": "allocate_slice_type"
        }
    )
    
    graph.add_conditional_edges(
        "allocate_slice_type",
        route_next,
        {
            "allocate_slice_type": "allocate_slice_type",
            "allocate_resources": "allocate_resources"
        }
    )
    
    graph.add_conditional_edges(
        "allocate_resources",
        route_next,
        {
            "allocate_resources": "allocate_resources",
            "evaluate_network": "evaluate_network"
        }
    )
    
    # Add termination edge
    graph.add_edge("evaluate_network", END)
    
    # Compile graph
    return graph.compile()

# ====================== Main Function ======================

def process_user_request(user_id, location, request, cqi=None, ground_truth=None, override_network_state=None):
    """Main function for processing user requests with resource utilization tracking
    
    Parameters:
    - user_id: User identifier
    - location: User location coordinates
    - request: Text of user request
    - cqi: Channel Quality Indicator (1-15), generated randomly if None
    - ground_truth: Ground truth slice label for evaluating intent understanding
    - override_network_state: Optional state override for testing
    """
    # Generate random CQI if not provided
    if cqi is None:
        cqi = generate_random_cqi()
    
    # Create workflow graph
    workflow = create_network_graph()
    
    # Get initial network state for resource utilization comparison
    initial_network_state = get_current_network_state()
    
    # Get initial total transmission rates
    initial_embb_total_rate, initial_urllc_total_rate = calculate_total_transmission_rates()
    
    # Get initial average resource utilization
    initial_avg_resource_util = calculate_average_resource_utilization()
    
    # Initial state
    initial_state = {
        "user_id": user_id,
        "location": location,
        "request": request,
        "cqi": cqi,
        "history": [],
        "memory": {},
        "step_count": 0,
        "current_step": "",
        "final_result": None
    }
    
    # If override network state is provided
    if override_network_state:
        # Temporarily override global network state (for testing only)
        update_network_state(override_network_state)
    
    # Execute workflow
    try:
        result = workflow.invoke(initial_state)
        
        # Get final network state for resource utilization comparison
        final_network_state = get_current_network_state()
        
        # Get final total transmission rates
        final_embb_total_rate, final_urllc_total_rate = calculate_total_transmission_rates()
        
        # Get final average resource utilization
        final_avg_resource_util = calculate_average_resource_utilization()
        
        final_embb_util = final_network_state["embb_slice"]["utilization_rate"]
        final_urllc_util = final_network_state["urllc_slice"]["utilization_rate"]
        
        # Get detailed allocation results
        detailed_result = {
            "user_id": user_id,
            "request": request,
            "cqi": cqi,
            "ground_truth": ground_truth,
            "allocation_failed": result["memory"].get("allocation_failed", False)
        }
        
        # Add slice type
        if "memory" in result and "final_slice" in result["memory"]:
            detailed_result["slice_type"] = result["memory"]["final_slice"]
        else:
            detailed_result["slice_type"] = "Failed"
        
        # Check if intent understanding matches ground truth
        if ground_truth is not None and detailed_result["slice_type"] != "Failed":
            detailed_result["intent_correct"] = (detailed_result["slice_type"] == ground_truth)
        else:
            detailed_result["intent_correct"] = None
        
        # Add bandwidth, rate, latency
        if "memory" in result and "beamforming_result" in result["memory"]:
            beamforming = result["memory"]["beamforming_result"]
            detailed_result["bandwidth"] = beamforming["allocated_bandwidth"]
            detailed_result["rate"] = beamforming["allocated_rate"]
            detailed_result["latency"] = beamforming["allocated_latency"]
        
        # Add adjustments made
        if "memory" in result and "adjustment_result" in result["memory"]:
            detailed_result["adjustments_made"] = result["memory"]["adjustment_result"].get("adjustments_made", False)
        
        # Add transmission rate statistics
        detailed_result["embb_total_rate_before"] = initial_embb_total_rate
        detailed_result["embb_total_rate_after"] = final_embb_total_rate
        detailed_result["urllc_total_rate_before"] = initial_urllc_total_rate
        detailed_result["urllc_total_rate_after"] = final_urllc_total_rate
        
        # Add average resource utilization
        detailed_result["avg_resource_util_before"] = initial_avg_resource_util
        detailed_result["avg_resource_util_after"] = final_avg_resource_util

        detailed_result["embb_util_after"] = final_embb_util
        detailed_result["urllc_util_after"] = final_urllc_util
        
        return detailed_result
    except Exception as e:
        print(f"Workflow execution error: {str(e)}")
        return {
            "user_id": user_id,
            "request": request,
            "cqi": cqi,
            "ground_truth": ground_truth,
            "slice_type": "Failed",
            "allocation_failed": True,
            "intent_correct": None,
            "error": str(e),
            # Include transmission rates even for failures
            "embb_total_rate_before": initial_embb_total_rate,
            "embb_total_rate_after": initial_embb_total_rate,
            "urllc_total_rate_before": initial_urllc_total_rate,
            "urllc_total_rate_after": initial_urllc_total_rate,
            "avg_resource_util_before": initial_avg_resource_util,
            "avg_resource_util_after": initial_avg_resource_util
        }

def main(num_users=4, export_file="fileName.csv"):
    """Main program with CSV-based user testing and enhanced analytics
    
    Parameters:
    - num_users: Number of users to test (default: 4)
    - export_file: Path to export results CSV file
    """
    print("Starting network slice management system with CSV-based user testing...\n")
    
    # Path to ray tracing results CSV
    ray_tracing_csv = r"C:\Users\Jingwen TONG\Desktop\我的文档\02_项目_202301\16-WirelessAgent-ChinaCom\Simulations\WirelessAgent_LangGraph\Knowledge_Base\ray_tracing_results.csv"
    
    # Load users from CSV (limit to specified number)
    users = load_user_data_from_csv(ray_tracing_csv, num_users)
    
    print(f"Testing {len(users)} users from ray tracing results CSV")
    
    # Initialize results tracker
    detailed_results = []

    # Track slice utilization across all allocations
    embb_utils = []
    urllc_utils = []
    workload_balanced_count = 0
    
    # Process each user
    for i, user in enumerate(users):
        print(f"\n{'-'*140}")
        print(f"PROCESSING USER {user['user_id']} ({i+1}/{len(users)})")
        print(f"Request: \"{user['request']}\"")
        print(f"CQI: {user['cqi']}")
        if user.get('ground_truth'):
            print(f"Ground Truth Slice: {user['ground_truth']}")
        print(f"{'-'*140}")
        
        # Reset network state for clean testing
        reset_network_state()
        
        # Process the user
        result = process_user_request(
            user_id=user['user_id'],
            location=user['location'],
            request=user['request'],
            cqi=user['cqi'],
            ground_truth=user.get('ground_truth')
        )
        
        # Store detailed result for CSV export
        detailed_results.append(result)
        # Track workload balancing
        if result.get("workload_balanced", False):
            workload_balanced_count += 1
        
        # Track slice utilization
        if not result.get("allocation_failed", True):
            # Convert percentage string to float (remove % and convert)
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
                # Handle cases where util might not be a string or doesn't have % format
                pass
    
    # Calculate average slice utilization
    avg_embb_util = sum(embb_utils) / len(embb_utils) if embb_utils else 0
    avg_urllc_util = sum(urllc_utils) / len(urllc_utils) if urllc_utils else 0
    
    # Get final total transmission rates
    final_embb_total_rate, final_urllc_total_rate = calculate_total_transmission_rates()
    
    # Get final average resource utilization
    final_avg_resource_util = calculate_average_resource_utilization()
    
    # Calculate intent understanding rate
    correct_intents = 0
    total_evaluated = 0
    
    for result in detailed_results:
        if result.get("intent_correct") is not None:
            total_evaluated += 1
            if result["intent_correct"]:
                correct_intents += 1
    
    intent_rate = 0 if total_evaluated == 0 else (correct_intents / total_evaluated) * 100
    
    # Calculate workload balancing rate
    workload_balanced_rate = 0 if len(detailed_results) == 0 else (workload_balanced_count / len(detailed_results)) * 100

    # Print summary of all results
    print("\n" + "="*60)
    print("SUMMARY OF USER ALLOCATIONS")
    print("="*60)
    
    # Generate summary table with ASCII-compatible symbols (no Unicode)
    summary_rows = []
    for res in detailed_results:
        status = "Success" if not res.get("allocation_failed", True) else "Failed"
        intent_match = ""
        if res.get("intent_correct") is not None:
            intent_match = "Yes" if res["intent_correct"] else "No"
        
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
            "Yes" if res.get("adjustments_made", False) else "No"
        ])
    
    headers = ["User ID", "Status", "Slice", "Ground Truth", "Intent Match", "CQI", "BW (MHz)", "Rate (Mbps)", "Latency (ms)", "Adjusted"]
    summary_table = tabulate(summary_rows, headers=headers, tablefmt="grid")
    print(summary_table)
    
    # Print statistics
    success_count = sum(1 for res in detailed_results if not res.get("allocation_failed", True))
    total_count = len(detailed_results)
    
    print("\nStatistics:")
    print(f"Success rate: {success_count}/{total_count} ({success_count/total_count*100:.1f}%)")
    
    # Print intent understanding statistics
    print("\nIntent Understanding Evaluation:")
    print(f"Correctly identified intents: {correct_intents}/{total_evaluated}")
    print(f"Intent understanding rate: {intent_rate:.1f}%")

    # Print workload balancing statistics
    print("\nWorkload Balancing Statistics:")
    print(f"Users with workload balancing: {workload_balanced_count}/{total_count}")
    print(f"Workload balancing rate: {workload_balanced_rate:.1f}%")
    
    # Print slice utilization statistics
    print("\nSlice Utilization Statistics:")
    print(f"Average eMBB utilization: {avg_embb_util:.2f}%")
    print(f"Average URLLC utilization: {avg_urllc_util:.2f}%")
    
    # Print transmission rate statistics
    print("\nTransmission Rate Statistics:")
    print(f"Final eMBB total rate: {final_embb_total_rate:.2f} Mbps")
    print(f"Final URLLC total rate: {final_urllc_total_rate:.2f} Mbps")
    
    # Print resource utilization statistics
    print("\nResource Utilization:")
    print(f"Average resource utilization: {final_avg_resource_util:.2f}%")
    
    # Prepare data for CSV export
    slice_stats = {
        "avg_resource_util": f"{final_avg_resource_util:.2f}",
        "final_resource_util": f"{final_avg_resource_util:.2f}",
        "final_embb_total_rate": f"{final_embb_total_rate:.2f}",
        "final_urllc_total_rate": f"{final_urllc_total_rate:.2f}"
    }
    
    intent_stats = {
        "total": total_evaluated,
        "correct": correct_intents,
        "rate": f"{intent_rate:.2f}"
    }
    
    # Export results to CSV
    export_results_to_csv(detailed_results, slice_stats, intent_stats, export_file)

if __name__ == "__main__":
    # Test with 10 users by default and export results to CSV
    main(num_users=3, export_file="network_slicing_results_DSv3NKB.csv") # The number of users can be adjusted as needed