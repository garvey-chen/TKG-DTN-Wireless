# 2025/3/25 add a csv file to store the optimal allocation results.
import numpy as np
from scipy import optimize
import math
import csv
import os
from datetime import datetime

def calculate_rate(b, q, alpha):
    """Calculate the rate for a user given bandwidth and CQI."""
    return alpha * b * math.log10(1 + 10**(q/10))

def is_allocation_feasible(B, M, Q, B_min, B_max, R_min, R_max, alpha):
    """Check if the allocation problem is feasible."""
    
    # Check if any minimum rate requirements are infeasible for any user
    for i in range(M):
        max_possible_rate = calculate_rate(B_max, Q[i], alpha)
        if max_possible_rate < R_min:
            print(f"User {i} cannot achieve minimum rate {R_min} even with maximum bandwidth {B_max}")
            return False
    
    # Check if total bandwidth constraints are feasible
    min_total_bandwidth = B_min * M
    if B < min_total_bandwidth:
        print(f"Total bandwidth {B} is less than the minimum required {min_total_bandwidth}")
        return False
    
    # Calculate bandwidth required for minimum rate for each user
    min_bandwidth_for_rate = []
    for i in range(M):
        spectral_efficiency = math.log10(1 + 10**(Q[i]/10))
        required_b = R_min / (alpha * spectral_efficiency)
        min_bandwidth_for_rate.append(max(required_b, B_min))
    
    # Check if minimum rate requirements can be met with available bandwidth
    total_min_bandwidth = sum(min_bandwidth_for_rate)
    if total_min_bandwidth > B:
        print(f"Total bandwidth {B} is insufficient to meet minimum rate requirements ({total_min_bandwidth})")
        return False
    
    return True

def fallback_allocation(B, M, Q, B_min, B_max, R_min, R_max, alpha):
    """Fallback algorithm when optimization fails - prioritize users with higher CQI."""
    # Start with minimum bandwidth for each user
    allocation = np.ones(M) * B_min
    remaining = B - np.sum(allocation)
    
    # Calculate minimum bandwidth needed for R_min for each user
    min_needed = []
    for i in range(M):
        spectral_efficiency = math.log10(1 + 10**(Q[i]/10))
        required_b = R_min / (alpha * spectral_efficiency)
        # If required bandwidth > B_min, allocate more
        additional = max(0, required_b - B_min)
        min_needed.append(additional)
    
    # Allocate minimum bandwidth needed for R_min first
    for i in range(M):
        if min_needed[i] > 0:
            to_allocate = min(remaining, min_needed[i])
            allocation[i] += to_allocate
            remaining -= to_allocate
            
    # Distribute remaining bandwidth to users with higher CQI
    if remaining > 0:
        # Sort users by CQI in descending order
        sorted_indices = np.argsort(-Q)
        for i in sorted_indices:
            # Calculate how much more bandwidth we can give to this user
            more_bandwidth = min(remaining, B_max - allocation[i])
            allocation[i] += more_bandwidth
            remaining -= more_bandwidth
            if remaining <= 1e-6:
                break
    
    # Calculate resulting rates
    rates = np.array([calculate_rate(allocation[i], Q[i], alpha) for i in range(M)])
    
    # Verify all constraints are met
    for i in range(M):
        if rates[i] < R_min:
            print(f"Warning: User {i} rate {rates[i]:.2f} is below minimum {R_min}")
        if rates[i] > R_max:
            # Cap the bandwidth to achieve R_max
            spectral_efficiency = math.log10(1 + 10**(Q[i]/10))
            max_b_needed = R_max / (alpha * spectral_efficiency)
            excess = allocation[i] - max_b_needed
            if excess > 0:
                allocation[i] = max_b_needed
                # Redistribute excess bandwidth to other users
                for j in sorted_indices:
                    if j != i and allocation[j] < B_max:
                        more = min(excess, B_max - allocation[j])
                        allocation[j] += more
                        excess -= more
                        if excess <= 1e-6:
                            break
    
    # Recalculate rates after adjustments
    rates = np.array([calculate_rate(allocation[i], Q[i], alpha) for i in range(M)])
    return allocation, rates, True

def solve_resource_allocation(B, M, Q, B_min, B_max, R_min, R_max, alpha):
    """Solve the resource allocation problem with proportional fairness."""
    
    # Define the objective function for proportional fairness (maximize sum of log rates)
    def objective(B_alloc):
        rates = np.array([calculate_rate(b, q, alpha) for b, q in zip(B_alloc, Q)])
        # Avoid log(0) by using a small epsilon
        epsilon = 1e-10
        return -np.sum(np.log(rates + epsilon))
    
    # Define constraint classes to handle closure issues
    class TotalBandwidthConstraint:
        def __init__(self, target_b):
            self.target_b = target_b
        
        def __call__(self, B_alloc):
            return np.sum(B_alloc) - self.target_b
    
    class MinRateConstraint:
        def __init__(self, user_idx, q_val, min_r):
            self.idx = user_idx
            self.q = q_val
            self.min_r = min_r
        
        def __call__(self, B_alloc):
            rate = calculate_rate(B_alloc[self.idx], self.q, alpha)
            return rate - self.min_r
    
    class MaxRateConstraint:
        def __init__(self, user_idx, q_val, max_r):
            self.idx = user_idx
            self.q = q_val
            self.max_r = max_r
        
        def __call__(self, B_alloc):
            rate = calculate_rate(B_alloc[self.idx], self.q, alpha)
            return self.max_r - rate
    
    # Check if the problem is feasible
    if not is_allocation_feasible(B, M, Q, B_min, B_max, R_min, R_max, alpha):
        # Try fallback allocation if standard feasibility check fails
        print("Standard feasibility check failed. Trying fallback allocation method...")
        return fallback_allocation(B, M, Q, B_min, B_max, R_min, R_max, alpha)
    
    # Create all constraints
    constraints = [{'type': 'eq', 'fun': TotalBandwidthConstraint(B)}]
    
    # Add rate constraints for each user
    for i in range(M):
        constraints.append({'type': 'ineq', 'fun': MinRateConstraint(i, Q[i], R_min)})
        constraints.append({'type': 'ineq', 'fun': MaxRateConstraint(i, Q[i], R_max)})
    
    # Set bounds for each user's bandwidth allocation
    bounds = [(B_min, B_max) for _ in range(M)]
    
    # Create a good initial guess to help convergence
    initial_allocation = np.zeros(M)
    for i in range(M):
        spectral_efficiency = math.log10(1 + 10**(Q[i]/10))
        required_b = R_min / (alpha * spectral_efficiency)
        initial_allocation[i] = max(required_b, B_min)
    
    # Distribute remaining bandwidth proportionally to CQI
    remaining = B - np.sum(initial_allocation)
    if remaining > 0:
        q_norm = Q / np.sum(Q)
        for i in range(M):
            additional = min(remaining * q_norm[i], B_max - initial_allocation[i])
            initial_allocation[i] += additional
            remaining -= additional
    
    # If there's still bandwidth left, allocate it to users with highest CQI
    if remaining > 0:
        sorted_indices = np.argsort(-Q)  # Sort in descending order
        for i in sorted_indices:
            additional = min(remaining, B_max - initial_allocation[i])
            initial_allocation[i] += additional
            remaining -= additional
            if remaining <= 1e-5:
                break
    
    # Solve the optimization problem
    try:
        result = optimize.minimize(
            objective,
            initial_allocation,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'ftol': 1e-6, 'maxiter': 1000}
        )
        
        # Check if optimization was successful
        if result.success:
            B_allocation = result.x
            R_allocation = np.array([calculate_rate(b, q, alpha) for b, q in zip(B_allocation, Q)])
            
            # Verify constraints are satisfied
            total_bandwidth_used = np.sum(B_allocation)
            bandwidth_constraint_satisfied = abs(total_bandwidth_used - B) < 1e-5
            
            rate_constraints_satisfied = True
            for i in range(M):
                if R_allocation[i] < R_min - 1e-5 or R_allocation[i] > R_max + 1e-5:
                    rate_constraints_satisfied = False
                    break
            
            if bandwidth_constraint_satisfied and rate_constraints_satisfied:
                return B_allocation, R_allocation, True
            else:
                print("Optimization completed but constraints are not satisfied. Using fallback method...")
                return fallback_allocation(B, M, Q, B_min, B_max, R_min, R_max, alpha)
        else:
            print(f"Optimization failed: {result.message}")
            print("Using fallback bandwidth allocation method...")
            return fallback_allocation(B, M, Q, B_min, B_max, R_min, R_max, alpha)
    except Exception as e:
        print(f"Error during optimization: {e}")
        print("Using fallback bandwidth allocation method...")
        return fallback_allocation(B, M, Q, B_min, B_max, R_min, R_max, alpha)

def process_embb_users(file_path, max_users=30):
    """
    Process eMBB users from the CSV file and perform bandwidth allocation.
    
    Parameters:
    file_path (str): Path to the CSV file
    max_users (int): Number of users to consider (from the beginning of the file)
    
    Returns:
    dict: Results including user allocations and summary statistics
    """
    # Parameters for resource allocation
    B = 90.0  # Total bandwidth
    B_min = 6.0
    B_max = 20.0
    R_min = 100.0
    R_max = 400.0
    alpha = 10.0
    
    # Extract users from CSV file
    embb_users = []
    
    try:
        with open(file_path, 'r') as csv_file:
            csvreader = csv.DictReader(csv_file)
            
            for i, row in enumerate(csvreader):
                if i >= max_users:
                    break
                    
                user_id = int(row['RX_ID'])
                cqi = int(row['CQI'])
                user_type = row['Request_Label']
                
                if user_type == 'eMBB':
                    embb_users.append((user_id, cqi))
    
        print(f"Analyzed first {max_users} users from the CSV file")
        print(f"Found {len(embb_users)} eMBB users")
        
        # Extract CQI values for eMBB users
        M = len(embb_users)
        if M == 0:
            print("No eMBB users found. Cannot perform bandwidth allocation.")
            return {
                'num_users': 0,
                'user_allocations': [],
                'total_bandwidth': 0,
                'total_rate': 0,
                'resource_utilization': 0
            }
            
        Q = np.array([user[1] for user in embb_users])
        
        # Solve the resource allocation problem for eMBB users
        B_allocation, R_allocation, success = solve_resource_allocation(
            B, M, Q, B_min, B_max, R_min, R_max, alpha
        )
        
        if success:
            print("\nResource allocation successful!")
            print("\nBandwidth and rate allocation for eMBB users:")
            
            user_allocations = []
            total_rate = 0
            for i in range(M):
                user_id = embb_users[i][0]
                print(f"User {user_id} (CQI={Q[i]}): {B_allocation[i]:.2f} bandwidth units, {R_allocation[i]:.2f} rate units")
                
                user_allocations.append({
                    'user_id': user_id,
                    'user_type': 'eMBB',
                    'cqi': Q[i],
                    'bandwidth': B_allocation[i],
                    'rate': R_allocation[i]
                })
                
                total_rate += R_allocation[i]
            
            total_bandwidth = np.sum(B_allocation)
            resource_utilization = (total_bandwidth / B) * 100
            
            print(f"\nTotal bandwidth used: {total_bandwidth:.2f}")
            print(f"Total transmission rate: {total_rate:.2f}")
            print(f"Resource utilization: {resource_utilization:.2f}%")
            
            return {
                'num_users': M,
                'user_allocations': user_allocations,
                'total_bandwidth': total_bandwidth,
                'total_rate': total_rate,
                'resource_utilization': resource_utilization
            }
        else:
            print("Failed to find a feasible allocation for this round.")
            return {
                'num_users': M,
                'user_allocations': [],
                'total_bandwidth': 0,
                'total_rate': 0,
                'resource_utilization': 0
            }
           
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return {
            'num_users': 0,
            'user_allocations': [],
            'total_bandwidth': 0,
            'total_rate': 0,
            'resource_utilization': 0
        }
    except Exception as e:
        print(f"Error processing CSV file: {e}")
        return {
            'num_users': 0,
            'user_allocations': [],
            'total_bandwidth': 0,
            'total_rate': 0,
            'resource_utilization': 0
        }

def process_urllc_users(file_path, max_users=30):
    """
    Process URLLC users from the CSV file and perform bandwidth allocation.
    
    Parameters:
    file_path (str): Path to the CSV file
    max_users (int): Number of users to consider (from the beginning of the file)
    
    Returns:
    dict: Results including user allocations and summary statistics
    """
    # Parameters for resource allocation
    B = 30.0  # Total bandwidth
    B_min = 1.0
    B_max = 5.0
    R_min = 1.0
    R_max = 100.0
    alpha = 10.0
    
    # Extract users from CSV file
    urllc_users = []
    
    try:
        with open(file_path, 'r') as csv_file:
            csvreader = csv.DictReader(csv_file)
            
            for i, row in enumerate(csvreader):
                if i >= max_users:
                    break
                    
                user_id = int(row['RX_ID'])
                cqi = int(row['CQI'])
                user_type = row['Request_Label']
                
                if user_type == 'URLLC':
                    urllc_users.append((user_id, cqi))
    
        print(f"Analyzed first {max_users} users from the CSV file")
        print(f"Found {len(urllc_users)} URLLC users")
        
        # Extract CQI values for URLLC users
        M = len(urllc_users)
        if M == 0:
            print("No URLLC users found. Cannot perform bandwidth allocation.")
            return {
                'num_users': 0,
                'user_allocations': [],
                'total_bandwidth': 0,
                'total_rate': 0,
                'resource_utilization': 0
            }
            
        Q = np.array([user[1] for user in urllc_users])
        
        # Solve the resource allocation problem for URLLC users
        B_allocation, R_allocation, success = solve_resource_allocation(
            B, M, Q, B_min, B_max, R_min, R_max, alpha
        )
        
        if success:
            print("\nResource allocation successful!")
            print("\nBandwidth and rate allocation for URLLC users:")
            
            user_allocations = []
            total_rate = 0
            for i in range(M):
                user_id = urllc_users[i][0]
                print(f"User {user_id} (CQI={Q[i]}): {B_allocation[i]:.2f} bandwidth units, {R_allocation[i]:.2f} rate units")
                
                user_allocations.append({
                    'user_id': user_id,
                    'user_type': 'URLLC',
                    'cqi': Q[i],
                    'bandwidth': B_allocation[i],
                    'rate': R_allocation[i]
                })
                
                total_rate += R_allocation[i]
            
            total_bandwidth = np.sum(B_allocation)
            resource_utilization = (total_bandwidth / B) * 100
            
            print(f"\nTotal bandwidth used: {total_bandwidth:.2f}")
            print(f"Total transmission rate: {total_rate:.2f}")
            print(f"Resource utilization: {resource_utilization:.2f}%")
            
            return {
                'num_users': M,
                'user_allocations': user_allocations,
                'total_bandwidth': total_bandwidth,
                'total_rate': total_rate,
                'resource_utilization': resource_utilization
            }
        else:
            print("Failed to find a feasible allocation for this round.")
            return {
                'num_users': M,
                'user_allocations': [],
                'total_bandwidth': 0,
                'total_rate': 0,
                'resource_utilization': 0
            }
            
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return {
            'num_users': 0,
            'user_allocations': [],
            'total_bandwidth': 0,
            'total_rate': 0,
            'resource_utilization': 0
        }
    except Exception as e:
        print(f"Error processing CSV file: {e}")
        return {
            'num_users': 0,
            'user_allocations': [],
            'total_bandwidth': 0,
            'total_rate': 0,
            'resource_utilization': 0
        }

def save_results_to_csv(results, output_file='Optimal_RA_Results.csv'):
    """
    Save all results to a CSV file.
    
    Parameters:
    results (list): List of dictionaries containing results for each round
    output_file (str): Path to the output CSV file
    """
    # Check if file exists to determine if we need to write headers
    file_exists = os.path.isfile(output_file)
    
    with open(output_file, 'w', newline='') as csvfile:
        # Define the CSV fields
        fieldnames = ['Round', 'MaxUsers', 'UserID', 'UserType', 'CQI', 'Bandwidth', 'Rate',
                    'TotalBandwidth', 'ResourceUtilization', 'TotalEMBBRate', 'TotalURLLCRate']
        
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        # Write data for each round
        for round_idx, round_data in enumerate(results):
            max_users = round_data['max_users']
            
            # Get user allocations for both types
            all_users = []
            if 'embb' in round_data:
                all_users.extend(round_data['embb']['user_allocations'])
            if 'urllc' in round_data:
                all_users.extend(round_data['urllc']['user_allocations'])
            
            # Get total rates for both types
            embb_total_rate = round_data['embb']['total_rate'] if 'embb' in round_data else 0
            urllc_total_rate = round_data['urllc']['total_rate'] if 'urllc' in round_data else 0
            
            # Get resource utilization (average if both types are present)
            embb_util = round_data['embb']['resource_utilization'] if 'embb' in round_data else 0
            urllc_util = round_data['urllc']['resource_utilization'] if 'urllc' in round_data else 0
            avg_util = (embb_util + urllc_util) / 2 if (embb_util > 0 and urllc_util > 0) else max(embb_util, urllc_util)
            
            # Get total bandwidth used
            embb_bw = round_data['embb']['total_bandwidth'] if 'embb' in round_data else 0
            urllc_bw = round_data['urllc']['total_bandwidth'] if 'urllc' in round_data else 0
            total_bw = embb_bw + urllc_bw
            
            # Write user-specific rows
            for user in all_users:
                writer.writerow({
                    'Round': round_idx + 1,
                    'MaxUsers': max_users,
                    'UserID': user['user_id'],
                    'UserType': user['user_type'],
                    'CQI': user['cqi'],
                    'Bandwidth': round(user['bandwidth'], 2),
                    'Rate': round(user['rate'], 2),
                    'TotalBandwidth': round(total_bw, 2),
                    'ResourceUtilization': round(avg_util, 2),
                    'TotalEMBBRate': round(embb_total_rate, 2),
                    'TotalURLLCRate': round(urllc_total_rate, 2)
                })
            
            # If no users were allocated, still write a summary row
            if not all_users:
                writer.writerow({
                    'Round': round_idx + 1,
                    'MaxUsers': max_users,
                    'UserID': 'N/A',
                    'UserType': 'N/A',
                    'CQI': 0,
                    'Bandwidth': 0,
                    'Rate': 0,
                    'TotalBandwidth': 0,
                    'ResourceUtilization': 0,
                    'TotalEMBBRate': 0,
                    'TotalURLLCRate': 0
                })

def main():
    # File path as specified
    file_path = r"C:\Users\Jingwen TONG\Desktop\我的文档\02_项目_202301\16-WirelessAgent-ChinaCom\Simulations\WirelessAgent_LangGraph\Knowledge_Base\ray_tracing_results_north.csv"
    
    # Default max_users setting
    max_users_list = np.array([1, 5, 10, 15, 20, 25, 30])
    
    # Store all results for CSV output
    all_results = []
    
    # You can change the max_users parameter here if needed
    for max_users in max_users_list:
        print("\n" + "="*120)
        print(f"SUMMARY OF USER ALLOCATIONS (MAX USERS: {max_users})")
        print("="*120) 
        
        # Process eMBB users
        embb_results = process_embb_users(file_path, max_users)
        print("-"*60)
        
        # Process URLLC users
        urllc_results = process_urllc_users(file_path, max_users)
        
        # Store results for this round
        round_results = {
            'max_users': max_users,
            'embb': embb_results,
            'urllc': urllc_results
        }
        
        all_results.append(round_results)
    
    # Save all results to CSV file
    output_file = "Optimal_RA_Results_north.csv"
    save_results_to_csv(all_results, output_file)
    print(f"\nResults saved to {output_file}")

if __name__ == "__main__":
    main()