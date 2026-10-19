# add ground truth to the request, in order to computer the intent understanding accuracy.
# add the CQI value in the output file
# add the user request to the output file 
import xml.etree.ElementTree as ET
import numpy as np
import matplotlib.pyplot as plt
import random
import math
from pyproj import Transformer
from shapely.geometry import Point, Polygon, LineString
import csv

def parse_osm_buildings(file_path):
    """Parse OSM file to extract building information."""
    tree = ET.parse(file_path)
    root = tree.getroot()
    
    buildings = []
    nodes = {}
    
    # Extract all nodes
    for node in root.findall('./node'):
        node_id = node.get('id')
        lat = float(node.get('lat'))
        lon = float(node.get('lon'))
        nodes[node_id] = (lat, lon)
    
    # Extract buildings
    for way in root.findall('./way'):
        is_building = False
        for tag in way.findall('./tag'):
            if tag.get('k') == 'building':
                is_building = True
                break
        
        if is_building:
            building_nodes = []
            height = 10.0  # Default height in meters
            
            # Extract height if available
            for tag in way.findall('./tag'):
                if tag.get('k') == 'height':
                    try:
                        height = float(tag.get('v'))
                    except ValueError:
                        pass
                elif tag.get('k') == 'building:levels':
                    try:
                        # Approximate height based on levels (3m per level)
                        height = float(tag.get('v')) * 3.0
                    except ValueError:
                        pass
            
            # Get building outline
            node_refs = []
            for nd in way.findall('./nd'):
                ref = nd.get('ref')
                node_refs.append(ref)
            
            # Ensure the building polygon is closed
            if node_refs and node_refs[0] != node_refs[-1]:
                node_refs.append(node_refs[0])
            
            for ref in node_refs:
                if ref in nodes:
                    building_nodes.append(nodes[ref])
            
            if len(building_nodes) >= 3:  # Need at least 3 points for a valid polygon
                buildings.append({
                    'nodes': building_nodes,
                    'height': height
                })
    
    return buildings

def convert_to_cartesian(buildings):
    """Convert geographic coordinates to local Cartesian coordinates."""
    if not buildings:
        return [], None, (0, 0)
    
    # Get reference point (center of the first building)
    ref_lat = sum(node[0] for node in buildings[0]['nodes']) / len(buildings[0]['nodes'])
    ref_lon = sum(node[1] for node in buildings[0]['nodes']) / len(buildings[0]['nodes'])
    
    # Create transformer
    transformer = Transformer.from_crs(
        "EPSG:4326",  # WGS84
        f"+proj=tmerc +lat_0={ref_lat} +lon_0={ref_lon} +k=1 +x_0=0 +y_0=0 +ellps=WGS84 +units=m +no_defs",
        always_xy=True
    )
    
    cartesian_buildings = []
    for building in buildings:
        cartesian_nodes = []
        for lat, lon in building['nodes']:
            x, y = transformer.transform(lon, lat)
            cartesian_nodes.append((x, y))
        
        cartesian_buildings.append({
            'nodes': cartesian_nodes,
            'height': building['height'],
            'polygon': Polygon(cartesian_nodes)
        })
    
    return cartesian_buildings, transformer, (ref_lat, ref_lon)

def find_tallest_building(buildings):
    """Find the tallest building in the dataset."""
    if not buildings:
        return None
    tallest = max(buildings, key=lambda x: x['height'])
    return tallest

def place_tx(tallest_building):
    """Place transmitter at the center of the tallest building."""
    # Calculate centroid of the building
    centroid = tallest_building['polygon'].centroid
    height = tallest_building['height']
    
    # Place TX at the top of the building
    tx_position = (centroid.x, centroid.y, height)
    return tx_position

def generate_rx_positions(bounds, num_rx, buildings, min_distance=5.0):
    """Generate random RX positions, avoiding building interiors."""
    min_x, max_x, min_y, max_y = bounds
    
    rx_positions = []
    attempts = 0
    max_attempts = 100000  # Prevent infinite loops
    
    while len(rx_positions) < num_rx and attempts < max_attempts:
        # Generate random position within bounds
        x = random.uniform(min_x, max_x)
        y = random.uniform(min_y, max_y)
        z = 1.5  # Assume RX is at human height (1.5m)
        
        point = Point(x, y)
        
        # Check if point is inside any building
        inside_building = False
        for building in buildings:
            if building['polygon'].contains(point):
                inside_building = True
                break
        
        # Check minimum distance from existing receivers
        too_close = False
        for rx_pos in rx_positions:
            dist = math.sqrt((x - rx_pos[0])**2 + (y - rx_pos[1])**2)
            if dist < min_distance:
                too_close = True
                break
        
        if not inside_building and not too_close:
            rx_positions.append((x, y, z))
        
        attempts += 1
    
    print(f"Generated {len(rx_positions)} RX positions after {attempts} attempts")
    if len(rx_positions) < num_rx:
        print(f"Warning: Only generated {len(rx_positions)} out of {num_rx} requested positions")
    
    return rx_positions

def has_line_of_sight(p1, p2, buildings):
    """Check if there's line of sight between two points."""
    line = LineString([(p1[0], p1[1]), (p2[0], p2[1])])
    
    # Vector from p1 to p2
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    dz = p2[2] - p1[2]
    
    # Total distance
    dist = math.sqrt(dx**2 + dy**2 + dz**2)
    
    for building in buildings:
        # Check if line intersects with building footprint
        if line.intersects(building['polygon']):
            # Skip if line only touches the polygon edge
            if line.touches(building['polygon']):
                continue
                
            # Get intersection points with the building polygon
            intersection = line.intersection(building['polygon'].boundary)
            
            # If intersection is a point or multipoint, check each intersection
            if intersection.geom_type == 'Point':
                points = [intersection]
            elif intersection.geom_type == 'MultiPoint':
                points = list(intersection.geoms)
            else:
                # For more complex intersections, simplify and take first/last points
                points = [Point(line.coords[0]), Point(line.coords[-1])]
            
            for point in points:
                # Calculate how far along the line the intersection occurs
                t_x = (point.x - p1[0]) / dx if dx != 0 else 0
                t_y = (point.y - p1[1]) / dy if dy != 0 else 0
                
                # Use average t value for more stable calculation
                t_values = [v for v in [t_x, t_y] if v != 0]
                if not t_values:
                    continue
                    
                t = sum(t_values) / len(t_values)
                
                # Calculate z at intersection
                z_intersect = p1[2] + t * dz
                
                # If intersection is below building height, path is blocked
                if 0 <= t <= 1 and z_intersect < building['height']:
                    return False
    
    return True

def calculate_path_loss(tx_position, rx_position, frequency, has_los):
    """Calculate path loss using appropriate propagation models."""
    # Calculate distance between TX and RX
    distance = math.sqrt(
        (tx_position[0] - rx_position[0])**2 +
        (tx_position[1] - rx_position[1])**2 +
        (tx_position[2] - rx_position[2])**2
    )
    
    # Avoid division by zero or very small distances
    distance = max(distance, 1.0)
    
    if has_los:
        # Line of sight - use free space path loss
        # FSPL(dB) = 20log10(d) + 20log10(f) - 147.55
        path_loss_dB = 20 * math.log10(distance) + 20 * math.log10(frequency) - 147.55
    else:
        # Non-line of sight - use simplified COST231 model
        # Basic model: NLOS loss = FSPL + 20 + 30log10(d/100)
        fspl_dB = 20 * math.log10(distance) + 20 * math.log10(frequency) - 147.55
        nlos_loss = 20 + 30 * math.log10(max(distance/100, 0.1))
        path_loss_dB = fspl_dB + nlos_loss
    
    return path_loss_dB

def calculate_cqi(snr_dB):
    """
    Calculate Channel Quality Indicator (CQI) based on SNR.
    Maps SNR values to CQI range of 1-15 where:
    - CQI 1: Worst channel quality (low SNR)
    - CQI 15: Best channel quality (high SNR)
    """
    # Define SNR thresholds for CQI mapping
    # These values can be adjusted based on specific system requirements
    min_snr = -10.0  # SNR value corresponding to CQI 1
    max_snr = 30.0   # SNR value corresponding to CQI 15
    
    # Clamp SNR to the defined range
    snr_clamped = max(min_snr, min(max_snr, snr_dB))
    
    # Normalize to [0,1] range
    normalized = (snr_clamped - min_snr) / (max_snr - min_snr)
    
    # Scale to CQI range [1,15] and round to nearest integer
    cqi = round(1 + normalized * 14)
    
    return cqi

def generate_user_request():
    """Generate a random user request from a predefined list with assigned labels."""
    # Define requests with their labels
    user_requests = [
        # High bandwidth, high reliability applications (eMBB)
        ("I want to watch 4K video", "eMBB"),
        ("I want to stream a live sports event in HD", "eMBB"),
        ("I need to participate in a video conference meeting", "eMBB"),
        ("I want to use augmented reality navigation", "eMBB"),
        ("I want to participate in a virtual reality gaming session", "eMBB"),
        ("I need to stream 8K video content", "eMBB"),
        ("I want to download a big game file", "eMBB"),
        ("I need AI-based video analytics", "eMBB"),
        ("I want to use holographic communication", "eMBB"),
        
        # Medium requirements (eMBB)
        ("I want to stream music while browsing social media", "eMBB"),
        ("I need to download large files", "eMBB"),
        ("I want to download a large software update", "eMBB"),
        ("I need to make a high-quality voice call", "eMBB"),
        ("I want to stream a webinar with interactive features", "eMBB"),
        ("I need to use cloud-based AI services for image processing", "eMBB"),
        ("I need real-time traffic updates for navigation", "eMBB"),
        ("I want to monitor my home security cameras remotely", "eMBB"),
        ("I want to remotely access my work computer", "eMBB"),
        ("I want to browse websites and check email", "eMBB"),
        ("I need to send text messages and use messaging apps", "eMBB"),
        ("I want to update my social media status", "eMBB"),
        ("I need to check weather forecasts", "eMBB"),
        ("I want to read news articles online", "eMBB"),
        ("I need to use maps for basic navigation", "eMBB"),
        ("I want to listen to low-quality audio streaming", "eMBB"),
        ("I need to sync my calendar and contacts", "eMBB"),
        
        # Lower requirements (URLLC)
        ("I need to control a robotic arm in real time", "URLLC"),
        ("I need my autonomous vehicle to communicate in real time", "URLLC"),  
        ("I need to participate in an online multiplayer game", "URLLC"),       
        ("I want to use remote surgery equipment", "URLLC"),
        ("I need to monitor IoT sensors in real-time", "URLLC"),
        ("I want to play competitive mobile games with ultra-low latency", "URLLC"),
        ("I need to synchronize multiple robots on a factory floor", "URLLC"),
        ("I need to monitor and control critical manufacturing processes in real-time", "URLLC"),  
        ("I need immediate machine shutdown capability for safety incidents", "URLLC"),       
        ("I need to control precision CNC machines with zero tolerance for delay", "URLLC"),
        ("I need to transmit real-time patient vital signs during critical care", "URLLC"),
        ("I need reliable connectivity for implanted medical devices", "URLLC"),       
        ("I need instant alerts for life-threatening patient conditions", "URLLC"),
        ("I need to control remote diagnostic equipment in rural clinics", "URLLC"),  
        ("I need emergency response coordination during a disaste", "URLLC"),       
        ("I need to deploy early warning systems for natural disasters", "URLLC"),
        ("I need reliable communication for firefighters inside buildings", "URLLC"),
        ("I need instant facial recognition for public security threats", "URLLC"),          
        ("I need microsecond-level latency for high-frequency tradin", "URLLC"),       
        ("I need real-time fraud detection for financial transactions", "URLLC"),
        ("I need to synchronize distributed financial ledgers instantly", "URLLC"),  
        ("I need to detect and isolate power grid faults instantly", "URLLC"),       
        ("I need to balance electrical load in real-time across microgrids", "URLLC"),
        ("I need to control critical infrastructure with zero downtime", "URLLC"),
        ("I need vehicle-to-vehicle collision avoidance systems", "URLLC"),                        
    ]
    
    # Select a random request and its label
    request, label = random.choice(user_requests)
    
    return request, label

def perform_ray_tracing(tx_position, rx_positions, buildings, tx_power_dBm=30):
    """Perform simplified ray tracing to calculate SNR and CQI at each receiver."""
    # Simulation parameters
    frequency = 2.4e9  # 2.4 GHz (in Hz)
    bandwidth = 20e6  # 20 MHz
    
    # Noise calculation
    thermal_noise_dBm = -174 + 10 * math.log10(bandwidth)  # -174 dBm/Hz + 10log10(bandwidth)
    noise_figure_dB = 8  # Typical receiver noise figure
    noise_floor_dBm = thermal_noise_dBm + noise_figure_dB
    
    results = []
    for rx_position in rx_positions:
        # Check if line of sight exists
        los = has_line_of_sight(tx_position, rx_position, buildings)
        
        # Calculate path loss
        path_loss_dB = calculate_path_loss(tx_position, rx_position, frequency, los)
        
        # Calculate received power
        rx_power_dBm = tx_power_dBm - path_loss_dB
        
        # Calculate SNR
        snr_dB = rx_power_dBm - noise_floor_dBm
        
        # Calculate CQI
        cqi = calculate_cqi(snr_dB)
        
        # Generate a random user request with its label
        user_request, request_label = generate_user_request()
        
        results.append({
            'position': rx_position,
            'snr_dB': snr_dB,
            'has_los': los,
            'rx_power_dBm': rx_power_dBm,
            'cqi': cqi,
            'user_request': user_request,
            'request_label': request_label
        })
    
    return results

def main():
    # File path
    file_path = r"C:\Users\Jingwen TONG\Desktop\我的文档\02_项目_202301\16-WirelessAgent-ChinaCom\Simulations\WirelessAgent_LangGraph\Knowledge_Base\HKUST_F.osm"
    
    try:
        # Parse buildings from OSM
        buildings = parse_osm_buildings(file_path)
        print(f"Found {len(buildings)} buildings in OSM file")
        
        if not buildings:
            print("No buildings found in the OSM file. Please check the file content.")
            return
        
        # Convert to Cartesian coordinates
        cartesian_buildings, transformer, (ref_lat, ref_lon) = convert_to_cartesian(buildings)
        
        # Find tallest building
        tallest_building = find_tallest_building(cartesian_buildings)
        if not tallest_building:
            print("No valid buildings found for analysis")
            return
        
        print(f"Tallest building height: {tallest_building['height']} meters")
        
        # Place TX at tallest building
        tx_position = place_tx(tallest_building)
        print(f"TX position: ({tx_position[0]:.2f}, {tx_position[1]:.2f}, {tx_position[2]:.2f}) m")
        
        # Calculate scene bounds
        all_nodes = [node for building in cartesian_buildings for node in building['nodes']]
        min_x = min(node[0] for node in all_nodes)
        max_x = max(node[0] for node in all_nodes)
        min_y = min(node[1] for node in all_nodes)
        max_y = max(node[1] for node in all_nodes)
        
        # Add some buffer to bounds
        buffer = max((max_x - min_x), (max_y - min_y)) * 0.1
        bounds = (min_x - buffer, max_x + buffer, min_y - buffer, max_y + buffer)
        
        # Generate random RX positions=======================================================================================================
        rx_positions = generate_rx_positions(bounds, 30, cartesian_buildings) # Number of RX positions is 30
        
        # Perform ray tracing
        results = perform_ray_tracing(tx_position, rx_positions, cartesian_buildings)
        
        # Save results to CSV=============================================================================================================
        output_path = 'ray_tracing_results.csv' # Output file path modifed by Jingwen TONG
        with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['RX_ID', 'X', 'Y', 'Z', 'SNR_dB', 'RX_Power_dBm', 'CQI', 'LOS', 'User_Request', 'Request_Label']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writeheader()
            for i, result in enumerate(results):
                pos = result['position']
                writer.writerow({
                    'RX_ID': i+1,
                    'X': f"{pos[0]:.2f}",
                    'Y': f"{pos[1]:.2f}",
                    'Z': f"{pos[2]:.2f}",
                    'SNR_dB': f"{result['snr_dB']:.2f}",
                    'RX_Power_dBm': f"{result['rx_power_dBm']:.2f}",
                    'CQI': result['cqi'],
                    'LOS': 1 if result['has_los'] else 0,
                    'User_Request': result['user_request'],
                    'Request_Label': result['request_label']
                })
        print(f"Results saved to {output_path}")
        
        # Visualize results with SNR
        plt.figure(figsize=(18, 13))
        
        # Plot buildings
        for building in cartesian_buildings:
            nodes = building['nodes']
            x = [node[0] for node in nodes]
            y = [node[1] for node in nodes]
            plt.fill(x, y, alpha=0.3, color='gray', edgecolor='black')
        
        # Plot TX position
        plt.scatter(tx_position[0], tx_position[1], color='red', s=200, marker='^', label='BS')
        
        # Plot RX positions with color based on SNR
        los_rx = [result for result in results if result['has_los']]
        nlos_rx = [result for result in results if not result['has_los']]
        
        if los_rx:
            los_x = [result['position'][0] for result in los_rx]
            los_y = [result['position'][1] for result in los_rx]
            los_snr = [result['snr_dB'] for result in los_rx]
            sc_los = plt.scatter(los_x, los_y, c=los_snr, cmap='viridis', s=100, marker='x', label='LOS User')
        
        if nlos_rx:
            nlos_x = [result['position'][0] for result in nlos_rx]
            nlos_y = [result['position'][1] for result in nlos_rx]
            nlos_snr = [result['snr_dB'] for result in nlos_rx]
            sc_nlos = plt.scatter(nlos_x, nlos_y, c=nlos_snr, cmap='viridis', s=100, marker='o', label='NLOS User')
        
        # Create colorbar for SNR values
        snr_values = [result['snr_dB'] for result in results]
        if snr_values:
            cbar = plt.colorbar()
        
        cbar.set_label('SNR (dB)', fontsize=24)  # 设置标题字体
        cbar.ax.tick_params(labelsize=24)        # 设置刻度标签字体
        plt.xlabel('X (m)', fontsize=24)
        plt.ylabel('Y (m)', fontsize=24)
        #plt.title('Ray Tracing Results - OpenStreetMap', fontsize=12)
        plt.legend(fontsize=24)
        plt.axis('equal')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.savefig('ray_tracing_map_center.png', dpi=300)
        plt.show()



        # Create CQI visualization
        plt.figure(figsize=(15, 12))
        
        # Plot buildings again for CQI visualization
        for building in cartesian_buildings:
            nodes = building['nodes']
            x = [node[0] for node in nodes]
            y = [node[1] for node in nodes]
            plt.fill(x, y, alpha=0.3, color='gray', edgecolor='black')
        
        # Plot TX position
        plt.scatter(tx_position[0], tx_position[1], color='red', s=150, marker='p', label='BS')
        
        # Plot RX positions with color based on CQI
        rx_x = [result['position'][0] for result in results]
        rx_y = [result['position'][1] for result in results]
        cqi_values = [result['cqi'] for result in results]
        
        sc = plt.scatter(rx_x, rx_y, c=cqi_values, cmap='viridis', vmin=1, vmax=15, s=30)
        
        # Create colorbar for CQI values
        cbar = plt.colorbar(sc, label='CQI (1-15)')
        cbar.set_ticks(range(1, 16))
        
        plt.xlabel('X (m)', fontsize=12)
        plt.ylabel('Y (m)', fontsize=12)
        plt.title('Channel Quality Indicator (CQI) Distribution')
        plt.axis('equal')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.savefig('cqi_distribution_map.png', dpi=300)
        
        # Output example results
        print("\nRX Positions, SNR, CQI, and User Requests (first 10 shown):")
        for i, result in enumerate(results[:10]):
            pos = result['position']
            snr = result['snr_dB']
            cqi = result['cqi']
            los = "LOS" if result['has_los'] else "NLOS"
            request = result['user_request']
            label = result['request_label']
            
            print(f"RX {i+1}: Position ({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}) m")
            print(f"    SNR: {snr:.2f} dB, CQI: {cqi}, {los}")
            print(f"    Request: \"{request}\" (Label: {label})")
            print("---")
        
        # Generate CQI statistics
        cqi_counts = {}
        for i in range(1, 16):
            cqi_counts[i] = len([r for r in results if r['cqi'] == i])
            
        print("\nCQI Distribution:")
        for cqi, count in cqi_counts.items():
            print(f"CQI {cqi}: {count} receivers ({count/len(results)*100:.1f}%)")
    
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()